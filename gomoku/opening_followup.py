"""Fixed-budget four-model opening audit, correction training and fresh evaluation."""
from pathlib import Path
import hashlib
import json
import time

import numpy as np
import torch

from .advanced_match_audit import audit
from .dagger import collect, save_records, records_from_file, heldout_canonical_keys
from .env import BatchBoard, black_center_action
from .history_aux import burn_in
from .midgame_data import canonical
from .models import load_policy, label_set_nll
from .train import atomic_json, atomic_torch

ROOT = Path(__file__).resolve().parent
RUN = ROOT / "runs" / "opening_followup_diverse_20261014"
BASELINE_RUN = ROOT / "runs" / "opening_followup_20261014"
SOURCE = ROOT / "runs" / "long_plan_f_20261009"
PATHS = {
    "fly": SOURCE / "structure_compare/fly_seed4810.pt",
    "gru": SOURCE / "f3/gru_seed4810.pt",
    "rnn": SOURCE / "structure_compare/rnn_seed4811.pt",
    "rewired": SOURCE / "structure_compare/rewired_seed4811.pt",
}
VALIDATION = 281000481
TEST_SEEDS = (282000481, 283000481)
COLLECTION = 284000481
GAMES, UPDATES, LR = 128, 100, 0.00005


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluation(path, tag, seed, force=True):
    target = RUN / f"{tag}_{seed}.json"
    if target.exists():
        result = json.loads(target.read_text(encoding="utf-8"))
        if (result["sha256"] != digest(path) or result["metrics"]["environment_seed"] != seed
                or result["metrics"]["force_black_center"] != force
                or not result["metrics"]["center_opponent_opening"]):
            raise RuntimeError("Saved evaluation does not match checkpoint/opening settings")
        return result
    cached = BASELINE_RUN / target.name
    if ("_before" in tag or "_raw" in tag) and cached.exists():
        result = json.loads(cached.read_text(encoding="utf-8"))
        if (result["sha256"] != digest(path) or result["metrics"]["force_black_center"] != force
                or not result["metrics"]["center_opponent_opening"]):
            raise RuntimeError("Baseline cache does not match experiment")
        atomic_json(target, result)
        return result
    model, saved = load_policy(path, torch.device("cpu"))
    row, replays = audit(model, saved["config"]["seed"], GAMES, True, seed,
                         force_center=force, center_opponent_opening=True)
    result = dict(checkpoint=str(path), sha256=digest(path), metrics=row, replays=replays)
    atomic_json(target, result)
    print(json.dumps(dict(event="evaluation", tag=tag, seed=seed, wins=row["wins"],
                         omissions=row["defense_omissions"], blocks=row["tactical_decisions"]["block"]["positions"],
                         black=row["by_color"]["black"]["wins"], white=row["by_color"]["white"]["wins"])), flush=True)
    return result


def replay_keys(results):
    reserved = set()
    for result in results:
        for replay in result["replays"]:
            board = BatchBoard(1, 12)
            for ply, action in enumerate(replay["moves"]):
                color = 1 if ply % 2 == 0 else -1
                reserved.add(canonical(board.board[0], color))
                board.place(np.array([action]), color)
    return reserved


def key(record):
    obs = record["obs"]
    color = 1 if obs[3, 0, 0] else -1
    return canonical(((obs[0] - obs[1]) * color).reshape(-1).astype(np.int8), color)


def verify_records(records):
    for record in records:
        board = BatchBoard(1, 12)
        moves = record["histories"][record["histories"] >= 0]
        for ply, action in enumerate(moves):
            board.place(np.array([action]), 1 if ply % 2 == 0 else -1)
        color = 1 if len(moves) % 2 == 0 else -1
        if not np.array_equal(board.observe(color)[0], record["obs"]):
            raise RuntimeError("History/observation mismatch")
        if not record["labels"].any() or np.any(record["labels"] & ~board.legal()[0]):
            raise RuntimeError("Invalid teaching labels")


def tensors(records, device):
    return dict(obs=torch.as_tensor(np.stack([r["obs"] for r in records]), dtype=torch.float32, device=device),
                labels=torch.as_tensor(np.stack([r["labels"] for r in records]), device=device),
                histories=torch.as_tensor(np.stack([r["histories"] for r in records]), dtype=torch.long, device=device))


def fine_tune(kind, fresh_records, legacy, device):
    destination = RUN / f"{kind}_candidate.pt"
    if destination.exists():
        return destination
    model, saved = load_policy(PATHS[kind], device)
    fresh, old = tensors(fresh_records, device), tensors(legacy, device)
    rng = np.random.default_rng(285000481)
    # Balance tactical categories so abundant blocking failures do not dominate.
    groups = [np.flatnonzero(np.array([r["category"] for r in fresh_records]) == c)
              for c in sorted({r["category"] for r in fresh_records})]
    schedule = [np.array([rng.choice(groups[rng.integers(len(groups))]) for _ in range(32)])
                for _ in range(UPDATES)]
    old_schedule = rng.integers(len(legacy), size=(UPDATES, 96))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0)
    model.train()
    opening_board = BatchBoard(8, 12)
    opening_obs = torch.from_numpy(opening_board.observe(1)).to(device)
    opening_labels = torch.zeros(8, 144, dtype=torch.bool, device=device)
    opening_labels[:, black_center_action(12)] = True
    progress = RUN / f"{kind}_progress.pt"
    done = 0
    if progress.exists():
        state = torch.load(progress, map_location=device, weights_only=True)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        done = state["completed_supervised_updates"]
    started = time.perf_counter()
    losses = []
    for update in range(done, UPDATES):
        optimizer.zero_grad(set_to_none=True)
        total = 0.0
        for data, ids, weight in ((old, old_schedule[update], 0.75), (fresh, schedule[update], 0.25)):
            ids = torch.as_tensor(ids, device=device)
            obs = data["obs"][ids]
            logits, _, _ = model.step(obs, burn_in(model, data["histories"][ids]),
                                      torch.zeros(len(ids), dtype=torch.bool, device=device))
            legal = (obs[:, 0] + obs[:, 1]).flatten(1) == 0
            loss = label_set_nll(logits, legal, data["labels"][ids]) * weight
            loss.backward()
            total += float(loss.detach())
        # Explicit opening convention supervision; shared start is not claimed as an unseen test board.
        logits, _, _ = model.step(opening_obs, model.initial(8, device),
                                  torch.ones(8, dtype=torch.bool, device=device))
        opening_loss = label_set_nll(logits, torch.ones_like(opening_labels), opening_labels) * 0.1
        opening_loss.backward()
        if not np.isfinite(total + float(opening_loss.detach())):
            raise RuntimeError("Nonfinite loss")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        losses.append(total)
        if (update + 1) % 25 == 0:
            state = dict(config=dict(saved["config"], initialization=str(PATHS[kind]),
                                     opening_supervision="black center", label_set_loss=True),
                         graph_sha256=saved["graph_sha256"],
                         model={k: v.detach().cpu() for k, v in model.state_dict().items()},
                         optimizer=optimizer.state_dict(), completed_updates=0,
                         completed_supervised_updates=update + 1)
            atomic_torch(progress, state)
            print(json.dumps(dict(event="training", kind=kind, update=update + 1,
                                  loss=total, seconds=round(time.perf_counter()-started, 1))), flush=True)
    atomic_torch(destination, state)
    atomic_json(RUN / f"{kind}_training.json", dict(updates=UPDATES, learning_rate=LR,
                fresh_records=len(fresh_records), legacy_records=len(legacy), quotas=[96, 32],
                opening_weight=0.1, losses=losses, seconds=time.perf_counter()-started))
    return destination


def aggregate(results):
    rows = [r["metrics"] for r in results]
    blocks = sum(r["tactical_decisions"]["block"]["positions"] for r in rows)
    omissions = sum(r["defense_omissions"] for r in rows)
    return dict(games=sum(r["games"] for r in rows), wins=sum(r["wins"] for r in rows),
                omissions=omissions, blocks=blocks, omission_rate=omissions/blocks if blocks else None,
                safe_forks=sum(r["actual_safe_forks_created"] for r in rows),
                black_wins=sum(r["by_color"]["black"]["wins"] for r in rows),
                white_wins=sum(r["by_color"]["white"]["wins"] for r in rows))


def improves(before, after):
    return (before["omission_rate"] is not None and after["omission_rate"] is not None
            and after["wins"] > before["wins"]
            and after["omission_rate"] <= before["omission_rate"]
            and after["omissions"] <= before["omissions"]
            and after["safe_forks"] >= before["safe_forks"])


def main():
    RUN.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    protocol = dict(games_per_seed=GAMES, validation_seed=VALIDATION, test_seeds=list(TEST_SEEDS),
                    collection_seed=COLLECTION, updates=UPDATES, learning_rate=LR,
                    correction_selection="first certified mistake outside reserved evaluation boards per lost game",
                    collection_opponent="hard: 85% tactical / 15% random; evaluation remains tactical",
                    source_hashes={k: digest(p) for k, p in PATHS.items()},
                    opponent="built-in tactical heuristic; not an external expert engine",
                    opening="both players black center; raw comparison disables only learner forcing",
                    selection="fixed final 100 updates; no checkpoint search; validation and test must improve wins, preserve omission rate/count and forks",
                    test_scope="new opponent RNG seeds; common opening is intentionally shared; baseline evaluation prefixes excluded from tactical training")
    protocol_path = RUN / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise RuntimeError("Changed experiment protocol")
    atomic_json(protocol_path, protocol)
    before = {}
    raw = {}
    for kind, path in PATHS.items():
        raw[kind] = evaluation(path, f"{kind}_raw", VALIDATION, False)
        before[kind] = [evaluation(path, f"{kind}_before", seed) for seed in (VALIDATION, *TEST_SEEDS)]
    reserved = replay_keys([r for rows in before.values() for r in rows])
    heldout = ROOT / "runs/offense_e_20261008/data/fresh_test_starts.npz"
    reserved |= heldout_canonical_keys([heldout])
    legacy = []
    for path in (ROOT / "runs/midgame_20261007/data/baseline_aux.npz",
                 SOURCE / "defense_round3/gru_seed4810_dagger.npz",
                 ROOT / "runs/offense_e_20261008/data/gru_seed4810_attack.npz"):
        legacy.extend(r for r in records_from_file(path) if key(r) not in reserved)
    verify_records(legacy)
    print(json.dumps(dict(event="reserved_and_legacy_ready", reserved_keys=len(reserved),
                          legacy_records=len(legacy))), flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = dict(status="training", protocol=protocol, models={}, reserved_keys=len(reserved), legacy_records=len(legacy))
    for kind, path in PATHS.items():
        data_path = RUN / f"{kind}_corrections.npz"
        if data_path.exists():
            fresh = records_from_file(data_path)
        else:
            model, _ = load_policy(path, device)
            fresh, stats = collect(model, GAMES, COLLECTION, device, excluded=reserved,
                                   round_index=1, center_opponent_opening=True, skip_excluded_departures=True,
                                   opponent_kind="hard")
            del model
            if fresh:
                save_records(data_path, fresh)
            atomic_json(RUN / f"{kind}_collection.json", stats)
            print(json.dumps(dict(event="collection", kind=kind, **stats)), flush=True)
        verify_records(fresh)
        if any(key(r) in reserved for r in fresh):
            raise RuntimeError("Evaluation board leaked into correction data")
        candidate = fine_tune(kind, fresh, legacy, device) if fresh else path
        after = [evaluation(candidate, f"{kind}_after", seed) for seed in (VALIDATION, *TEST_SEEDS)]
        bval, aval = aggregate(before[kind][:1]), aggregate(after[:1])
        btest, atest = aggregate(before[kind][1:]), aggregate(after[1:])
        accepted = improves(bval, aval) and improves(btest, atest) and candidate != path
        result["models"][kind] = dict(raw_validation=aggregate([raw[kind]]),
                    before_validation=bval, after_validation=aval, before_test=btest, after_test=atest,
                    candidate=str(candidate), adopted=accepted, selected=str(candidate if accepted else path),
                    fresh_records=len(fresh), correction_eval_overlap=0)
        atomic_json(RUN / "summary.json", result)
    if any(digest(PATHS[k]) != protocol["source_hashes"][k] for k in PATHS):
        raise RuntimeError("Original checkpoint changed")
    result["status"] = "complete"
    result["original_hashes_unchanged"] = True
    atomic_json(RUN / "summary.json", result)
    print(json.dumps(dict(event="complete", selected={k: r["adopted"] for k, r in result["models"].items()})), flush=True)


if __name__ == "__main__":
    main()

"""Two fixed stages, four original models, defense preservation, unopened final seeds."""
from pathlib import Path
import hashlib
import json
import time

import numpy as np
import torch

from .advanced_match_audit import audit
from .balanced_curriculum import STAGES, sample_batch, preservation_loss
from .dagger import records_from_file, heldout_canonical_keys, save_records
from .env import BatchBoard
from .history_aux import burn_in
from .models import load_policy, label_set_nll
from .opening_followup import PATHS, SOURCE, key, verify_records, tensors, replay_keys, aggregate, improves
from .train import atomic_json, atomic_torch

ROOT = Path(__file__).resolve().parent
RUN = ROOT / "runs/balanced_curriculum_20261014"
PREVIOUS = ROOT / "runs/opening_followup_diverse_20261014"
VALIDATION = 291000481
FINAL_SEEDS = (292000481, 293000481)
VAL_GAMES, FINAL_GAMES = 64, 128
UPDATES_PER_STAGE, LR, ANCHOR_WEIGHT = 50, 0.00002, 0.5


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def white_openings(seed, games):
    if games % 2:
        raise ValueError("Paired color evaluation requires an even game count")
    candidates = np.array([r*12+c for r in range(3, 8) for c in range(3, 8) if r*12+c != 65])
    rng = np.random.default_rng(seed)
    paired = np.tile(candidates, (games//2+len(candidates)-1)//len(candidates))[:games//2]
    rng.shuffle(paired)
    return np.repeat(paired, 2)


def evaluate(path, kind, stage, seed, games, mode):
    target = RUN / f"{kind}_{stage}_{mode}_{seed}.json"
    if target.exists():
        saved = json.loads(target.read_text())
        if saved["sha256"] != sha(path):
            raise RuntimeError("Evaluation checkpoint changed")
        return saved
    model, checkpoint = load_policy(path, torch.device("cpu"))
    openings = white_openings(seed, games) if mode == "diverse" else None
    metrics, replays = audit(model, checkpoint["config"]["seed"], games, True, seed,
                             force_center=True, center_opponent_opening=True, opening_white_moves=openings)
    result = dict(checkpoint=str(path), sha256=sha(path), mode=mode, metrics=metrics, replays=replays)
    atomic_json(target, result)
    print(json.dumps(dict(event="evaluation", kind=kind, stage=stage, mode=mode, seed=seed,
                         wins=metrics["wins"], omissions=metrics["defense_omissions"],
                         blocks=metrics["tactical_decisions"]["block"]["positions"])), flush=True)
    return result


def data_records(kind, reserved):
    cache = RUN / f"{kind}_data.npz"
    source_path = RUN / f"{kind}_sources.json"
    if cache.exists():
        records = records_from_file(cache)
        sources = json.loads(source_path.read_text())
        for record, source in zip(records, sources):
            record["source"] = source
        if len(records) != len(sources):
            raise RuntimeError("Data provenance count mismatch")
        return records
    records = []
    for path, source in ((ROOT / "runs/midgame_20261007/data/baseline_aux.npz", "base"),
                         (SOURCE / "defense_round3/gru_seed4810_dagger.npz", "guard"),
                         (PREVIOUS / f"{kind}_corrections.npz", "fresh")):
        for row in records_from_file(path):
            if key(row) not in reserved:
                row["source"] = source
                records.append(row)
    verify_records(records)
    save_records(cache, records)
    atomic_json(source_path, [r["source"] for r in records])
    return records


def train_stage(kind, start_path, stage_index, records, device):
    stage_name, quotas = STAGES[stage_index]
    destination = RUN / f"{kind}_{stage_name}.pt"
    if destination.exists():
        return destination
    model, saved = load_policy(start_path, device)
    reference, _ = load_policy(PATHS[kind], device)
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    reference.eval()
    data = tensors(records, device)
    categories = torch.as_tensor([r["category"] for r in records], device=device)
    rng = np.random.default_rng(294000481+stage_index)
    schedules = [sample_batch(records, quotas, rng)[0] for _ in range(UPDATES_PER_STAGE)]
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0)
    model.train()
    log = []
    started = time.perf_counter()
    progress = RUN / f"{kind}_{stage_name}_progress.pt"
    done = 0
    if progress.exists():
        state = torch.load(progress, map_location=device, weights_only=True)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        done = state["completed_supervised_updates_in_stage"]
        log = state["curve"]
    for update in range(done, UPDATES_PER_STAGE):
        ids = torch.as_tensor(schedules[update], device=device)
        obs, labels, histories = data["obs"][ids], data["labels"][ids], data["histories"][ids]
        legal = (obs[:, 0]+obs[:, 1]).flatten(1) == 0
        resets = torch.zeros(len(ids), dtype=torch.bool, device=device)
        optimizer.zero_grad(set_to_none=True)
        logits, _, _ = model.step(obs, burn_in(model, histories), resets)
        with torch.no_grad():
            teacher_logits, _, _ = reference.step(obs, burn_in(reference, histories), resets)
        supervised = label_set_nll(logits, legal, labels)
        anchor, anchored = preservation_loss(logits, teacher_logits, legal, labels, categories[ids])
        loss = supervised+ANCHOR_WEIGHT*anchor
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite curriculum loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        log.append(dict(update=update+1, supervised=float(supervised.detach()),
                        preservation=float(anchor.detach()), anchored=anchored))
        if (update+1) % 25 == 0:
            state = dict(config=dict(saved["config"], curriculum_stage=stage_name,
                                     initialization=str(start_path), quotas=quotas,
                                     preservation_reference=str(PATHS[kind]), preservation_weight=ANCHOR_WEIGHT),
                         graph_sha256=saved["graph_sha256"],
                         model={k: v.detach().cpu() for k,v in model.state_dict().items()},
                         optimizer=optimizer.state_dict(), completed_updates=0,
                         completed_supervised_updates=(stage_index*UPDATES_PER_STAGE+update+1),
                         completed_supervised_updates_in_stage=update+1, curve=log)
            atomic_torch(progress, state)
            print(json.dumps(dict(event="training", kind=kind, stage=stage_name, update=update+1,
                                  supervised=log[-1]["supervised"], preservation=log[-1]["preservation"],
                                  anchored=anchored)), flush=True)
    atomic_torch(destination, state)
    atomic_json(RUN / f"{kind}_{stage_name}_training.json", dict(curve=log, quotas=quotas,
                black_quota=64, white_quota=64, training_seconds=time.perf_counter()-started))
    del model, reference, optimizer, data
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return destination


def main():
    RUN.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    protocol = dict(stages=[dict(name=name, quotas=quotas, updates=UPDATES_PER_STAGE) for name,quotas in STAGES],
                    learning_rate=LR, preservation_weight=ANCHOR_WEIGHT, color_quota=[64,64],
                    validation_seed=VALIDATION, validation_games_per_mode=VAL_GAMES,
                    final_seeds=list(FINAL_SEEDS), final_games_per_mode=FINAL_GAMES,
                    modes=["empty", "diverse"], source_hashes={k:sha(p) for k,p in PATHS.items()},
                    adoption="fixed final forcing stage; wins increase and defense rate/count do not rise in EACH mode, on validation and final tests; safe forks do not decrease",
                    teacher="existing immediate/fork/forcing-chain labels; no external expert engine",
                    final_test="run only after all four curricula finish; no final-test checkpoint selection")
    path = RUN / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise RuntimeError("Curriculum protocol changed")
    atomic_json(path, protocol)
    baseline = {}
    reserved_results = []
    for kind, path in PATHS.items():
        baseline[kind] = {mode:evaluate(path,kind,"original",VALIDATION,VAL_GAMES,mode)
                          for mode in protocol["modes"]}
        reserved_results.extend(baseline[kind].values())
    for path in PREVIOUS.glob("*_*.json"):
        d = json.loads(path.read_text())
        if isinstance(d, dict) and "replays" in d:
            reserved_results.append(d)
    reserved = replay_keys(reserved_results)
    reserved |= heldout_canonical_keys([ROOT / "runs/offense_e_20261008/data/fresh_test_starts.npz"])
    atomic_json(RUN / "data_partition.json", dict(reserved_canonical_boards=len(reserved),
                scope="prior evaluation and current baseline-validation prefixes excluded; final tests remain unopened; common openings intentionally shared"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = dict(status="training", protocol=protocol, models={})
    candidates = {}
    for kind, original in PATHS.items():
        records = data_records(kind, reserved)
        if any(key(r) in reserved for r in records):
            raise RuntimeError("Reserved board leaked into curriculum")
        ledger = {source:{str(c):{color:sum(r["source"]==source and r["category"]==c and
                   bool(r["obs"][3,0,0])==(color=="black") for r in records) for color in ("black","white")}
                   for c in range(1,7)} for source in ("base","guard","fresh")}
        atomic_json(RUN / f"{kind}_data_audit.json", dict(records=len(records), ledger=ledger,
                    reserved_overlap=0, bad_observations=0, illegal_labels=0))
        print(json.dumps(dict(event="data_ready",kind=kind,records=len(records))),flush=True)
        stages = {}
        current = original
        for stage_index,(stage_name,_) in enumerate(STAGES):
            current = train_stage(kind,current,stage_index,records,device)
            stages[stage_name] = {mode:evaluate(current,kind,stage_name,VALIDATION,VAL_GAMES,mode)
                                   for mode in protocol["modes"]}
        candidates[kind] = current
        result["models"][kind] = dict(baseline_validation={m:aggregate([v]) for m,v in baseline[kind].items()},
                    stage_validation={stage:{m:aggregate([v]) for m,v in rows.items()} for stage,rows in stages.items()},
                    candidate=str(current), data_records=len(records))
        atomic_json(RUN / "summary.json", result)
    print(json.dumps(dict(event="all_training_complete_final_tests_begin")),flush=True)
    for kind, original in PATHS.items():
        before, after = {}, {}
        for mode in protocol["modes"]:
            before[mode] = aggregate([evaluate(original,kind,"original",seed,FINAL_GAMES,mode) for seed in FINAL_SEEDS])
            after[mode] = aggregate([evaluate(candidates[kind],kind,"forcing",seed,FINAL_GAMES,mode) for seed in FINAL_SEEDS])
        row = result["models"][kind]
        valid = all(improves(row["baseline_validation"][m],row["stage_validation"]["forcing"][m]) for m in protocol["modes"])
        adopted = valid and all(improves(before[m],after[m]) for m in protocol["modes"])
        row.update(before_test=before,after_test=after,adopted=adopted,
                   selected=str(candidates[kind] if adopted else original))
        atomic_json(RUN / "summary.json", result)
    if any(sha(PATHS[k])!=protocol["source_hashes"][k] for k in PATHS):
        raise RuntimeError("Original checkpoint changed")
    result.update(status="complete",original_hashes_unchanged=True)
    atomic_json(RUN / "summary.json",result)
    print(json.dumps(dict(event="complete",adopted={k:r["adopted"] for k,r in result["models"].items()})),flush=True)


if __name__ == "__main__":
    main()

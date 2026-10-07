"""Sequential actor-path training, held-in rate selection, and multi-graph comparison."""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from minigrid.core.world_object import Ball
from torch import nn
from torch.nn import functional as F

from evaluate_minigrid_learned_relation import evaluate as autonomous
from minigrid_compositional_policy import VisualRelations
from minigrid_learned_relation import (
    LearnedRelation,
    evaluate,
    features,
    matched_hidden,
    pretrain_comparator,
    select,
)
from minigrid_memory_diagnostic import build_model
from minigrid_memory_pilot import (
    GRAPH_DIR,
    ROOT,
    masked_cue_observation,
    parameter_count,
    sha256,
    tensor_observation,
)
from minigrid_relation_data import (
    EXCLUDED_JUNCTION_X,
    VARIANTS,
    make_env,
    reset_variant,
)

OUT = ROOT / "data/minigrid_relation_followup"
OLD_DETECTOR = ROOT / "data/minigrid_compositional/compositional_rnn_seed401_detector501.pt"
ACTOR_PATH = ROOT / "data/minigrid_relation_curriculum/relation_variable_control_rnn_1000_seed401.pt"
KINDS = ("fly", "rewired", "rnn", "gru")


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_navigation(device):
    actor = build_model("rnn", 1000, 401, device)
    actor.load_state_dict(torch.load(ACTOR_PATH, map_location=device, weights_only=False)["model_state"])
    actor.eval()
    return actor


def load_detector(path, device):
    detector = VisualRelations().to(device)
    detector.load_state_dict(torch.load(path, map_location=device, weights_only=False)["detector_state"])
    detector.eval()
    return detector


@torch.no_grad()
def collect(actor, detector, *, size, random_length, start_seed, base_count, device):
    """Record the deployed explicit policy's observations; labels never control navigation."""
    probe = make_env(size, random_length=random_length)
    accepted, candidate = [], start_seed
    try:
        while len(accepted) < base_count:
            _, meta = reset_variant(probe, candidate)
            if meta["junction_x"] not in EXCLUDED_JUNCTION_X:
                accepted.append(candidate)
            candidate += 1
    finally:
        probe.close()
    envs, observations, rows = [], [], []
    for seed in accepted:
        for cue_flip, target_flip in VARIANTS:
            env = make_env(size, random_length=random_length)
            observation, meta = reset_variant(env, seed, cue_flip=cue_flip, target_flip=target_flip)
            base = env.unwrapped
            top = base.grid.get(meta["junction_x"], base.height // 2 - 2)
            rows.append({"observations": [], "actions": [], "cue_frames": [], "branch_frames": [],
                         "upper_frames": [], "branch_index": None, "seen_cue": False,
                         "meta": meta, "seed": seed, "upper_label": int(not isinstance(top, Ball))})
            envs.append(env)
            observations.append(observation)
    states = actor.initial_state(len(rows), device)
    matrix = actor.recurrent_matrix()
    remembered = [None] * len(rows)
    done = [False] * len(rows)
    successes = [False] * len(rows)
    try:
        while not all(done):
            active = [index for index, flag in enumerate(done) if not flag]
            indices = torch.tensor(active, device=device)
            images, directions = tensor_observation([observations[i] for i in active], device)
            logits, _, next_state = actor.step(images, directions, states[indices], matrix)
            cue_logits, branch_logits, upper_logits = detector(images, directions)
            cue_predictions = cue_logits.argmax(dim=-1).cpu().tolist()
            branch_predictions = branch_logits.argmax(dim=-1).cpu().tolist()
            upper_predictions = upper_logits.argmax(dim=-1).cpu().tolist()
            actions = logits.argmax(dim=-1).cpu().tolist()
            states[indices] = next_state
            for local, index in enumerate(active):
                row, env = rows[index], envs[index]
                base = env.unwrapped
                visible = not np.array_equal(observations[index]["image"], masked_cue_observation(env)["image"])
                true_branch = (tuple(base.agent_pos) == (row["meta"]["junction_x"], base.height // 2)
                               and base.agent_dir == 0)
                cue_label = 1 + int(row["meta"]["cue_type"] == "Key") if visible else 0
                row["seen_cue"] |= visible
                if true_branch and row["branch_index"] is None:
                    row["branch_index"] = len(row["actions"])
                    row["cue_seen_before_branch"] = row["seen_cue"]
                row["observations"].append(observations[index])
                row["cue_frames"].append(cue_label)
                row["branch_frames"].append(int(true_branch))
                row["upper_frames"].append(row["upper_label"])
                if remembered[index] is None and cue_predictions[local] > 0:
                    remembered[index] = cue_predictions[local] - 1
                action = actions[local]
                if branch_predictions[local] == 1 and remembered[index] is not None:
                    action = int(upper_predictions[local] != remembered[index])
                row["actions"].append(action)
                observation, reward, terminated, truncated, _ = env.step(action)
                observations[index] = observation
                if terminated or truncated:
                    done[index] = True
                    successes[index] = reward > 0
    finally:
        for env in envs:
            env.close()
    for row in rows:
        row["images"], row["directions"] = tensor_observation(row.pop("observations"), torch.device("cpu"))
        row["actions"] = torch.tensor(row["actions"])
    # Retain complete groups for the four-way metric. Unusable groups remain in detector training.
    retained = []
    for start in range(0, len(rows), 4):
        group = rows[start:start + 4]
        if all(row["branch_index"] is not None and row["cue_seen_before_branch"] for row in group):
            retained.extend(group)
    frames = {"images": torch.cat([row["images"] for row in rows]),
              "directions": torch.cat([row["directions"] for row in rows]),
              "cue": torch.tensor([x for row in rows for x in row["cue_frames"]]),
              "branch": torch.tensor([x for row in rows for x in row["branch_frames"]]),
              "upper": torch.tensor([x for row in rows for x in row["upper_frames"]])}
    stats = {"size": size, "random_length": random_length, "start_seed": start_seed,
             "scanned_seeds": candidate - start_seed, "base_episodes": base_count,
             "retained_base_episodes": len(retained) // 4, "frames": len(frames["cue"]),
             "collection_success": sum(successes) / len(rows),
             "junction_x": sorted({row["meta"]["junction_x"] for row in rows}),
             "mean_branch_index": statistics.mean(row["branch_index"] for row in retained)}
    if any(x in (7, 13, 17) for x in stats["junction_x"]):
        raise RuntimeError("Held-out junction leaked into collected training/validation data")
    return retained, frames, stats


@torch.no_grad()
def detector_metrics(detector, data, device):
    detector.eval()
    cue_preds, branch_preds, upper_preds = [], [], []
    losses = []
    for indices in torch.arange(len(data["cue"])).split(1024):
        batch = select(data, indices, device)
        cue, branch, upper = detector(batch["images"], batch["directions"])
        cue_preds.append(cue.argmax(dim=-1).cpu())
        branch_preds.append(branch.argmax(dim=-1).cpu())
        upper_preds.append(upper.argmax(dim=-1).cpu())
        losses.append(float(F.cross_entropy(cue, batch["cue"], reduction="sum")
                            + F.cross_entropy(branch, batch["branch"], reduction="sum")))
    cue, branch, upper = torch.cat(cue_preds), torch.cat(branch_preds), torch.cat(upper_preds)
    truth = data["branch"].bool()
    positive = branch.bool()
    tp, fp, fn = int((truth & positive).sum()), int((~truth & positive).sum()), int((truth & ~positive).sum())
    return {"cue_accuracy": float(cue.eq(data["cue"]).float().mean()),
            "branch_precision": tp / max(1, tp + fp), "branch_recall": tp / max(1, tp + fn),
            "branch_f1": 2 * tp / max(1, 2 * tp + fp + fn),
            "branch_false_positive_frames": fp, "branch_positive_frames": int(truth.sum()),
            "upper_accuracy": float(upper[truth].eq(data["upper"][truth]).float().mean()),
            "loss": sum(losses) / len(cue), "frames": len(cue)}


def prepare(args, device):
    actor = load_navigation(device)
    old = load_detector(OLD_DETECTOR, device)
    raw = {}
    started = time.perf_counter()
    for split, start, count in (("train", 40_000_000, args.train_base),
                                ("validation", 42_000_000, args.validation_base)):
        parts, all_rows, stats = [], [], []
        for size in (7, 11, 13, 17):
            rows, frames, item = collect(actor, old, size=size, random_length=size == 17,
                                        start_seed=start + size * 100_000, base_count=count, device=device)
            parts.append(frames)
            all_rows.extend(rows)
            stats.append(item)
            print(json.dumps({"split": split, **item}), flush=True)
        raw[split] = {"rows": all_rows, "frames": {key: torch.cat([part[key] for part in parts])
                                                   for key in parts[0]}, "stats": stats}
    torch.save(raw, args.output_dir / "actor_paths.pt")
    detector = load_detector(OLD_DETECTOR, device)
    before = detector_metrics(detector, raw["validation"]["frames"], device)
    torch.manual_seed(801)
    optimizer = torch.optim.AdamW(detector.parameters(), lr=0.0003)
    train = raw["train"]["frames"]
    generator = torch.Generator().manual_seed(801)
    best, best_score, selected_epoch, history = None, (-1.0, -1.0), 0, []
    for epoch in range(1, 9):
        detector.train()
        for indices in torch.randperm(len(train["cue"]), generator=generator).split(512):
            batch = select(train, indices, device)
            cue, branch, upper = detector(batch["images"], batch["directions"])
            loss = F.cross_entropy(cue, batch["cue"], weight=torch.tensor([1., 4., 4.], device=device))
            loss = loss + F.cross_entropy(branch, batch["branch"], weight=torch.tensor([1., 4.], device=device))
            positive = batch["branch"] == 1
            if positive.any():
                loss = loss + F.cross_entropy(upper[positive], batch["upper"][positive])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        metrics = detector_metrics(detector, raw["validation"]["frames"], device)
        score = ((metrics["cue_accuracy"] + metrics["branch_f1"] + metrics["upper_accuracy"]) / 3,
                 -metrics["loss"])
        history.append({"epoch": epoch, "validation": metrics})
        if score > best_score:
            best, best_score, selected_epoch = copy.deepcopy(detector.state_dict()), score, epoch
        print(json.dumps(history[-1]), flush=True)
    detector.load_state_dict(best)
    after = detector_metrics(detector, raw["validation"]["frames"], device)
    torch.save({"detector_state": best}, args.output_dir / "detector.pt")
    bundle = {split: features(raw[split]["rows"], detector, device) for split in raw}
    torch.save(bundle, args.output_dir / "memory_data.pt")
    result = {"before": before, "after": after, "selected_epoch": selected_epoch,
              "history": history, "collections": {key: item["stats"] for key, item in raw.items()},
              "detector_seed": 801, "duration_seconds": time.perf_counter() - started}
    save_json(args.output_dir / "stage1.json", result)


def train_memory(kind, seed, rewire_seed, rate, train, validation, device, epochs, output_dir,
                 fixed_zero_edges=False):
    torch.manual_seed(0)
    target = parameter_count(LearnedRelation("fly", 1000, 0))
    hidden = matched_hidden(target, kind)
    torch.manual_seed(seed)
    model = LearnedRelation(kind, hidden, seed, rewire_seed).to(device)
    if fixed_zero_edges:
        with torch.no_grad():
            model.edge_values.zero_()
        model.edge_values.requires_grad_(False)
    comparator_accuracy = pretrain_comparator(model, train, device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=rate)
    generator = torch.Generator().manual_seed(seed + 1000)
    best, best_score, best_epoch, history = None, (-1., -1., -1., -float("inf")), 0, []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for indices in torch.randperm(len(train["turn"]), generator=generator).split(128):
            batch = select(train, indices, device)
            turn, cue = model(batch["cue"], batch["branch"], batch["upper"])
            loss = F.cross_entropy(turn, batch["turn"]) + F.cross_entropy(cue, batch["cue_label"])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            losses.append(float(loss.detach()))
        metrics = evaluate(model, validation, device)
        score = (metrics["all_four_success"], metrics["turn_accuracy"], metrics["cue_accuracy"], -metrics["loss"])
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "validation": metrics})
        if score > best_score:
            best, best_score, best_epoch = copy.deepcopy(model.state_dict()), score, epoch
    model.load_state_dict(best)
    name = f"{kind}_graph{rewire_seed}_seed{seed}_lr{rate:g}"
    if fixed_zero_edges:
        name += "_noedges"
    row = {"name": name, "kind": kind, "seed": seed, "rewire_seed": rewire_seed,
           "learning_rate": rate, "hidden": model.hidden, "parameters": parameter_count(model),
           "fixed_zero_edges": fixed_zero_edges,
           "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
           "selected_epoch": best_epoch, "validation": evaluate(model, validation, device),
           "comparator_train_accuracy": comparator_accuracy, "history": history}
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": best, "result": row}, output_dir / f"{name}.pt")
    save_json(output_dir / f"{name}.json", row)
    print(json.dumps({key: row[key] for key in ("name", "selected_epoch", "validation")}), flush=True)
    return row


def tune(args, device):
    data = torch.load(args.output_dir / "memory_data.pt", weights_only=False)
    rows = []
    for rate in (0.0003, 0.001, 0.003):
        for seed in (701, 702):
            for kind in KINDS:
                rows.append(train_memory(kind, seed, 0, rate, data["train"], data["validation"],
                                         device, args.epochs, args.output_dir / "tuning"))
    def score(group):
        return tuple(statistics.mean(row["validation"][key] * (-1 if key == "loss" else 1)
                                     for row in group) for key in ("all_four_success", "turn_accuracy", "cue_accuracy", "loss"))
    selected, candidates = {}, {}
    for family, kinds in (("fly_pair", ("fly", "rewired")), ("rnn", ("rnn",)), ("gru", ("gru",))):
        values = {rate: score([row for row in rows if row["kind"] in kinds and row["learning_rate"] == rate])
                  for rate in (0.0003, 0.001, 0.003)}
        winner = max(values, key=values.get)
        for kind in kinds:
            selected[kind] = winner
        candidates[family] = values
    save_json(args.output_dir / "stage2.json", {"selected_rates": selected, "candidates": candidates,
                                                "tuning_seeds": [701, 702], "runs": rows})
    print(json.dumps({"selected_rates": selected}), flush=True)


def graph_audit():
    records = []
    with np.load(GRAPH_DIR / "malecns_cx_1000.npz") as source:
        pre, post, ids = source["pre_index"], source["post_index"], source["body_ids"]
        for graph_seed in range(5):
            path = GRAPH_DIR / f"malecns_cx_1000_rewired_seed{graph_seed}.npz"
            with np.load(path) as graph:
                assert np.array_equal(ids, graph["body_ids"])
                assert np.array_equal(np.bincount(pre, minlength=1000), np.bincount(graph["pre_index"], minlength=1000))
                assert np.array_equal(np.bincount(post, minlength=1000), np.bincount(graph["post_index"], minlength=1000))
                pairs = set(zip(graph["pre_index"].tolist(), graph["post_index"].tolist()))
                assert len(pairs) == len(pre) and not any(a == b for a, b in pairs)
            for initialization_seed in range(711, 716):
                torch.manual_seed(initialization_seed)
                real = LearnedRelation("fly", 1000, initialization_seed)
                torch.manual_seed(initialization_seed)
                rewired = LearnedRelation("rewired", 1000, initialization_seed, graph_seed)
                assert all(torch.equal(a, b) for a, b in zip(real.parameters(), rewired.parameters()))
            records.append({"rewire_seed": graph_seed, "path": str(path), "sha256": sha256(path),
                            "degree_preserved": True, "paired_parameters_identical": True,
                            "paired_initialization_seeds": list(range(711, 716))})
    return records


def run(args, device):
    audit = graph_audit()
    data = torch.load(args.output_dir / "memory_data.pt", weights_only=False)
    rates = json.loads((args.output_dir / "stage2.json").read_text(encoding="utf-8"))["selected_rates"]
    rows = []
    for seed in range(711, 716):
        for kind in ("fly", "rnn", "gru"):
            rows.append(train_memory(kind, seed, 0, rates[kind], data["train"], data["validation"],
                                     device, args.epochs, args.output_dir / "main"))
        for graph_seed in range(5):
            rows.append(train_memory("rewired", seed, graph_seed, rates["rewired"],
                                     data["train"], data["validation"], device, args.epochs, args.output_dir / "main"))
    save_json(args.output_dir / "stage3_training.json", {"graph_audit": audit, "seeds": list(range(711, 716)),
                                                         "runs": rows})


def evaluate_final(args, device):
    training = json.loads((args.output_dir / "stage3_training.json").read_text(encoding="utf-8"))
    actor = load_navigation(device)
    detector = load_detector(args.output_dir / "detector.pt", device)
    output = []
    path = args.output_dir / "stage3_autonomous.json"
    existing = json.loads(path.read_text(encoding="utf-8"))["runs"] if path.exists() else []
    by_name = {row["name"]: row for row in existing}
    for row in training["runs"]:
        if row["name"] in by_name:
            output.append(by_name[row["name"]])
            continue
        checkpoint = torch.load(args.output_dir / "main" / f"{row['name']}.pt", map_location=device, weights_only=False)
        model = LearnedRelation(row["kind"], row["hidden"], row["seed"], row["rewire_seed"]).to(device)
        model.load_state_dict(checkpoint["model_state"])
        metrics = {str(size): autonomous(actor, detector, model, size=size,
                                         start_seed=44_000_000 + size * 100_000,
                                         base_count=args.final_base, device=device) for size in (9, 15, 19)}
        result = {key: row[key] for key in ("name", "kind", "seed", "rewire_seed", "learning_rate")}
        result["sizes"] = metrics
        output.append(result)
        save_json(path, {"base_per_size": args.final_base, "start_seed": 44_000_000, "runs": output})
        print(json.dumps(result), flush=True)


def controls(args, device):
    training = json.loads((args.output_dir / "stage3_training.json").read_text(encoding="utf-8"))
    actor = load_navigation(device)
    detector = load_detector(args.output_dir / "detector.pt", device)
    results = []
    for row in training["runs"]:
        modes = []
        if row["seed"] == 711 and (row["kind"] != "rewired" or row["rewire_seed"] == 0):
            modes.extend(("mask_cue", "reset_at_branch"))
        if row["kind"] == "fly":
            modes.append("zero_recurrence")
        for mode in modes:
            checkpoint = torch.load(args.output_dir / "main" / f"{row['name']}.pt", map_location=device, weights_only=False)
            model = LearnedRelation(row["kind"], row["hidden"], row["seed"], row["rewire_seed"]).to(device)
            model.load_state_dict(checkpoint["model_state"])
            if mode == "zero_recurrence":
                with torch.no_grad():
                    model.edge_values.zero_()
            metrics = autonomous(actor, detector, model, size=19, start_seed=45_900_000,
                                 base_count=args.final_base, device=device,
                                 ablation="none" if mode == "zero_recurrence" else mode)
            result = {"name": row["name"], "kind": row["kind"], "seed": row["seed"],
                      "ablation": mode, "size": 19, "metrics": metrics}
            results.append(result)
            save_json(args.output_dir / "controls.json", {"runs": results})
            print(json.dumps(result), flush=True)


def leaky_baseline(args, device):
    data = torch.load(args.output_dir / "memory_data.pt", weights_only=False)
    rates = json.loads((args.output_dir / "stage2.json").read_text(encoding="utf-8"))["selected_rates"]
    actor = load_navigation(device)
    detector = load_detector(args.output_dir / "detector.pt", device)
    results = []
    for seed in range(711, 716):
        row = train_memory("fly", seed, 0, rates["fly"], data["train"], data["validation"],
                           device, args.epochs, args.output_dir / "leaky", fixed_zero_edges=True)
        checkpoint = torch.load(args.output_dir / "leaky" / f"{row['name']}.pt", map_location=device, weights_only=False)
        model = LearnedRelation("fly", 1000, seed).to(device)
        model.load_state_dict(checkpoint["model_state"])
        assert torch.count_nonzero(model.edge_values) == 0
        row["sizes"] = {str(size): autonomous(actor, detector, model, size=size,
                                              start_seed=44_000_000 + size * 100_000,
                                              base_count=args.final_base, device=device)
                        for size in (9, 15, 19)}
        results.append(row)
        save_json(args.output_dir / "leaky_baseline.json", {"runs": results})
        print(json.dumps({"name": row["name"], "sizes": row["sizes"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "tune", "run", "evaluate", "controls", "leaky"))
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--train-base", type=int, default=128)
    parser.add_argument("--validation-base", type=int, default=64)
    parser.add_argument("--final-base", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(min(torch.get_num_threads(), 8))
    device = torch.device(args.device)
    {"prepare": prepare, "tune": tune, "run": run, "evaluate": evaluate_final,
     "controls": controls, "leaky": leaky_baseline}[args.stage](args, device)


if __name__ == "__main__":
    main()

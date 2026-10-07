"""Frozen readout recovery and independent difficult-condition replications."""

from __future__ import annotations

import argparse
import json

import torch
from torch import nn
from torch.nn import functional as F

import multi_cue_followup as base
from multi_cue_memory import COLORS, TaskConfig, make_batch

OUT = base.ROOT / "data" / "multi_cue_recovery"
VARIANTS = ("relu_state", "leaky_state", "relu_query", "leaky_query")
KINDS = ("fly", "rewired", "rnn", "gru", "leaky")


@torch.no_grad()
def final_states(memory, inputs, device, lengths=None):
    params, buffers = base.bundle([memory])
    states = []
    for start in range(0, len(inputs), 256):
        x = inputs[start : start + 256].to(device)
        _, trace = base.batched_forward(
            params, buffers, memory.kind, x[None], trace=True
        )
        if lengths is None:
            state = trace[0, :, -1]
        else:
            lens = lengths[start : start + 256].to(device)
            state = trace[0, torch.arange(len(x), device=device), lens - 1]
        states.append(state.cpu())
    return torch.cat(states)


def head_for(hidden, variant):
    return nn.Sequential(
        nn.Linear(hidden + (COLORS if variant.endswith("query") else 0), 128),
        nn.ReLU() if variant.startswith("relu") else nn.LeakyReLU(0.01),
        nn.Linear(128, 3),
    )


def head_inputs(features, mean, scale, variant, device):
    x = (features["states"].to(device) - mean.to(device)) / scale.to(device)
    if variant.endswith("query"):
        x = torch.cat(
            (x, F.one_hot(features["query_color"].to(device), COLORS).float()), -1
        )
    return x


def fit_head(train, val, variant, seed, path, device):
    if path.exists():
        return torch.load(path, map_location="cpu", weights_only=True)
    torch.manual_seed(seed + 30000)
    mean = train["states"].mean(0)
    scale = train["states"].std(0).clamp(min=1e-4)
    tx = head_inputs(train, mean, scale, variant, device)
    vx = head_inputs(val, mean, scale, variant, device)
    ty, vy = train["answers"].to(device), val["answers"].to(device)
    head = head_for(train["states"].shape[1], variant).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=0.003)
    gen = torch.Generator(device=device).manual_seed(seed + 30001)
    best = float("inf")
    best_state = None
    curve = []
    for update in range(1, 501):
        ids = torch.randint(len(tx), (256,), generator=gen, device=device)
        loss = F.cross_entropy(head(tx[ids]), ty[ids])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step()
        if update % 50 == 0:
            with torch.no_grad():
                logits = head(vx)
                ce = float(F.cross_entropy(logits, vy))
                acc = float(logits.argmax(-1).eq(vy).float().mean())
            curve.append({"update": update, "validation_loss": ce, "validation": acc})
            if ce < best:
                best = ce
                best_state = {
                    n: p.detach().cpu().clone() for n, p in head.state_dict().items()
                }
                best_accuracy = acc
                best_update = update
    result = {
        "variant": variant,
        "seed": seed,
        "updates": 500,
        "batch_size": 256,
        "learning_rate": 0.003,
        "training_examples": len(tx),
        "validation_examples": len(vx),
        "best_validation_loss": best,
        "best_validation_accuracy": best_accuracy,
        "best_update": best_update,
        "parameters": sum(p.numel() for p in head.parameters()),
        "curve": curve,
    }
    saved = {"head_state": best_state, "mean": mean, "scale": scale, "result": result}
    base.atomic_torch(path, saved)
    return saved


@torch.no_grad()
def head_predict(saved, features, device):
    head = head_for(features["states"].shape[1], saved["result"]["variant"]).to(device)
    head.load_state_dict(saved["head_state"])
    x = head_inputs(
        features, saved["mean"], saved["scale"], saved["result"]["variant"], device
    )
    return torch.cat([head(part).argmax(-1).cpu() for part in x.split(512)])


def prediction_metrics(pred, features):
    correct = pred.eq(features["answers"])
    return {
        "accuracy": float(correct.float().mean()),
        "by_slot": {
            str(i): {
                "accuracy": float(correct[features["query_slot"] == i].float().mean()),
                "count": int((features["query_slot"] == i).sum()),
            }
            for i in (0, 1)
        },
    }


@torch.no_grad()
def original_predict(memory, features, device):
    return torch.cat(
        [
            memory.head(x.to(device)).argmax(-1).cpu()
            for x in features["states"].split(512)
        ]
    )


def sequence_features(memory, config, count, seed, device, switch=False):
    data = make_batch(config, count, seed)
    colors, kinds, slots = base.metadata(data, config)
    if switch:
        slots = 1 - slots
        rows = torch.arange(count)
        data["inputs"][:, -1, :COLORS] = 0
        data["query_color"] = colors[rows, slots]
        data["inputs"][rows, -1, data["query_color"]] = 1
        data["answers"] = kinds[rows, slots]
    return {
        "states": final_states(memory, data["inputs"], device),
        "query_color": data["query_color"],
        "answers": data["answers"],
        "query_slot": slots,
        "distinct_kinds": kinds[:, 0].ne(kinds[:, 1]),
    }


def cached_sequence(memory, config, seed, folder, bases, device):
    features = []
    for split, count, origin in zip(
        ("train", "val", "test", "switch"),
        (4096, 1024, 4096, 4096),
        (*bases, bases[-1]),
    ):
        path = folder / f"{split}.pt"
        if path.exists():
            item = torch.load(path, weights_only=True)
        else:
            item = sequence_features(
                memory, config, count, origin + seed, device, switch=split == "switch"
            )
            base.atomic_torch(path, item)
        features.append(item)
    return features


def paired_metrics(pred, switched, test, other):
    result = prediction_metrics(pred, test)
    mask = test["distinct_kinds"]
    result.update(
        {
            "changed_query_accuracy": float(
                switched.eq(other["answers"]).float().mean()
            ),
            "both_queries_correct": float(
                (pred.eq(test["answers"]) & switched.eq(other["answers"]))
                .float()
                .mean()
            ),
            "prediction_changes_when_kinds_differ": float(
                switched[mask].ne(pred[mask]).float().mean()
            ),
        }
    )
    return result


def source_runs():
    return [
        base.Run(kind, seed, seed % 5 if kind == "rewired" else 0)
        for kind in KINDS
        for seed in base.SEEDS
    ]


def readout_stage(device):
    config = TaskConfig(2, 2, 4, 2)
    rows = []
    for run in source_runs():
        memory, record = base.load_model(
            base.OUT / "baseline" / f"{config.name}_{run.name}.pt", device
        )
        feats = cached_sequence(
            memory,
            config,
            run.seed,
            OUT / "readout/cache" / run.name,
            (5_100_000_000, 5_200_000_000, 5_300_000_000),
            device,
        )
        for variant in VARIANTS:
            path = OUT / "readout/heads" / f"{run.name}_{variant}.pt"
            saved = fit_head(feats[0], feats[1], variant, run.seed, path, device)
            rows.append(
                {
                    **saved["result"],
                    "run": run.name,
                    "kind": run.kind,
                    "checkpoint": str(path),
                    "source_checkpoint": record["checkpoint"],
                }
            )
        print(
            json.dumps(
                {
                    "event": "readout_fit",
                    "run": run.name,
                    "validation": [r["best_validation_accuracy"] for r in rows[-4:]],
                }
            ),
            flush=True,
        )
    base.atomic_json(OUT / "readout/validation.json", rows)
    scores = {}
    for variant in VARIANTS:
        group = [r for r in rows if r["variant"] == variant]
        scores[variant] = {
            "mean_validation_loss": sum(r["best_validation_loss"] for r in group)
            / len(group),
            "mean_validation_accuracy": sum(
                r["best_validation_accuracy"] for r in group
            )
            / len(group),
        }
    chosen = min(
        VARIANTS,
        key=lambda v: (
            scores[v]["mean_validation_loss"],
            -scores[v]["mean_validation_accuracy"],
            VARIANTS.index(v),
        ),
    )
    selection = {"variant": chosen, "scores": scores, "selection_uses_test": False}
    base.atomic_json(OUT / "selected_head.json", selection)
    print(json.dumps({"event": "selected_head", **selection}), flush=True)
    for run in source_runs():
        result_path = OUT / "readout/results" / f"{run.name}.json"
        if result_path.exists():
            continue
        memory, _ = base.load_model(
            base.OUT / "baseline" / f"{config.name}_{run.name}.pt", device
        )
        folder = OUT / "readout/cache" / run.name
        test, switch = [
            torch.load(folder / f"{s}.pt", weights_only=True)
            for s in ("test", "switch")
        ]
        result = {
            "kind": run.kind,
            "seed": run.seed,
            "graph": run.graph,
            "run": run.name,
            "original": paired_metrics(
                original_predict(memory, test, device),
                original_predict(memory, switch, device),
                test,
                switch,
            ),
            "heads": {},
            "selected_variant": chosen,
        }
        for variant in VARIANTS:
            saved = torch.load(
                OUT / "readout/heads" / f"{run.name}_{variant}.pt", weights_only=True
            )
            result["heads"][variant] = paired_metrics(
                head_predict(saved, test, device),
                head_predict(saved, switch, device),
                test,
                switch,
            )
        base.atomic_json(result_path, result)
        print(
            json.dumps(
                {
                    "event": "readout_test",
                    "run": run.name,
                    "original": result["original"]["accuracy"],
                    "repaired": result["heads"][chosen]["accuracy"],
                }
            ),
            flush=True,
        )


def repeat_stage(device):
    chosen = json.loads((OUT / "selected_head.json").read_text(encoding="utf-8"))[
        "variant"
    ]
    seeds = range(2201, 2206)
    for delay, distractors in ((16, 2), (8, 6)):
        config = TaskConfig(2, 2, delay, distractors)
        for seed in seeds:
            runs = [base.Run("fly", seed)] + [
                base.Run("rewired", seed, graph) for graph in range(5)
            ]
            base.train_group(
                runs, config, 12000, OUT / "repeat/raw", device, counterfactual=True
            )
        for kind in ("rnn", "gru", "leaky"):
            base.train_group(
                [base.Run(kind, seed) for seed in seeds],
                config,
                12000,
                OUT / "repeat/raw",
                device,
                counterfactual=True,
            )
        runs = (
            [base.Run("fly", s) for s in seeds]
            + [base.Run("rewired", s, g) for s in seeds for g in range(5)]
            + [base.Run(k, s) for k in ("rnn", "gru", "leaky") for s in seeds]
        )
        for run in runs:
            name = f"{config.name}_{run.name}"
            result_path = OUT / "repeat/repaired" / f"{name}.json"
            if result_path.exists():
                continue
            memory, source = base.load_model(OUT / "repeat/raw" / f"{name}.pt", device)
            # No persistent feature cache for 90 repeats: resume from each completed result/head.
            feats = [
                sequence_features(
                    memory, config, count, origin + run.seed, device, switch=switch
                )
                for count, origin, switch in (
                    (4096, 6_100_000_000, False),
                    (1024, 6_200_000_000, False),
                    (4096, 6_300_000_000, False),
                    (4096, 6_300_000_000, True),
                )
            ]
            saved = fit_head(
                feats[0],
                feats[1],
                chosen,
                run.seed,
                OUT / "repeat/heads" / f"{name}.pt",
                device,
            )
            test, other = feats[2:]
            result = {
                "run": name,
                "kind": run.kind,
                "seed": run.seed,
                "graph": run.graph,
                "task": config.name,
                "selected_variant": chosen,
                "source_checkpoint": source["checkpoint"],
                "original": paired_metrics(
                    original_predict(memory, test, device),
                    original_predict(memory, other, device),
                    test,
                    other,
                ),
                "repaired": paired_metrics(
                    head_predict(saved, test, device),
                    head_predict(saved, other, device),
                    test,
                    other,
                ),
                "extra_head_training": saved["result"],
            }
            base.atomic_json(result_path, result)
            print(
                json.dumps(
                    {
                        "event": "repeat_repair",
                        "run": name,
                        "original": result["original"]["accuracy"],
                        "repaired": result["repaired"]["accuracy"],
                    }
                ),
                flush=True,
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("readout", "repeat"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    {"readout": readout_stage, "repeat": repeat_stage}[args.stage](torch.device("cuda"))


if __name__ == "__main__":
    main()

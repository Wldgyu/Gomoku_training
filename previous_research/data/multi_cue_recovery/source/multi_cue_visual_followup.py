"""Stage 3b: validation-only rate tuning and prespecified MiniGrid observation comparison."""

from __future__ import annotations

import argparse
import json
import time

import torch
from torch.nn import functional as F

from grid_multi_cue import (
    FEATURES,
    GridMemory,
    GridSpec,
    accuracy_by_delay,
    build_dataset,
    matched_hidden_for,
)
from multi_cue_followup import OUT, SEEDS, atomic_json, atomic_torch
from multi_cue_memory import MemoryModel, parameter_count

KINDS = ("fly", "rewired", "rnn", "gru", "leaky")
RATES = (0.0003, 0.003, 0.01)
VISUAL = OUT / "visual"


def data(split, count, delays, wait=0):
    seeds = {
        "tune_train": 51,
        "tune_val": 52,
        "train": 61,
        "val": 62,
        "test": 63,
        "unseen": 64,
        "wait": 65,
    }
    path = (
        VISUAL
        / "cache"
        / f"{split}_n{count}_delays{'-'.join(map(str, delays))}_wait{wait}.pt"
    )
    if path.exists():
        return torch.load(path, weights_only=True)
    result = build_dataset(
        [GridSpec(2, d, 2, wait) for d in delays], count, seeds[split]
    )
    atomic_torch(path, result)
    return result


@torch.no_grad()
def detailed(model, dataset, device):
    preds = []
    for start in range(0, len(dataset["answers"]), 128):
        part = slice(start, start + 128)
        preds.append(
            model(
                dataset["images"][part].to(device), dataset["lengths"][part].to(device)
            )
            .argmax(-1)
            .cpu()
        )
    correct = torch.cat(preds).eq(dataset["answers"])
    return {
        "accuracy": float(correct.float().mean()),
        "by_delay": accuracy_by_delay(model, dataset, device),
        "by_slot": {
            str(i): {
                "accuracy": float(correct[dataset["query_slot"] == i].float().mean()),
                "count": int((dataset["query_slot"] == i).sum()),
            }
            for i in range(2)
        },
    }


def fit(kind, seed, lr, hidden, train, val, epochs, device, output):
    torch.manual_seed(seed)
    model = GridMemory(kind, hidden, seed % 5 if kind == "rewired" else 0).to(device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    gen = torch.Generator().manual_seed(seed + 7)
    curve = []
    best = (-1.0, float("-inf"))
    best_state = None
    best_epoch = 0
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(train["answers"]), generator=gen)
        for ids in order.split(128):
            loss = F.cross_entropy(
                model(
                    train["images"][ids].to(device), train["lengths"][ids].to(device)
                ),
                train["answers"][ids].to(device),
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        with torch.no_grad():
            correct, total, loss_sum = 0, 0, 0.0
            for start in range(0, len(val["answers"]), 128):
                part = slice(start, start + 128)
                logits = model(
                    val["images"][part].to(device), val["lengths"][part].to(device)
                )
                answers = val["answers"][part].to(device)
                correct += int(logits.argmax(-1).eq(answers).sum())
                total += len(answers)
                loss_sum += float(F.cross_entropy(logits, answers, reduction="sum"))
        score, vloss = correct / total, loss_sum / total
        curve.append({"epoch": epoch, "validation": score, "validation_loss": vloss})
        if (score, -vloss) > best:
            best = (score, -vloss)
            best_epoch = epoch
            best_state = {
                n: v.detach().cpu().clone() for n, v in model.state_dict().items()
            }
    model.load_state_dict(best_state)
    result = {
        "kind": kind,
        "seed": seed,
        "rewire_seed": seed % 5 if kind == "rewired" else 0,
        "hidden": hidden,
        "memory_parameters": parameter_count(model.memory),
        "total_parameters": parameter_count(model),
        "learning_rate": lr,
        "epochs": epochs,
        "updates": epochs * ((len(train["answers"]) + 127) // 128),
        "training_episodes": len(train["answers"]),
        "best_validation": best[0],
        "best_validation_loss": -best[1],
        "best_epoch": best_epoch,
        "curve": curve,
    }
    atomic_torch(output, {"model_state": best_state, "result": result})
    return model, result


def tuning(device):
    train = data("tune_train", 2048, (6, 10))
    val = data("tune_val", 1024, (6, 10))
    target = parameter_count(MemoryModel("fly", 1000, input_dim=FEATURES))
    hidden = {k: matched_hidden_for(k, target) for k in KINDS}
    rows = []
    for kind in KINDS:
        for lr in RATES:
            for seed in (901, 902):
                path = VISUAL / "tuning" / f"{kind}_seed{seed}_lr{lr}.pt"
                if path.exists():
                    result = torch.load(path, weights_only=True)["result"]
                else:
                    _, result = fit(
                        kind, seed, lr, hidden[kind], train, val, 8, device, path
                    )
                rows.append(result)
                atomic_json(VISUAL / "tuning.json", rows)
                print(
                    json.dumps(
                        {
                            "event": "visual_tuning",
                            "kind": kind,
                            "lr": lr,
                            "seed": seed,
                            "val": result["best_validation"],
                        }
                    ),
                    flush=True,
                )
    selected = {}
    for kind in KINDS:
        related = ("fly", "rewired") if kind in ("fly", "rewired") else (kind,)

        def score(lr, related=related):
            group = [
                r for r in rows if r["kind"] in related and r["learning_rate"] == lr
            ]
            return (
                sum(r["best_validation"] for r in group) / len(group),
                -sum(r["best_validation_loss"] for r in group) / len(group),
                -lr,
            )

        selected[kind] = max(RATES, key=score)
    atomic_json(VISUAL / "selected_rates.json", selected)
    print(json.dumps({"event": "visual_rates", "rates": selected}), flush=True)


def final(device):
    rates = json.loads((VISUAL / "selected_rates.json").read_text())
    train = data("train", 8192, (6, 10))
    val = data("val", 1024, (6, 10))
    tests = {
        "trained_delays": data("test", 2048, (6, 10)),
        "unseen_delays": data("unseen", 2048, (8, 14)),
        "unseen_delays_wait": data("wait", 2048, (8, 14), 0.5),
    }
    target = parameter_count(MemoryModel("fly", 1000, input_dim=FEATURES))
    for kind in KINDS:
        for seed in SEEDS:
            result_path = VISUAL / "main" / f"{kind}_seed{seed}.json"
            if result_path.exists():
                continue
            path = result_path.with_suffix(".pt")
            started = time.perf_counter()
            model, result = fit(
                kind,
                seed,
                rates[kind],
                matched_hidden_for(kind, target),
                train,
                val,
                20,
                device,
                path,
            )
            result["tests"] = {
                name: detailed(model, d, device) for name, d in tests.items()
            }
            result["seconds"] = time.perf_counter() - started
            result["checkpoint"] = str(path)
            atomic_json(result_path, result)
            print(
                json.dumps(
                    {
                        "event": "visual_result",
                        "kind": kind,
                        "seed": seed,
                        "tests": {n: v["accuracy"] for n, v in result["tests"].items()},
                    }
                ),
                flush=True,
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "tune", "final"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.stage == "prepare":
        for split, count, delays, wait in (
            ("tune_train", 2048, (6, 10), 0),
            ("tune_val", 1024, (6, 10), 0),
            ("train", 8192, (6, 10), 0),
            ("val", 1024, (6, 10), 0),
            ("test", 2048, (6, 10), 0),
            ("unseen", 2048, (8, 14), 0),
            ("wait", 2048, (8, 14), 0.5),
        ):
            dataset = data(split, count, delays, wait)
            print(
                json.dumps(
                    {
                        "split": split,
                        "episodes": len(dataset["answers"]),
                        "max_frames": int(dataset["lengths"].max()),
                    }
                ),
                flush=True,
            )
    else:
        {"tune": tuning, "final": final}[args.stage](torch.device("cuda"))


if __name__ == "__main__":
    main()

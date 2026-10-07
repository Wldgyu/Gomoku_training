"""Replay published aux runs, freeze their weights, and test distribution shifts.

The binding study saved metrics but no checkpoints. Replay uses its exact seeds,
batch stream, optimizer, and update count; it records the published-score check.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from multi_cue_binding import (BindMemory, LEARNING_RATES, count_parameters,
                               evaluate, loss_for, matched_hidden)
from multi_cue_memory import MODEL_KINDS, TaskConfig, make_batch

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "multi_cue_aux_transfer"
OLD = ROOT / "data" / "multi_cue_binding" / "final"
TRAIN = TaskConfig(2, 2, 16, 2)
CONDITIONS = {
    "trained": TaskConfig(2, 2, 16, 2),
    "delay24": TaskConfig(2, 2, 24, 2),
    "delay32": TaskConfig(2, 2, 32, 2),
    "distractors6": TaskConfig(2, 2, 16, 6),
    "distractors10": TaskConfig(2, 2, 16, 10),
    "combined24_6": TaskConfig(2, 2, 24, 6),
    "three_cues": TaskConfig(3, 3, 16, 2),
}


def replay(kind: str, seed: int, device: torch.device) -> Path:
    path = OUT / "checkpoints" / f"{kind}_aux_seed{seed}.pt"
    if path.exists():
        return path
    target = count_parameters("fly", 1000, "aux")
    hidden = matched_hidden(kind, "aux", target)
    torch.manual_seed(seed)
    model = BindMemory(kind, hidden, "aux", seed % 5).to(device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=LEARNING_RATES[kind])
    for update in range(1, 8001):
        batch = make_batch(TRAIN, 128, 1_000_000_000 + seed * 1_000_000 + update)
        loss = loss_for(model, batch["inputs"].to(device), batch["answers"].to(device))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if update % 2000 == 0:
            print(json.dumps({"kind": kind, "seed": seed, "update": update}), flush=True)
    old_path = OLD / f"cues2_delay16_distract2_{kind}_aux_seed{seed}.json"
    old = json.loads(old_path.read_text(encoding="utf-8"))
    original_test = make_batch(TRAIN, 4096, 3_000_000_000 + seed)
    replay_score = evaluate(model, original_test, device)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": {k: v.cpu() for k, v in model.state_dict().items()},
                "kind": kind, "seed": seed, "hidden": hidden,
                "published_test": old["test"]["accuracy"],
                "replayed_test": replay_score}, path)
    print(json.dumps({"kind": kind, "seed": seed, "published": old["test"]["accuracy"],
                      "replayed": replay_score}), flush=True)
    return path


def load_model(path: Path, device: torch.device) -> tuple[BindMemory, dict]:
    saved = torch.load(path, map_location="cpu", weights_only=True)
    model = BindMemory(saved["kind"], saved["hidden"], "aux", saved["seed"] % 5).to(device)
    model.load_state_dict(saved["state"])
    model.eval()
    return model, saved


def evaluate_shifts(path: Path, device: torch.device) -> dict:
    model, meta = load_model(path, device)
    seed = meta["seed"]
    scores = {}
    for index, (name, config) in enumerate(CONDITIONS.items()):
        # Separate held-out episodes per condition; no gradient or parameter update.
        data = make_batch(config, 4096, 4_000_000_000 + seed * 100 + index)
        scores[name] = evaluate(model, data, device, detail=True)
        if name == "trained":
            swapped = {k: v.clone() for k, v in data.items()}
            swapped["inputs"][:, :2] = data["inputs"][:, [1, 0]]
            scores["reversed_cue_order"] = evaluate(model, swapped, device, detail=True)
    result = {k: v for k, v in meta.items() if k != "state"}
    result["conditions"] = scores
    result["checkpoint"] = str(path)
    result_path = OUT / "generalization" / f"{meta['kind']}_seed{seed}.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"kind": meta["kind"], "seed": seed,
                      "scores": {k: round(v["accuracy"], 4) for k, v in scores.items()}}),
          flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=MODEL_KINDS, default=list(MODEL_KINDS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[3104])
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for seed in args.seeds:
        for kind in args.models:
            path = OUT / "checkpoints" / f"{kind}_aux_seed{seed}.pt"
            if not args.evaluate_only:
                path = replay(kind, seed, device)
            evaluate_shifts(path, device)


if __name__ == "__main__":
    main()

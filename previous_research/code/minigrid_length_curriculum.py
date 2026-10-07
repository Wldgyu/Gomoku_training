"""여러 복도 길이의 교사 경로를 함께 학습하고 미사용 S9에서 평가."""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

from minigrid_full_policy_curriculum import evaluate_reach
from minigrid_memory_diagnostic import build_model
from minigrid_memory_pilot import ROOT, parameter_count
from minigrid_random_start_curriculum import make_dataset, train_batch

TRAIN_SIZES = (7, 11, 13)
HOLDOUT_SIZE = 9


def evaluate_sizes(model, *, episodes: int, seed: int, device: torch.device,
                   sizes: tuple[int, ...], ablations: bool) -> dict:
    rows = {}
    for size in sizes:
        common = {"size": size, "episodes": episodes, "seed": seed,
                  "device": device, "start_mode": "default"}
        row = {"normal": evaluate_reach(model, **common)}
        if ablations:
            row["masked"] = evaluate_reach(model, **common, mask_cue=True)
            row["swapped"] = evaluate_reach(model, **common, swap_cue=True)
        rows[str(size)] = row
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("fly", "rewired", "rnn", "gru"), required=True)
    parser.add_argument("--neurons", choices=(1000, 2000), type=int, default=1000)
    parser.add_argument("--seed", type=int, default=401)
    parser.add_argument("--train-episodes-per-size", type=int, default=1024)
    parser.add_argument("--eval-episodes", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "minigrid_length_curriculum")
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    checkpoint = torch.load(args.source_checkpoint, map_location=device, weights_only=False)
    model = build_model(args.model, args.neurons, args.seed, device)
    model.load_state_dict(checkpoint["model_state"])
    start = time.perf_counter()
    train = []
    for size in TRAIN_SIZES:
        first_seed = 13_000_000 + size * 100_000
        train.extend(make_dataset(range(first_seed, first_seed + args.train_episodes_per_size), size, device))
    initial = evaluate_sizes(model, episodes=args.eval_episodes, seed=14_000_000,
                             device=device, sizes=TRAIN_SIZES + (HOLDOUT_SIZE,), ablations=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    history, best_state, best_row = [], None, None
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(len(train)).tolist()
        losses = []
        for offset in range(0, len(train), args.batch_size):
            batch = [train[index] for index in order[offset:offset + args.batch_size]]
            loss = train_batch(model, batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            losses.append(float(loss.detach()))
        if epoch % 5 == 0 or epoch == args.epochs:
            validated = evaluate_sizes(model, episodes=args.eval_episodes, seed=14_000_000,
                                       device=device, sizes=TRAIN_SIZES, ablations=False)
            scores = [validated[str(size)]["normal"]["success_rate"] for size in TRAIN_SIZES]
            row = {"epoch": epoch, "teacher_loss": float(np.mean(losses)),
                   "mean_train_size_success": float(np.mean(scores)),
                   "min_train_size_success": float(min(scores)), "by_size": validated}
            history.append(row)
            print(json.dumps(row), flush=True)
            if best_row is None or (
                row["mean_train_size_success"], row["min_train_size_success"]
            ) > (best_row["mean_train_size_success"], best_row["min_train_size_success"]):
                best_row = row
                best_state = copy.deepcopy(model.state_dict())
    assert best_state is not None and best_row is not None
    model.load_state_dict(best_state)
    final = evaluate_sizes(model, episodes=args.eval_episodes * 2, seed=15_000_000,
                           device=device, sizes=TRAIN_SIZES + (HOLDOUT_SIZE,), ablations=True)
    result = {"environment_family": "MiniGrid-MemoryS{size}-v0", "start_mode": "default",
              "method": "mixed-length teacher route imitation",
              "train_sizes": TRAIN_SIZES, "holdout_size": HOLDOUT_SIZE,
              "model": args.model, "neurons_reference": args.neurons, "seed": args.seed,
              "parameters": parameter_count(model), "source_checkpoint": str(args.source_checkpoint),
              "train_episodes_per_size": args.train_episodes_per_size,
              "eval_episodes_final_per_size": args.eval_episodes * 2,
              "epochs": args.epochs, "learning_rate": args.learning_rate,
              "initial": initial, "history": history, "selected_epoch": best_row["epoch"],
              "selection_rule": "mean train-size success, then minimum train-size success; S9 excluded",
              "final_fresh": final, "duration_seconds": time.perf_counter() - start}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"mixed_s7_11_13_{args.model}_{args.neurons}_seed{args.seed}"
    (args.output_dir / f"{stem}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    torch.save({"model_state": model.state_dict(), "result": result}, args.output_dir / f"{stem}.pt")
    print(json.dumps({"selected_epoch": best_row["epoch"], "final": final}), flush=True)


if __name__ == "__main__":
    main()

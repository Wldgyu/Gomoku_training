"""원래 무작위 시작 분포의 PPO 후 정책을 새 에피소드와 단서 대조로 평가."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from minigrid_full_policy_curriculum import evaluate_reach
from minigrid_memory_diagnostic import build_model
from minigrid_memory_pilot import ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=256)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint-dir", type=Path,
                        default=ROOT / "data" / "minigrid_random_start" / "ppo_low_lr")
    args = parser.parse_args()
    device = torch.device(args.device)
    directory = args.checkpoint_dir
    rows = []
    for kind in ("fly", "rewired", "rnn", "gru"):
        for seed in (401, 402, 403):
            source = directory / f"memory_s11_{kind}_1000_seed{seed}.pt"
            if not source.exists():
                continue
            checkpoint = torch.load(source, map_location=device, weights_only=False)
            model = build_model(kind, 1000, seed, device)
            model.load_state_dict(checkpoint["model_state"])
            common = {"size": 11, "episodes": args.episodes, "seed": 11_000_000,
                      "device": device, "start_mode": "default"}
            row = {"model": kind, "seed": seed,
                   "normal": evaluate_reach(model, **common),
                   "masked": evaluate_reach(model, **common, mask_cue=True),
                   "swapped": evaluate_reach(model, **common, swap_cue=True)}
            rows.append(row)
            print(json.dumps(row), flush=True)
    output = directory / "heldout_ablation.json"
    output.write_text(json.dumps({"environment": "MiniGrid-MemoryS11-v0",
                                  "start_mode": "default", "episodes": args.episodes,
                                  "rows": rows}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

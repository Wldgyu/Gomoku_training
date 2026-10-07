"""S11 학습 전체 이동 정책의 S7·S13 무재학습 전이 평가."""

from __future__ import annotations

import argparse
import json

import torch

from minigrid_full_policy_curriculum import evaluate_reach
from minigrid_memory_diagnostic import build_model
from minigrid_memory_pilot import ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    rows = []
    for kind in ("fly", "rewired", "rnn", "gru"):
        for seed in (401, 402, 403):
            source = ROOT / "data" / "minigrid_curriculum" / "ppo_low_lr" / f"memory_s11_{kind}_1000_seed{seed}.pt"
            if not source.exists():
                continue
            checkpoint = torch.load(source, map_location=device, weights_only=False)
            model = build_model(kind, 1000, seed, device)
            model.load_state_dict(checkpoint["model_state"])
            for size in (7, 13):
                common = {"size": size, "episodes": args.episodes, "seed": 7_000_000, "device": device}
                row = {"model": kind, "seed": seed, "size": size,
                       "normal": evaluate_reach(model, **common),
                       "masked": evaluate_reach(model, **common, mask_cue=True)}
                rows.append(row)
                print(json.dumps(row), flush=True)
    output = ROOT / "data" / "minigrid_curriculum" / "ppo_low_lr" / "length_generalization.json"
    output.write_text(json.dumps({"source_environment": "MiniGrid-MemoryS11-v0",
                                  "episodes": args.episodes, "rows": rows}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

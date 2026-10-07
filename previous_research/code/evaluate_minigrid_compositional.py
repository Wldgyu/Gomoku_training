"""새 시드에서 원래 이동 정책과 관계 규칙 결합 정책을 짝지어 평가."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from minigrid_compositional_policy import VisualRelations, evaluate_policy
from minigrid_memory_diagnostic import build_model
from minigrid_memory_pilot import ROOT
from minigrid_relation_curriculum import evaluate_variants


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor-model", choices=("fly", "rewired", "rnn", "gru"), required=True)
    parser.add_argument("--actor-seed", type=int, required=True)
    parser.add_argument("--actor-checkpoint", type=Path, required=True)
    parser.add_argument("--detector-checkpoint", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=(9, 15, 19))
    parser.add_argument("--episodes", type=int, default=256)
    parser.add_argument("--first-seed", type=int, default=29_000_000)
    parser.add_argument("--cue-threshold", type=float, default=0.5)
    parser.add_argument("--branch-threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "minigrid_compositional" / "heldout")
    args = parser.parse_args()
    if args.episodes < 1 or not (0.5 <= args.cue_threshold <= 1.0) or not (0.5 <= args.branch_threshold <= 1.0):
        parser.error("episodes와 분류 확률 임계값을 확인하세요")
    device = torch.device(args.device)
    actor = build_model(args.actor_model, 1000, args.actor_seed, device)
    actor.load_state_dict(torch.load(args.actor_checkpoint, map_location=device, weights_only=False)["model_state"])
    detector = VisualRelations().to(device)
    detector.load_state_dict(torch.load(args.detector_checkpoint, map_location=device,
                                        weights_only=False)["detector_state"])
    seeds = range(args.first_seed, args.first_seed + args.episodes)
    by_size = {}
    for size in args.sizes:
        original = evaluate_variants(actor, size=size, seeds=seeds, device=device)
        composed = evaluate_policy(actor, detector, size=size, seeds=seeds,
                                   device=device, cue_threshold=args.cue_threshold,
                                   branch_threshold=args.branch_threshold)
        by_size[str(size)] = {"original_actor": original, "compositional": composed}
        print(json.dumps({"size": size, "original_all_four": original["all_four_success"],
                          "compositional_all_four": composed["all_four_success"],
                          "compositional_success": composed["success_rate"]}), flush=True)
    result = {"actor_model": args.actor_model, "actor_seed": args.actor_seed,
              "actor_checkpoint": str(args.actor_checkpoint),
              "detector_checkpoint": str(args.detector_checkpoint),
              "sizes": args.sizes, "base_episodes_per_size": args.episodes,
              "first_seed": args.first_seed, "cue_threshold": args.cue_threshold,
              "branch_threshold": args.branch_threshold, "by_size": by_size}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = (f"heldout_{args.actor_model}_seed{args.actor_seed}_"
            f"{args.actor_checkpoint.stem}_{args.detector_checkpoint.stem}")
    (args.output_dir / f"{stem}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()

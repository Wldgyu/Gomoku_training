"""저장된 기억 진단 모델의 단서 제거·교체·선택 직전 상태 초기화 평가."""

from __future__ import annotations

import argparse
import json

import torch

from minigrid_memory_diagnostic import accuracy, build_model, make_dataset
from minigrid_memory_pilot import ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=11)
    parser.add_argument("--neurons", type=int, default=1000)
    parser.add_argument("--episodes", type=int, default=512)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    seeds = range(5_000_000, 5_000_000 + args.episodes)
    normal = make_dataset(seeds, args.size, device)
    masked = make_dataset(seeds, args.size, device, mask_cue=True)
    swapped = make_dataset(seeds, args.size, device, swap_cue=True)
    if not torch.equal(normal[2], masked[2]) or not torch.equal(normal[2], swapped[2]):
        raise RuntimeError("대조 조건 사이에 정답이 달라졌습니다")
    rows = []
    for kind in ("fly", "rewired", "rnn", "gru"):
        for seed in (401, 402, 403):
            source = ROOT / "data" / "minigrid_diagnostic" / f"choice_s{args.size}_{kind}_{args.neurons}_seed{seed}.pt"
            checkpoint = torch.load(source, map_location=device, weights_only=False)
            model = build_model(kind, args.neurons, seed, device)
            model.load_state_dict(checkpoint["model_state"])
            row = {"model": kind, "seed": seed,
                   "normal": accuracy(model, normal), "masked": accuracy(model, masked),
                   "branch_state_reset": accuracy(model, normal, reset_at_branch=True),
                   "swapped_cue": accuracy(model, swapped)}
            rows.append(row)
            print(json.dumps(row), flush=True)
    result = {"environment": f"MiniGrid-MemoryS{args.size}-v0", "episodes": args.episodes,
              "label_distribution": normal[2].bincount(minlength=2).tolist(), "rows": rows}
    output = ROOT / "data" / "minigrid_diagnostic" / f"s{args.size}_heldout_ablation_{args.neurons}.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

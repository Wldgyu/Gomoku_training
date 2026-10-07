"""S9 미사용 길이에서 혼합 길이 정책의 마지막 선택 오류를 분해."""

from __future__ import annotations

import argparse
import json
from collections import Counter

import torch

from minigrid_memory_diagnostic import build_model
from minigrid_memory_pilot import ROOT, create_env, reset_episode, tensor_observation
from minigrid_random_start_curriculum import teacher_episode


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("fly", "rewired", "rnn", "gru"), default="rnn")
    parser.add_argument("--seed", type=int, default=401)
    parser.add_argument("--episodes", type=int, default=128)
    parser.add_argument("--size", type=int, default=9)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    source = ROOT / "data" / "minigrid_length_curriculum" / (
        f"mixed_s7_11_13_{args.model}_1000_seed{args.seed}.pt"
    )
    checkpoint = torch.load(source, map_location=device, weights_only=False)
    model = build_model(args.model, 1000, args.seed, device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    counts = Counter()
    for episode_seed in range(15_000_000, 15_000_000 + args.episodes):
        observations, actions, _start_x = teacher_episode(episode_seed, args.size)
        images, directions = tensor_observation(observations, device)
        images = images.unsqueeze(1)
        directions = directions.unsqueeze(1)
        resets = torch.zeros(images.shape[:2], dtype=torch.bool, device=device)
        logits, _ = model.trajectory(images, directions, resets, model.initial_state(1, device))
        predicted = int(logits[-2, 0, :2].argmax())
        correct = int(actions[-2])
        env = create_env(args.size, 100)
        try:
            reset_episode(env, episode_seed, "default")
            base = env.unwrapped
            center = base.height // 2
            cue = type(base.grid.get(1, center - 1)).__name__
            upper = type(base.grid.get(args.size - 2, center - 2)).__name__
        finally:
            env.close()
        counts[(cue, upper, correct, predicted)] += 1
    rows = [{"cue": key[0], "upper": key[1], "correct": key[2],
             "predicted": key[3], "count": count} for key, count in sorted(counts.items())]
    print(json.dumps({"size": args.size, "episodes": args.episodes,
                      "teacher_path_choice_accuracy": sum(row["count"] for row in rows
                          if row["correct"] == row["predicted"]) / args.episodes,
                      "rows": rows}, indent=2))


if __name__ == "__main__":
    main()

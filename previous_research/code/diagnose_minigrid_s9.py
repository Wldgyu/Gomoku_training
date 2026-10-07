"""미사용 S9의 단서 탐색, 분기점 이동, 마지막 선택을 에피소드별로 분해."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from minigrid_memory_diagnostic import build_model, swapped_cue_observation
from minigrid_memory_pilot import (
    ROOT,
    create_env,
    masked_cue_observation,
    reset_episode,
    tensor_observation,
)
from minigrid_random_start_curriculum import teacher_episode


def autonomous_episode(model, seed: int, size: int, device: torch.device) -> dict:
    env = create_env(size, 100)
    try:
        observation = reset_episode(env, seed, "default")
        base = env.unwrapped
        center = base.height // 2
        cue_type = type(base.grid.get(1, center - 1)).__name__
        start_x = int(base.agent_pos[0])
        state = model.initial_state(1, device)
        matrix = model.recurrent_matrix()
        seen, reached, turn = False, False, None
        while True:
            seen |= not np.array_equal(observation["image"], masked_cue_observation(env)["image"])
            image, direction = tensor_observation([observation], device)
            logits, _, state = model.step(image, direction, state, matrix)
            action = int(logits.argmax(dim=-1).item())
            if tuple(base.agent_pos) == (size - 2, center):
                reached = True
                if turn is None:
                    turn = action
            observation, reward, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                return {"start_x": start_x, "cue_type": cue_type, "cue_seen": seen,
                        "branch_reached": reached, "first_branch_action": turn,
                        "correct_turn": int(base.success_pos[1] > center), "success": reward > 0}
    finally:
        env.close()


def teacher_choices(model, seed: int, size: int, device: torch.device) -> tuple[int, int]:
    observations, actions, _ = teacher_episode(seed, size)
    env = create_env(size, 100)
    try:
        reset_episode(env, seed, "default")
        swapped = []
        for index, observation in enumerate(observations):
            visible = not np.array_equal(observation["image"], masked_cue_observation(env)["image"])
            swapped.append(swapped_cue_observation(env) if visible else observation)
            if index < len(observations) - 1:
                env.step(actions[index])
    finally:
        env.close()

    def predict(sequence: list[dict]) -> int:
        images, directions = tensor_observation(sequence, device)
        resets = torch.zeros((len(sequence), 1), dtype=torch.bool, device=device)
        logits, _ = model.trajectory(images.unsqueeze(1), directions.unsqueeze(1), resets,
                                     model.initial_state(1, device))
        return int(logits[-2, 0, :2].argmax().item())

    return predict(observations), predict(swapped)


@torch.no_grad()
def diagnose(model, *, episodes: int, first_seed: int, size: int, device: torch.device) -> dict:
    model.eval()
    rows = []
    for seed in range(first_seed, first_seed + episodes):
        row = autonomous_episode(model, seed, size, device)
        row["teacher_choice"], row["teacher_swap_choice"] = teacher_choices(model, seed, size, device)
        rows.append(row)
    counts = Counter()
    by_start = Counter()
    by_cue = Counter()
    for row in rows:
        if not row["cue_seen"]:
            category = "cue_not_seen"
        elif not row["branch_reached"]:
            category = "navigation_after_cue"
        elif not row["success"]:
            category = "branch_or_finish"
        else:
            category = "success"
        counts[category] += 1
        by_start[(row["start_x"], category)] += 1
        by_cue[(row["cue_type"], category)] += 1
    rates = {key: sum(bool(row[key]) for row in rows) / episodes
             for key in ("cue_seen", "branch_reached", "success")}
    return {"episodes": episodes, "first_seed": first_seed, "size": size,
            "cue_seen_rate": rates["cue_seen"], "branch_reach_rate": rates["branch_reached"],
            "success_rate": rates["success"],
            "first_branch_turn_accuracy_given_reach": sum(
                row["first_branch_action"] == row["correct_turn"] for row in rows if row["branch_reached"]
            ) / max(1, sum(row["branch_reached"] for row in rows)),
            "teacher_choice_accuracy": sum(row["teacher_choice"] == row["correct_turn"] for row in rows) / episodes,
            "teacher_swap_flip_rate": sum(row["teacher_choice"] != row["teacher_swap_choice"] for row in rows) / episodes,
            "failure_counts": dict(counts),
            "failure_by_start_x": [{"start_x": x, "category": category, "count": count}
                                   for (x, category), count in sorted(by_start.items())],
            "failure_by_cue_type": [{"cue_type": cue, "category": category, "count": count}
                                    for (cue, category), count in sorted(by_cue.items())]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("fly", "rewired", "rnn", "gru"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--size", type=int, default=9)
    parser.add_argument("--episodes", type=int, default=128)
    parser.add_argument("--first-seed", type=int, default=16_000_000)
    parser.add_argument("--checkpoint", type=Path,
                        help="기본 혼합 길이 체크포인트 대신 평가할 명시적 경로")
    parser.add_argument("--checkpoint-dir", type=Path, default=ROOT / "data" / "minigrid_length_curriculum")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "minigrid_s9_diagnostic")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    stem = f"mixed_s7_11_13_{args.model}_1000_seed{args.seed}"
    source = args.checkpoint or args.checkpoint_dir / f"{stem}.pt"
    checkpoint = torch.load(source, map_location=device, weights_only=False)
    model = build_model(args.model, 1000, args.seed, device)
    model.load_state_dict(checkpoint["model_state"])
    result = {"model": args.model, "seed": args.seed, "checkpoint": str(source),
              **diagnose(model, episodes=args.episodes, first_seed=args.first_seed,
                         size=args.size, device=device)}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_stem = source.stem
    (args.output_dir / f"{output_stem}_diagnostic.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

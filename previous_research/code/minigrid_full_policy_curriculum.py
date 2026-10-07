"""기억 진단 체크포인트에서 전체 MiniGrid 이동 정책으로 확장하는 지도학습 탐색.

훈련용 교사 경로의 정답에는 환경 내부 상태를 쓰지만 모델 입력은 기존의
부분관측 image/direction뿐이다. 평가는 교사 없이 세 이동 행동을 자율 선택한다.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from minigrid_memory_diagnostic import build_model, swapped_cue_observation
from minigrid_memory_pilot import (
    ROOT,
    create_env,
    masked_cue_observation,
    parameter_count,
    reset_episode,
    tensor_observation,
)


def teacher_episode(seed: int, size: int) -> tuple[list[dict], list[int]]:
    env = create_env(size, 100)
    try:
        observation = reset_episode(env, seed, "fixed_cue")
        base = env.unwrapped
        center = base.height // 2
        turn = int(base.success_pos[1] > center)
        observations, actions = [], []
        for action in [2] * (size - 3) + [turn, 2]:
            observations.append(observation)
            actions.append(action)
            observation, reward, terminated, truncated, _ = env.step(action)
            if truncated or (terminated != (len(actions) == size - 1)):
                raise RuntimeError("교사 경로가 예상과 다른 단계에서 종료됐습니다")
        if reward <= 0:
            raise RuntimeError("교사 경로의 최종 보상이 없습니다")
        return observations, actions
    finally:
        env.close()


def make_dataset(seeds: range, size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    trajectories = [teacher_episode(seed, size) for seed in seeds]
    observations = [observation for sequence, _ in trajectories for observation in sequence]
    images, directions = tensor_observation(observations, device)
    count, length = len(trajectories), len(trajectories[0][1])
    images = images.reshape(count, length, *images.shape[1:]).transpose(0, 1)
    directions = directions.reshape(count, length).transpose(0, 1)
    actions = torch.tensor([action for _, sequence in trajectories for action in sequence], device=device)
    actions = actions.reshape(count, length).transpose(0, 1)
    return images, directions, actions


@torch.no_grad()
def evaluate_reach(model, *, size: int, episodes: int, seed: int, device: torch.device,
                   mask_cue: bool = False, swap_cue: bool = False,
                   start_mode: str = "fixed_cue") -> dict:
    """성공률과 분기점 도달률을 같은 독립 에피소드에서 측정한다."""
    model.eval()
    if mask_cue and swap_cue:
        raise ValueError("단서 제거와 교체를 동시에 평가할 수 없습니다")
    observed = swapped_cue_observation if swap_cue else masked_cue_observation if mask_cue else None
    successes, reached, cue_seen_count, cue_first_count = 0, 0, 0, 0
    for episode_seed in range(seed, seed + episodes):
        env = create_env(size, 100)
        try:
            observation = reset_episode(env, episode_seed, start_mode)
            cue_visible = not np.array_equal(observation["image"], masked_cue_observation(env)["image"])
            cue_first_count += cue_visible
            if observed is not None:
                observation = observed(env)
            state = model.initial_state(1, device)
            matrix = model.recurrent_matrix()
            did_reach = False
            while True:
                image, direction = tensor_observation([observation], device)
                logits, _, state = model.step(image, direction, state, matrix)
                observation, reward, terminated, truncated, _ = env.step(int(logits.argmax(dim=1).item()))
                base = env.unwrapped
                if tuple(base.agent_pos) == (size - 2, base.height // 2):
                    did_reach = True
                if not terminated and not truncated:
                    cue_visible |= not np.array_equal(observation["image"], masked_cue_observation(env)["image"])
                if observed is not None:
                    observation = observed(env)
                if terminated or truncated:
                    successes += reward > 0
                    reached += did_reach
                    cue_seen_count += cue_visible
                    break
        finally:
            env.close()
    return {"success_rate": successes / episodes, "branch_reach_rate": reached / episodes,
            "cue_seen_rate": cue_seen_count / episodes,
            "cue_visible_at_start_rate": cue_first_count / episodes}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("fly", "rewired", "rnn", "gru"), required=True)
    parser.add_argument("--neurons", choices=(1000, 2000), type=int, default=1000)
    parser.add_argument("--size", choices=(7, 11, 13), type=int, default=11)
    parser.add_argument("--seed", type=int, default=401)
    parser.add_argument("--train-episodes", type=int, default=1024)
    parser.add_argument("--eval-episodes", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--freeze-memory", action="store_true",
                        help="진단에서 학습한 관측 인코더와 순환부를 고정하고 출력층만 학습")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--diagnostic-dir", type=Path, default=ROOT / "data" / "minigrid_diagnostic")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "minigrid_curriculum")
    args = parser.parse_args()
    device = torch.device(args.device)
    stem = f"choice_s{args.size}_{args.model}_{args.neurons}_seed{args.seed}"
    source = args.diagnostic_dir / f"{stem}.pt"
    checkpoint = torch.load(source, map_location=device, weights_only=False)
    model = build_model(args.model, args.neurons, args.seed, device)
    model.load_state_dict(checkpoint["model_state"])
    if args.freeze_memory:
        for parameter in model.encoder.parameters():
            parameter.requires_grad_(False)
        if model.kind == "fly":
            model.edge_values.requires_grad_(False)
            for parameter in model.input_layer.parameters():
                parameter.requires_grad_(False)
        else:
            for parameter in model.core.parameters():
                parameter.requires_grad_(False)
    start = time.perf_counter()
    train = make_dataset(range(3_000_000, 3_000_000 + args.train_episodes), args.size, device)
    eval_seed = 4_000_000
    before = evaluate_reach(model, size=args.size, episodes=args.eval_episodes,
                            seed=eval_seed, device=device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.learning_rate)
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(args.train_episodes, device=device)
        losses = []
        for indices in order.split(args.batch_size):
            image, direction, actions = train[0][:, indices], train[1][:, indices], train[2][:, indices]
            reset = torch.zeros(image.shape[:2], dtype=torch.bool, device=device)
            logits, _ = model.trajectory(image, direction, reset, model.initial_state(len(indices), device))
            per_step_loss = F.cross_entropy(logits.reshape(-1, 3), actions.reshape(-1), reduction="none")
            per_step_loss = per_step_loss.reshape_as(actions)
            # 전진 행동 9번이 한 번의 기억 선택을 덮지 않게 분기점에 가중치를 준다.
            weights = torch.ones_like(per_step_loss)
            weights[-2] = 8.0
            loss = (per_step_loss * weights).sum() / weights.sum()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            losses.append(float(loss.detach()))
        if epoch % 10 == 0 or epoch == args.epochs:
            row = {"epoch": epoch, "teacher_loss": float(np.mean(losses)),
                   **evaluate_reach(model, size=args.size, episodes=args.eval_episodes,
                                    seed=eval_seed, device=device)}
            history.append(row)
            print(json.dumps(row), flush=True)
    final = evaluate_reach(model, size=args.size, episodes=args.eval_episodes,
                           seed=eval_seed + 100_000, device=device)
    masked = evaluate_reach(model, size=args.size, episodes=args.eval_episodes,
                            seed=eval_seed + 100_000, device=device, mask_cue=True)
    result = {"environment": f"MiniGrid-MemoryS{args.size}-v0", "method": "teacher route imitation after diagnostic pretraining",
              "observation": "partial 7x7 image and direction only", "model": args.model,
              "neurons_reference": args.neurons, "seed": args.seed,
              "parameters": parameter_count(model), "source_checkpoint": str(source),
              "train_episodes": args.train_episodes, "eval_episodes": args.eval_episodes,
              "epochs": args.epochs, "freeze_memory": args.freeze_memory, "before": before, "history": history,
              "final_fresh": final, "final_fresh_cue_masked": masked,
              "duration_seconds": time.perf_counter() - start}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dest_stem = f"full_s{args.size}_{args.model}_{args.neurons}_seed{args.seed}"
    if args.freeze_memory:
        dest_stem += "_frozen"
    (args.output_dir / f"{dest_stem}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    torch.save({"model_state": model.state_dict(), "result": result}, args.output_dir / f"{dest_stem}.pt")
    print(json.dumps({"final": final, "masked": masked}), flush=True)


if __name__ == "__main__":
    main()

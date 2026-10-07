"""MiniGrid S11 원래 무작위 시작 분포에서 단서 찾기·기억·전체 이동 학습.

성공 경로는 훈련 라벨에만 사용한다. 모델 입력은 기존 부분관측 이미지와
방향이며, 검증과 최종 평가는 교사 없이 세 행동을 자율 선택한다.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from minigrid_full_policy_curriculum import evaluate_reach
from minigrid_memory_diagnostic import build_model
from minigrid_memory_pilot import (
    ROOT,
    create_env,
    masked_cue_observation,
    parameter_count,
    reset_episode,
    tensor_observation,
)


def teacher_episode(seed: int, size: int) -> tuple[list[dict], list[int], int]:
    """단서가 안 보이면 뒤로 가서 확인한 뒤 목표 분기로 가는 경로."""
    env = create_env(size, 100)
    try:
        observation = reset_episode(env, seed, "default")
        base = env.unwrapped
        start_x = int(base.agent_pos[0])
        center = base.height // 2
        if int(base.agent_pos[1]) != center or base.agent_dir != 0:
            raise RuntimeError("예상 밖의 시작 위치·방향")
        turn = int(base.success_pos[1] > center)
        if start_x == 1:
            actions = [2] * (size - 3) + [turn, 2]
        else:
            # 동쪽을 향해 시작하므로 남쪽·서쪽으로 돌아 출발 방까지 복귀한다.
            # x=1에서 북쪽을 보면 단서가 보이고, 다시 동쪽으로 이동한다.
            actions = [1, 1] + [2] * (start_x - 1) + [1, 1]
            actions += [2] * (size - 3) + [turn, 2]
        observations = []
        cue_seen = False
        for index, action in enumerate(actions):
            observations.append(observation)
            cue_seen |= not np.array_equal(observation["image"], masked_cue_observation(env)["image"])
            observation, reward, terminated, truncated, _ = env.step(action)
            if truncated or (terminated != (index == len(actions) - 1)):
                raise RuntimeError("교사 경로가 예상과 다른 단계에서 종료됐습니다")
        if reward <= 0 or not cue_seen:
            raise RuntimeError("교사가 단서를 보지 못했거나 목표에 실패했습니다")
        return observations, actions, start_x
    finally:
        env.close()


def make_dataset(seeds: range, size: int, device: torch.device) -> list[dict]:
    rows = []
    for seed in seeds:
        observations, actions, start_x = teacher_episode(seed, size)
        images, directions = tensor_observation(observations, torch.device("cpu"))
        rows.append({"images": images, "directions": directions,
                     "actions": torch.tensor(actions), "start_x": start_x})
    return rows


def train_batch(model, rows: list[dict], device: torch.device) -> torch.Tensor:
    length, batch = max(len(row["actions"]) for row in rows), len(rows)
    images = torch.zeros((length, batch, 7, 7, 3), dtype=torch.long, device=device)
    directions = torch.zeros((length, batch), dtype=torch.long, device=device)
    actions = torch.zeros((length, batch), dtype=torch.long, device=device)
    weights = torch.zeros((length, batch), device=device)
    for index, row in enumerate(rows):
        count = len(row["actions"])
        images[:count, index] = row["images"].to(device)
        directions[:count, index] = row["directions"].to(device)
        actions[:count, index] = row["actions"].to(device)
        weights[:count, index] = 1.0
        weights[count - 2, index] = 8.0  # 마지막 기억 선택
    resets = torch.zeros((length, batch), dtype=torch.bool, device=device)
    logits, _ = model.trajectory(images, directions, resets, model.initial_state(batch, device))
    loss = F.cross_entropy(logits.reshape(-1, 3), actions.reshape(-1), reduction="none")
    return (loss.reshape(length, batch) * weights).sum() / weights.sum()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("fly", "rewired", "rnn", "gru"), required=True)
    parser.add_argument("--neurons", choices=(1000, 2000), type=int, default=1000)
    parser.add_argument("--size", choices=(11, 13), type=int, default=11)
    parser.add_argument("--seed", type=int, default=401)
    parser.add_argument("--train-episodes", type=int, default=2048)
    parser.add_argument("--eval-episodes", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--source-dir", type=Path, default=ROOT / "data" / "minigrid_curriculum" / "ppo_low_lr")
    parser.add_argument("--source-checkpoint", type=Path, default=None,
                        help="source-dir의 기본 체크포인트 대신 사용할 명시적 경로")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "minigrid_random_start")
    args = parser.parse_args()
    if min(args.train_episodes, args.eval_episodes, args.epochs, args.batch_size) < 1:
        parser.error("에피소드·epoch·batch 인자는 양수여야 합니다")
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    source = args.source_checkpoint or (
        args.source_dir / f"memory_s{args.size}_{args.model}_{args.neurons}_seed{args.seed}.pt"
    )
    checkpoint = torch.load(source, map_location=device, weights_only=False)
    model = build_model(args.model, args.neurons, args.seed, device)
    model.load_state_dict(checkpoint["model_state"])
    start = time.perf_counter()
    train = make_dataset(range(8_000_000, 8_000_000 + args.train_episodes), args.size, device)
    start_counts = np.bincount([row["start_x"] for row in train], minlength=args.size).tolist()
    validation_seed = 9_000_000
    initial = evaluate_reach(model, size=args.size, episodes=args.eval_episodes,
                             seed=validation_seed, device=device, start_mode="default")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    best_state = None
    best_row = None
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        permutation = torch.randperm(len(train)).tolist()
        losses = []
        for offset in range(0, len(train), args.batch_size):
            batch = [train[index] for index in permutation[offset:offset + args.batch_size]]
            loss = train_batch(model, batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            losses.append(float(loss.detach()))
        if epoch % 5 == 0 or epoch == args.epochs:
            evaluated = evaluate_reach(model, size=args.size, episodes=args.eval_episodes,
                                       seed=validation_seed, device=device, start_mode="default")
            row = {"epoch": epoch, "teacher_loss": float(np.mean(losses)), **evaluated}
            history.append(row)
            print(json.dumps(row), flush=True)
            if best_row is None or (row["success_rate"], row["cue_seen_rate"]) > (
                best_row["success_rate"], best_row["cue_seen_rate"]
            ):
                best_row = row
                best_state = copy.deepcopy(model.state_dict())
    assert best_state is not None and best_row is not None
    model.load_state_dict(best_state)
    fresh_seed = 10_000_000
    common = {"size": args.size, "episodes": args.eval_episodes,
              "seed": fresh_seed, "device": device, "start_mode": "default"}
    final = evaluate_reach(model, **common)
    masked = evaluate_reach(model, **common, mask_cue=True)
    swapped = evaluate_reach(model, **common, swap_cue=True)
    result = {"environment": f"MiniGrid-MemoryS{args.size}-v0", "start_mode": "default",
              "method": "teacher route imitation from S11 fixed-start checkpoint",
              "model": args.model, "neurons_reference": args.neurons, "seed": args.seed,
              "parameters": parameter_count(model), "source_checkpoint": str(source),
              "train_episodes": args.train_episodes, "start_x_counts": start_counts,
              "eval_episodes": args.eval_episodes, "epochs": args.epochs,
              "learning_rate": args.learning_rate, "initial": initial,
              "history": history, "selected_epoch": best_row["epoch"],
              "selection_rule": "highest validation success, then cue-seen rate",
              "final_fresh": final, "final_fresh_cue_masked": masked,
              "final_fresh_cue_swapped": swapped,
              "duration_seconds": time.perf_counter() - start}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"random_s{args.size}_{args.model}_{args.neurons}_seed{args.seed}"
    (args.output_dir / f"{stem}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    torch.save({"model_state": model.state_dict(), "result": result}, args.output_dir / f"{stem}.pt")
    print(json.dumps({"selected_epoch": best_row["epoch"], "final": final,
                      "masked": masked, "swapped": swapped}), flush=True)


if __name__ == "__main__":
    main()

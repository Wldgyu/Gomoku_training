"""MiniGrid Memory의 복도 이동을 고정한 마지막 좌·우 기억 진단.

모델에는 기존 PPO와 같은 7x7 이미지·방향 시퀀스만 입력한다. 환경 내부의
success_pos는 학습 라벨과 검증에만 사용한다. 복도는 정해진 forward 행동으로
이동하며, 분기점에서 모델이 선택한 turn과 forward를 실제 환경에 적용한다.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from minigrid.core.world_object import Ball, Key
from torch.nn import functional as F

from minigrid_memory_pilot import (
    GRAPH_DIR,
    ROOT,
    MemoryPolicy,
    create_env,
    masked_cue_observation,
    matched_hidden_size,
    parameter_count,
    reset_episode,
    tensor_observation,
)


def swapped_cue_observation(env) -> dict:
    """환경의 목표는 그대로 두고 출발 물체의 관측 종류만 반대로 만든다."""
    base = env.unwrapped
    x, y = 1, base.height // 2 - 1
    original = base.grid.get(x, y)
    swapped = Key(original.color) if isinstance(original, Ball) else Ball(original.color)
    base.grid.set(x, y, swapped)
    try:
        return base.gen_obs()
    finally:
        base.grid.set(x, y, original)


def episode(seed: int, size: int, *, mask_cue: bool = False,
            swap_cue: bool = False) -> tuple[list[dict], int, int]:
    """출발에서 분기점까지의 관측, 정답 turn, 첫 단서 종류."""
    env = create_env(size, 100)
    try:
        if mask_cue and swap_cue:
            raise ValueError("단서 제거와 교체를 동시에 지정할 수 없습니다")
        observation = reset_episode(env, seed, "fixed_cue")
        base = env.unwrapped
        center = base.height // 2
        label = int(base.success_pos[1] > center)
        cue_name = type(base.grid.get(1, center - 1)).__name__
        if cue_name not in ("Ball", "Key"):
            raise RuntimeError(f"예상 밖의 출발 단서: {cue_name}")
        cue_label = int(cue_name == "Key")
        observed = swapped_cue_observation if swap_cue else masked_cue_observation if mask_cue else None
        observations = [observed(env) if observed is not None else observation]
        # 출발 x=1, 분기점 x=size-2. 이 이동 중에는 보상이나 정답을 입력하지 않는다.
        for _ in range(size - 3):
            observation, reward, terminated, truncated, _ = env.step(2)
            if reward or terminated or truncated:
                raise RuntimeError("분기점 전에 에피소드가 끝났습니다")
            observations.append(observed(env) if observed is not None else observation)
        if tuple(base.agent_pos) != (size - 2, center) or base.agent_dir != 0:
            raise RuntimeError("고정 복도 이동이 분기점에 도달하지 못했습니다")
        # 라벨이 실제 MiniGrid 성공 보상과 일치하는지도 확인한다.
        _, _, terminated, truncated, _ = env.step(label)
        if terminated or truncated:
            raise RuntimeError("회전 중 에피소드가 끝났습니다")
        _, reward, terminated, _, _ = env.step(2)
        if not terminated or reward <= 0:
            raise RuntimeError("정답 turn이 실제 환경에서 성공하지 못했습니다")
        return observations, label, cue_label
    finally:
        env.close()


def make_dataset(seeds: range, size: int, device: torch.device,
                 *, mask_cue: bool = False, swap_cue: bool = False
                 ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    trajectories, labels, cue_labels = [], [], []
    for seed in seeds:
        observations, label, cue_label = episode(seed, size, mask_cue=mask_cue, swap_cue=swap_cue)
        trajectories.append(observations)
        labels.append(label)
        cue_labels.append(cue_label)
    length = len(trajectories[0])
    if any(len(item) != length for item in trajectories):
        raise RuntimeError("서로 다른 시퀀스 길이")
    images, directions = tensor_observation(
        [observation for trajectory in trajectories for observation in trajectory], device
    )
    images = images.reshape(len(labels), length, *images.shape[1:]).transpose(0, 1)
    directions = directions.reshape(len(labels), length).transpose(0, 1)
    return images, directions, torch.tensor(labels, device=device), torch.tensor(cue_labels, device=device)


@torch.no_grad()
def accuracy(model: MemoryPolicy, dataset: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
             *, reset_at_branch: bool = False, batch_size: int = 64) -> float:
    model.eval()
    images, directions, labels, _ = dataset
    correct = 0
    for start in range(0, len(labels), batch_size):
        stop = start + batch_size
        image, direction, label = images[:, start:stop], directions[:, start:stop], labels[start:stop]
        resets = torch.zeros(image.shape[:2], device=images.device, dtype=torch.bool)
        if reset_at_branch:
            resets[-1] = True
        logits, _ = model.trajectory(image, direction, resets, model.initial_state(len(label), images.device))
        correct += (logits[-1, :, :2].argmax(dim=-1) == label).sum().item()
    return correct / len(labels)


def build_model(kind: str, neurons: int, seed: int, device: torch.device) -> MemoryPolicy:
    torch.manual_seed(seed)
    reference = MemoryPolicy("fly", neurons, GRAPH_DIR / f"malecns_cx_{neurons}.npz")
    if kind == "fly":
        model = reference
    elif kind == "rewired":
        torch.manual_seed(seed)
        model = MemoryPolicy("fly", neurons, GRAPH_DIR / f"malecns_cx_{neurons}_rewired_seed0.npz")
    else:
        hidden = matched_hidden_size(kind, parameter_count(reference))
        model = MemoryPolicy(kind, neurons, hidden_size=hidden)
    return model.to(device)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("fly", "rewired", "rnn", "gru"), required=True)
    parser.add_argument("--neurons", choices=(1000, 2000), type=int, default=1000)
    parser.add_argument("--size", choices=(7, 11, 13), type=int, default=11)
    parser.add_argument("--seed", type=int, default=401)
    parser.add_argument("--train-episodes", type=int, default=1024)
    parser.add_argument("--eval-episodes", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--cue-loss-weight", type=float, default=0.5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "minigrid_diagnostic")
    args = parser.parse_args()
    if min(args.train_episodes, args.eval_episodes, args.epochs, args.batch_size) < 1:
        parser.error("에피소드·epoch·batch 인자는 양수여야 합니다")
    device = torch.device(args.device)
    start = time.perf_counter()
    # 학습/평가 에피소드는 겹치지 않으며, 각 모델에 동일한 환경 시드를 쓴다.
    train = make_dataset(range(1_000_000, 1_000_000 + args.train_episodes), args.size, device)
    eval_seeds = range(2_000_000, 2_000_000 + args.eval_episodes)
    heldout = make_dataset(eval_seeds, args.size, device)
    masked = make_dataset(eval_seeds, args.size, device, mask_cue=True)
    if not torch.equal(heldout[2], masked[2]) or not torch.equal(heldout[3], masked[3]):
        raise RuntimeError("단서 제거가 정답을 변경했습니다")
    model = build_model(args.model, args.neurons, args.seed, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        permutation = torch.randperm(args.train_episodes, device=device)
        losses = []
        for indices in permutation.split(args.batch_size):
            image, direction, label, cue_label = (
                train[0][:, indices], train[1][:, indices], train[2][indices], train[3][indices]
            )
            reset = torch.zeros(image.shape[:2], dtype=torch.bool, device=device)
            logits, _ = model.trajectory(image, direction, reset, model.initial_state(len(indices), device))
            loss = F.cross_entropy(logits[-1, :, :2], label)
            if args.cue_loss_weight:
                # 훈련 중 첫 단서의 종류를 상태에 유지하도록 보조 신호를 준다.
                # 최종 선택 평가는 라벨 없이 새 에피소드에서만 수행한다.
                cue_logits = logits[:-1, :, :2].reshape(-1, 2)
                cue_targets = cue_label.unsqueeze(0).expand(image.shape[0] - 1, -1).reshape(-1)
                loss = loss + args.cue_loss_weight * F.cross_entropy(cue_logits, cue_targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            losses.append(float(loss.detach()))
        if epoch == 1 or epoch % 4 == 0 or epoch == args.epochs:
            row = {"epoch": epoch, "loss": float(np.mean(losses)),
                   "heldout_accuracy": accuracy(model, heldout),
                   "masked_accuracy": accuracy(model, masked),
                   "branch_state_reset_accuracy": accuracy(model, heldout, reset_at_branch=True)}
            history.append(row)
            print(json.dumps(row), flush=True)
    result = {"environment": f"MiniGrid-MemoryS{args.size}-v0", "task": "fixed corridor, final left/right",
              "model": args.model, "neurons_reference": args.neurons, "seed": args.seed,
              "parameters": parameter_count(model), "train_episodes": args.train_episodes,
              "eval_episodes": args.eval_episodes, "epochs": args.epochs,
              "learning_rate": args.learning_rate, "cue_loss_weight": args.cue_loss_weight,
              "label_distribution_train": train[2].bincount(minlength=2).tolist(),
              "label_distribution_eval": heldout[2].bincount(minlength=2).tolist(),
              "history": history, "duration_seconds": time.perf_counter() - start}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"choice_s{args.size}_{args.model}_{args.neurons}_seed{args.seed}"
    (args.output_dir / f"{stem}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    torch.save({"model_state": model.state_dict(), "result": result}, args.output_dir / f"{stem}.pt")
    print(f"Saved {stem}", flush=True)


if __name__ == "__main__":
    main()

"""기존 '기억 반전 게임' 1·2단계의 핵심 카드 규칙을 학습용으로 재현한다.

원본 game.js의 1단계 규칙: 서로 다른 카드 4장을 보여주고, 잠시 비운 뒤
한 자리를 새로운 카드로 바꾼다. 에이전트는 바뀐 위치 1개를 선택한다.
`supervised` 진단은 정답을 직접 알려주고, `reinforce`는 선택 후 보상만 준다.
2단계는 색상 또는 이모지 카드가 나오는 규칙을 더한다. 화면 효과·실시간
타이머·가짜 흔들림 힌트와 3단계 이후 규칙은 파일럿에서 제외한다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn import functional as F

from project_config import PILOT_NEURON_COUNTS

CARD_TYPES_PER_POOL = 8  # 원본 게임의 색상·이모지 종류 수는 각각 8개
CARD_SLOTS = 4
READOUT_HIDDEN = 128  # 세 모델에 동일하게 적용할 카드 비교용 출력층


def observation_size(stage: int) -> int:
    """1단계는 색상 8종, 2단계는 색상·이모지 16종을 구별한다."""
    return CARD_SLOTS * CARD_TYPES_PER_POOL * stage + 3


def make_readout(state_size: int, input_size: int) -> nn.Module:
    """기억 상태와 현재 카드 관측을 함께 읽는 공통 비선형 출력층."""
    return nn.Sequential(
        nn.Linear(state_size + input_size, READOUT_HIDDEN),
        nn.ReLU(),
        nn.Linear(READOUT_HIDDEN, CARD_SLOTS),
    )


def make_games(
    count: int, blank_steps: int, device: torch.device, rng: torch.Generator,
    stage: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """정답을 입력에 누출하지 않고 카드 표시→공백→문제 순서의 게임을 만든다."""
    # 한 게임에서 5개 고유 카드를 뽑아 4개를 보여주고 나머지 1개를 교체용으로 둔다.
    if stage not in (1, 2):
        raise ValueError("stage must be 1 or 2")
    cards = torch.rand(count, CARD_TYPES_PER_POOL, device=device, generator=rng).argsort(dim=1)
    # 원본 2단계는 색상/이모지 중 하나를 무작위로 고른다. 두 종류의 ID를
    # 분리해 모델이 다른 카드 종류를 같은 카드로 오인하지 않도록 한다.
    pool_offset = (
        torch.randint(0, 2, (count, 1), device=device, generator=rng) * CARD_TYPES_PER_POOL
        if stage == 2 else 0
    )
    original = cards[:, :CARD_SLOTS] + pool_offset
    new_card = cards[:, CARD_SLOTS] + (pool_offset.squeeze(1) if stage == 2 else 0)
    answer = torch.randint(0, CARD_SLOTS, (count,), device=device, generator=rng)
    changed = original.clone()
    changed[torch.arange(count, device=device), answer] = new_card

    card_features = CARD_SLOTS * CARD_TYPES_PER_POOL * stage
    observations = torch.zeros(count, blank_steps + 2, observation_size(stage), device=device)
    observations[:, 0, :card_features] = F.one_hot(original, CARD_TYPES_PER_POOL * stage).flatten(1).float()
    observations[:, -1, :card_features] = F.one_hot(changed, CARD_TYPES_PER_POOL * stage).flatten(1).float()
    observations[:, 0, card_features] = 1  # 처음 카드가 보이는 시점
    observations[:, 1:-1, card_features + 1] = 1  # 카드가 가려진 지연 구간
    observations[:, -1, card_features + 2] = 1  # 바뀐 카드를 선택하는 시점
    return observations, answer


class FlyPolicy(nn.Module):
    """MaleCNS의 방향성 연결 위치에만 학습 가중치를 두는 게임 정책."""

    def __init__(self, graph_path: Path, leak: float = 0.9, stage: int = 1):
        super().__init__()
        graph = np.load(graph_path)
        self.neurons = len(graph["body_ids"])
        pre = torch.from_numpy(graph["pre_index"].astype(np.int64))
        post = torch.from_numpy(graph["post_index"].astype(np.int64))
        in_degree = torch.bincount(post, minlength=self.neurons).clamp(min=1)
        self.register_buffer("pre", pre)
        self.register_buffer("post", post)
        self.register_buffer("edge_scale", in_degree[post].float().rsqrt())
        self.edge_values = nn.Parameter(torch.randn(len(pre)) * 0.2)
        input_size = observation_size(stage)
        self.input_layer = nn.Linear(input_size, self.neurons)
        self.action_layer = make_readout(self.neurons, input_size)
        self.leak = leak

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch, sequence_length, _ = observations.shape
        state = observations.new_zeros(batch, self.neurons)
        matrix = observations.new_zeros(self.neurons, self.neurons)
        # W[post, pre]로 채워 원본의 pre→post 연결 방향을 유지한다.
        matrix = matrix.index_put(
            (self.post, self.pre), self.edge_values * self.edge_scale, accumulate=True
        )
        for step in range(sequence_length):
            drive = self.input_layer(observations[:, step])
            state = self.leak * state + (1 - self.leak) * torch.tanh(
                drive + F.linear(state, matrix)
            )
        return self.action_layer(torch.cat((state, observations[:, -1]), dim=1))


class BaselinePolicy(nn.Module):
    """동일 관측과 4개 행동을 사용하는 Vanilla RNN 또는 GRU 정책."""

    def __init__(self, kind: str, hidden_size: int, stage: int = 1,
                 rnn_init: str = "default"):
        super().__init__()
        recurrent = nn.RNN if kind == "rnn" else nn.GRU
        input_size = observation_size(stage)
        self.core = recurrent(input_size, hidden_size, batch_first=True)
        self.action_layer = make_readout(hidden_size, input_size)
        if rnn_init == "orthogonal":
            if kind != "rnn":
                raise ValueError("orthogonal recurrent initialization is defined for rnn only")
            # 초기 공백 구간에서 상태가 빠르게 사라지는지 분리해 보기 위해
            # 은닉→은닉 행렬만 직교 초기화한다. 나머지 파라미터는 그대로 둔다.
            # 학습 중 행동 샘플링에 쓰이는 전역 RNG 순서를 유지한다.
            # 따라서 기존 초기화와의 비교에서 다른 파라미터와 행동 난수
            # 흐름은 동일하고, 은닉→은닉 행렬의 초기값만 바뀐다.
            with torch.random.fork_rng(devices=[]):
                nn.init.orthogonal_(self.core.weight_hh_l0)
        elif rnn_init != "default":
            raise ValueError(f"unknown rnn_init: {rnn_init}")

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        states, _ = self.core(observations)
        return self.action_layer(torch.cat((states[:, -1], observations[:, -1]), dim=1))


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def matched_hidden_size(kind: str, target_parameters: int, stage: int = 1) -> int:
    """Fly 모델과 학습 파라미터 수가 가장 가까운 RNN·GRU 크기를 찾는다."""
    def estimated_count(hidden: int) -> int:
        input_size = observation_size(stage)
        readout_constant = input_size * READOUT_HIDDEN + READOUT_HIDDEN
        readout_constant += READOUT_HIDDEN * CARD_SLOTS + CARD_SLOTS
        if kind == "rnn":
            return hidden * hidden + (input_size + 2 + READOUT_HIDDEN) * hidden + readout_constant
        return 3 * hidden * hidden + (3 * input_size + 6 + READOUT_HIDDEN) * hidden + readout_constant

    upper = max(2, math.ceil(math.sqrt(target_parameters)))
    while estimated_count(upper) < target_parameters:
        upper *= 2
    lower = 1
    while lower < upper:
        middle = (lower + upper) // 2
        if estimated_count(middle) < target_parameters:
            lower = middle + 1
        else:
            upper = middle
    return min((max(1, lower - 1), lower), key=lambda h: abs(estimated_count(h) - target_parameters))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def evaluate(
    model: nn.Module, *, count: int, blank_steps: int, device: torch.device, seed: int,
    stage: int = 1,
) -> tuple[float, float]:
    """새 게임의 성공률과 첫 카드 정보를 지운 대조군 성공률을 측정한다."""
    model.eval()
    rng = torch.Generator(device=device).manual_seed(seed)
    correct = 0
    no_memory_correct = 0
    seen = 0
    with torch.no_grad():
        while seen < count:
            batch = min(256, count - seen)
            observations, answer = make_games(batch, blank_steps, device, rng, stage)
            correct += int((model(observations).argmax(1) == answer).sum().item())
            observations[:, 0, :CARD_SLOTS * CARD_TYPES_PER_POOL * stage] = 0
            no_memory_correct += int((model(observations).argmax(1) == answer).sum().item())
            seen += batch
    model.train()
    return correct / count, no_memory_correct / count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["fly", "rnn", "gru"], default="fly")
    parser.add_argument("--rnn-init", choices=["default", "orthogonal"], default="default")
    parser.add_argument("--algorithm", choices=["supervised", "reinforce"], default="reinforce")
    parser.add_argument("--stage", type=int, choices=[1, 2], default=1)
    parser.add_argument("--neurons", type=int, choices=PILOT_NEURON_COUNTS, default=PILOT_NEURON_COUNTS[0])
    parser.add_argument("--graph-dir", type=Path, default=Path("data/subgraphs"))
    parser.add_argument("--graph-variant", choices=["real", "rewired"], default="real")
    parser.add_argument("--rewire-seed", type=int, default=0)
    parser.add_argument("--blank-steps", type=int, default=1)
    parser.add_argument("--updates", type=int, default=1200)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--eval-games", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--entropy-bonus", type=float, default=0.01)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=Path, default=Path("data/game_pilot"))
    args = parser.parse_args()
    if min(args.blank_steps, args.updates, args.batch, args.eval_every, args.eval_games) < 1:
        parser.error("all counts must be positive")
    if args.model != "fly" and args.graph_variant != "real":
        parser.error("rewired variant is only defined for the Fly policy")
    if args.rnn_init != "default" and args.model != "rnn":
        parser.error("non-default rnn-init is only defined for the RNN policy")

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    real_graph_path = args.graph_dir / f"malecns_cx_{args.neurons}.npz"
    graph_path = (
        real_graph_path
        if args.graph_variant == "real"
        else args.graph_dir / f"malecns_cx_{args.neurons}_rewired_seed{args.rewire_seed}.npz"
    )
    fly_reference = FlyPolicy(graph_path, stage=args.stage)
    target_parameters = parameter_count(fly_reference)
    if args.model == "fly":
        model = fly_reference
        hidden_size = args.neurons
    else:
        hidden_size = matched_hidden_size(args.model, target_parameters, stage=args.stage)
        model = BaselinePolicy(args.model, hidden_size, stage=args.stage, rnn_init=args.rnn_init)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    train_rng = torch.Generator(device=device).manual_seed(args.seed + 1)
    history = []

    initial_accuracy, initial_no_memory = evaluate(
        model, count=args.eval_games, blank_steps=args.blank_steps,
        device=device, seed=args.seed + 10_000, stage=args.stage,
    )
    print(f"Initial: accuracy={initial_accuracy:.3f}, no-memory={initial_no_memory:.3f}", flush=True)
    for update in range(1, args.updates + 1):
        observations, answer = make_games(args.batch, args.blank_steps, device, train_rng, args.stage)
        logits = model(observations)
        if args.algorithm == "supervised":
            # 관측·모델이 문제를 풀 수 있는지 보는 진단이다. RL 성능으로 해석하지 않는다.
            loss = F.cross_entropy(logits, answer)
            rewards = (logits.argmax(1) == answer).float()
        else:
            distribution = Categorical(logits=logits)
            actions = distribution.sample()
            rewards = (actions == answer).float()
            # 정답 인덱스는 보상 계산에만 사용한다. 보상 평균은 정책 경사의 기준선이다.
            advantage = rewards - rewards.mean()
            loss = -(advantage.detach() * distribution.log_prob(actions)).mean()
            loss -= args.entropy_bonus * distribution.entropy().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if update == 1 or update % args.eval_every == 0 or update == args.updates:
            accuracy, no_memory = evaluate(
                model, count=args.eval_games, blank_steps=args.blank_steps,
                device=device, seed=args.seed + 10_000, stage=args.stage,
            )
            row = {
                "update": update,
                "choice_interactions": update * args.batch,
                "training_reward": float(rewards.mean().item()),
                "heldout_success_rate": accuracy,
                "no_memory_success_rate": no_memory,
            }
            history.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)

    final_success, final_no_memory = evaluate(
        model, count=args.eval_games, blank_steps=args.blank_steps,
        device=device, seed=args.seed + 20_000, stage=args.stage,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    variant_tag = "" if args.graph_variant == "real" else f"_rewired{args.rewire_seed}"
    stem = (
        f"card_stage{args.stage}_mlphead_{args.algorithm}_{args.model}"
        f"{variant_tag}_{args.neurons}_delay{args.blank_steps}_seed{args.seed}"
    )
    checkpoint = args.output_dir / f"{stem}.pt"
    result_path = args.output_dir / f"{stem}.json"
    result = {
        "source_game": "C:/Users/a/Desktop/자바스크립트 게임/game.js",
        "pilot_rule": (
            "stage 1: four unique colors; choose the replaced slot"
            if args.stage == 1 else
            "stage 2: four unique colors or emojis; choose the replaced slot"
        ),
        "excluded_from_pilot": "real-time timer, animations, fake-shake cue, focus mode, reverse rule, stage progression, stages 3+",
        "algorithm": (
            "cross-entropy with game answer (diagnostic only)"
            if args.algorithm == "supervised"
            else "REINFORCE with batch reward baseline and entropy bonus"
        ),
        "model": args.model,
        "rnn_init": args.rnn_init,
        "orthogonal_rng_preserved": args.rnn_init == "orthogonal",
        "stage": args.stage,
        "graph_variant": args.graph_variant,
        "rewire_seed": args.rewire_seed if args.graph_variant == "rewired" else None,
        "readout": "shared MLP 128 with current observation skip",
        "neurons_reference": args.neurons,
        "hidden_size": hidden_size,
        "parameters": parameter_count(model),
        "fly_reference_parameters": target_parameters,
        "graph_sha256": sha256(graph_path) if args.model == "fly" else None,
        "connection_direction": "pre_index -> post_index; W[post, pre]",
        "blank_steps": args.blank_steps,
        "updates": args.updates,
        "batch": args.batch,
        "learning_rate": args.learning_rate,
        "entropy_bonus": args.entropy_bonus,
        "choice_interactions": args.updates * args.batch,
        "eval_games": args.eval_games,
        "seed": args.seed,
        "random_success_rate": 1 / CARD_SLOTS,
        "initial_success_rate": initial_accuracy,
        "initial_no_memory_success_rate": initial_no_memory,
        "final_fresh_success_rate": final_success,
        "final_fresh_no_memory_success_rate": final_no_memory,
        "history": history,
        "checkpoint": str(checkpoint),
    }
    torch.save({"model_state": model.state_dict(), "result": result}, checkpoint)
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved results: {result_path}", flush=True)
    print(f"Final fresh success: {final_success:.3f}; no-memory: {final_no_memory:.3f}", flush=True)


if __name__ == "__main__":
    main()

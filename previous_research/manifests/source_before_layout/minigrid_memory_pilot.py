"""MaleCNS Fly·재배선 Fly·RNN·GRU의 MiniGrid Memory PPO 파일럿.

모든 모델에 동일한 부분관측 이미지, 방향, 이동 3행동, 보상, 예산을 준다.
에피소드 경계에서 은닉 상태를 초기화하고 전체 rollout을 역전파한다.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import gymnasium as gym
import minigrid  # 가져오면 Gymnasium 환경이 등록되며 버전도 기록한다.
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn import functional as F

from project_config import PILOT_NEURON_COUNTS

ROOT = Path(__file__).resolve().parent
GRAPH_DIR = ROOT / "data" / "subgraphs"
OBJECT_TYPES = 11
COLOR_TYPES = 6
STATE_TYPES = 3
OBSERVATION_SIZE = 7 * 7 * (OBJECT_TYPES + COLOR_TYPES + STATE_TYPES) + 4
FEATURES = 64
ACTIONS = 3  # MiniGrid의 left, right, forward


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_observation(observations: list[dict], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    images = torch.as_tensor(np.stack([item["image"] for item in observations]), device=device, dtype=torch.long)
    directions = torch.as_tensor([item["direction"] for item in observations], device=device, dtype=torch.long)
    return images, directions


def masked_cue_observation(env: gym.Env) -> dict:
    """평가 중 출발 방 물체만 관측에서 가린다. 환경의 정답과 보상은 보존한다."""
    base = env.unwrapped
    x, y = 1, base.height // 2 - 1
    original = base.grid.get(x, y)
    base.grid.set(x, y, None)
    try:
        return base.gen_obs()
    finally:
        base.grid.set(x, y, original)


def create_env(size: int, max_steps: int) -> gym.Env:
    return gym.make(f"MiniGrid-MemoryS{size}-v0", max_steps=max_steps)


def reset_episode(env: gym.Env, seed: int, start_mode: str) -> dict:
    observation, _ = env.reset(seed=seed)
    if start_mode == "fixed_cue":
        # 공식 지형·보상은 사용하되 시작 위치를 단서 옆으로 고정한다.
        # 이 조건에서는 모든 에피소드의 첫 관측에 기억할 물체가 들어온다.
        base = env.unwrapped
        base.agent_pos = np.array((1, base.height // 2))
        base.agent_dir = 0
        observation = base.gen_obs()
    return observation


class ObservationEncoder(nn.Module):
    """보이는 7×7 격자의 세 범주 채널과 방향만 사용한다."""

    def __init__(self) -> None:
        super().__init__()
        self.layer = nn.Sequential(nn.Linear(OBSERVATION_SIZE, 128), nn.ReLU(), nn.Linear(128, FEATURES), nn.ReLU())

    def forward(self, image: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
        objects = F.one_hot(image[..., 0].clamp(0, OBJECT_TYPES - 1), OBJECT_TYPES)
        colors = F.one_hot(image[..., 1].clamp(0, COLOR_TYPES - 1), COLOR_TYPES)
        states = F.one_hot(image[..., 2].clamp(0, STATE_TYPES - 1), STATE_TYPES)
        pixels = torch.cat((objects, colors, states), dim=-1).flatten(1).float()
        encoded = torch.cat((pixels, F.one_hot(direction, 4).float()), dim=1)
        return self.layer(encoded)


class MemoryPolicy(nn.Module):
    """공통 인코더·출력층에 Fly 연결 또는 파라미터 수가 맞는 RNN/GRU를 결합한다."""

    def __init__(self, kind: str, neurons: int, graph_path: Path | None = None,
                 hidden_size: int | None = None, leak: float = 0.9) -> None:
        super().__init__()
        self.kind = kind
        self.encoder = ObservationEncoder()
        if kind == "fly":
            if graph_path is None:
                raise ValueError("Fly 모델에는 그래프 경로가 필요합니다")
            with np.load(graph_path) as graph:
                pre = torch.from_numpy(graph["pre_index"].astype(np.int64))
                post = torch.from_numpy(graph["post_index"].astype(np.int64))
                self.hidden_size = len(graph["body_ids"])
            degree = torch.bincount(post, minlength=self.hidden_size).clamp(min=1)
            self.register_buffer("pre", pre)
            self.register_buffer("post", post)
            self.register_buffer("edge_scale", degree[post].float().rsqrt())
            self.edge_values = nn.Parameter(torch.randn(len(pre)) * 0.2)
            self.input_layer = nn.Linear(FEATURES, self.hidden_size)
            self.leak = leak
        elif kind in ("rnn", "gru"):
            if hidden_size is None:
                raise ValueError("기준 모델에는 hidden_size가 필요합니다")
            self.hidden_size = hidden_size
            cell = nn.RNNCell if kind == "rnn" else nn.GRUCell
            self.core = cell(FEATURES, hidden_size)
        else:
            raise ValueError(f"알 수 없는 모델: {kind}")
        self.head = nn.Sequential(nn.Linear(self.hidden_size + FEATURES, 64), nn.Tanh())
        self.actor = nn.Linear(64, ACTIONS)
        self.critic = nn.Linear(64, 1)
        if kind == "rnn":
            # 카드 게임에서 검증한 설정. 다른 초기 파라미터와 전역 RNG는 유지한다.
            with torch.random.fork_rng(devices=[]):
                nn.init.orthogonal_(self.core.weight_hh)

    def initial_state(self, batch: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch, self.hidden_size, device=device)

    def recurrent_matrix(self) -> torch.Tensor | None:
        if self.kind != "fly":
            return None
        matrix = self.edge_values.new_zeros(self.hidden_size, self.hidden_size)
        # 연결 방향은 원본 pre→post이며 계산 행렬은 W[post, pre]다.
        return matrix.index_put((self.post, self.pre), self.edge_values * self.edge_scale, accumulate=True)

    def step(self, image: torch.Tensor, direction: torch.Tensor, state: torch.Tensor,
             matrix: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.encoder(image, direction)
        if self.kind == "fly":
            if matrix is None:
                matrix = self.recurrent_matrix()
            assert matrix is not None
            state = self.leak * state + (1 - self.leak) * torch.tanh(
                self.input_layer(features) + F.linear(state, matrix)
            )
        else:
            state = self.core(features, state)
        output = self.head(torch.cat((state, features), dim=1))
        return self.actor(output), self.critic(output).squeeze(-1), state

    def trajectory(self, images: torch.Tensor, directions: torch.Tensor,
                   reset_before: torch.Tensor, initial_state: torch.Tensor,
                   matrix: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        logits, values = [], []
        state = initial_state
        if self.kind == "fly" and matrix is None:
            matrix = self.recurrent_matrix()
        for step in range(images.shape[0]):
            state = state * (~reset_before[step]).unsqueeze(1)
            action_logits, value, state = self.step(images[step], directions[step], state, matrix)
            logits.append(action_logits)
            values.append(value)
        return torch.stack(logits), torch.stack(values)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def matched_hidden_size(kind: str, target: int) -> int:
    """공통 인코더·출력층까지 포함한 총 학습 파라미터를 Fly에 가깝게 맞춘다."""
    encoder_count = parameter_count(ObservationEncoder())
    head_constant = 64 * FEATURES + 64 + 64 * ACTIONS + ACTIONS + 64 + 1
    def estimate(hidden: int) -> int:
        recurrent = (hidden * hidden + (FEATURES + 2) * hidden) if kind == "rnn" else (
            3 * hidden * hidden + (3 * FEATURES + 6) * hidden
        )
        return encoder_count + recurrent + 64 * hidden + head_constant
    lower, upper = 1, 2
    while estimate(upper) < target:
        upper *= 2
    while lower < upper:
        middle = (lower + upper) // 2
        if estimate(middle) < target:
            lower = middle + 1
        else:
            upper = middle
    return min((max(1, lower - 1), lower), key=lambda value: abs(estimate(value) - target))


@torch.no_grad()
def evaluate(model: MemoryPolicy, *, size: int, max_steps: int, episodes: int,
             seed: int, device: torch.device, mask_cue: bool = False,
             start_mode: str = "fixed_cue", batch_size: int = 16) -> dict:
    model.eval()
    rewards, successes, lengths = [], [], []
    cue_visible_at_start = 0
    matrix = model.recurrent_matrix()
    for offset in range(0, episodes, batch_size):
        count = min(batch_size, episodes - offset)
        envs = [create_env(size, max_steps) for _ in range(count)]
        try:
            observations = []
            for index, env in enumerate(envs):
                observation = reset_episode(env, seed + offset + index, start_mode)
                masked = masked_cue_observation(env)
                cue_visible_at_start += not np.array_equal(observation["image"], masked["image"])
                observations.append(masked if mask_cue else observation)
            state = model.initial_state(count, device)
            active = np.ones(count, dtype=bool)
            episode_rewards = np.zeros(count, dtype=np.float64)
            episode_lengths = np.zeros(count, dtype=np.int32)
            episode_successes = np.zeros(count, dtype=bool)
            while active.any():
                image, direction = tensor_observation(observations, device)
                logits, _, state = model.step(image, direction, state, matrix)
                actions = logits.argmax(dim=1).cpu().tolist()
                for index, env in enumerate(envs):
                    if not active[index]:
                        continue
                    observation, reward, terminated, truncated, _ = env.step(actions[index])
                    observations[index] = masked_cue_observation(env) if mask_cue else observation
                    episode_rewards[index] += reward
                    episode_lengths[index] += 1
                    if terminated or truncated:
                        episode_successes[index] = reward > 0
                        active[index] = False
            rewards.extend(episode_rewards.tolist())
            successes.extend(episode_successes.tolist())
            lengths.extend(episode_lengths.tolist())
        finally:
            for env in envs:
                env.close()
    return {
        "episodes": episodes,
        "success_rate": float(np.mean(successes)),
        "mean_reward": float(np.mean(rewards)),
        "mean_episode_steps": float(np.mean(lengths)),
        "cue_visible_at_start_rate": cue_visible_at_start / episodes,
    }


def collect_rollout(model: MemoryPolicy, envs: list[gym.Env], observations: list[dict],
                    state: torch.Tensor, *, length: int, device: torch.device,
                    reset_counter: list[int], train_seed_base: int,
                    start_mode: str) -> tuple[dict, list[dict], torch.Tensor, int]:
    model.eval()
    start_state = state.detach().clone()
    images, directions, actions, log_probs, values, rewards, dones, resets = ([] for _ in range(8))
    completed_episodes = 0
    previous_done = torch.zeros(len(envs), dtype=torch.bool, device=device)
    with torch.no_grad():
        matrix = model.recurrent_matrix()
        for _ in range(length):
            image, direction = tensor_observation(observations, device)
            logits, value, next_state = model.step(image, direction, state, matrix)
            distribution = Categorical(logits=logits)
            action = distribution.sample()
            step_rewards = np.zeros(len(envs), dtype=np.float32)
            step_dones = np.zeros(len(envs), dtype=bool)
            next_observations = []
            for index, env in enumerate(envs):
                observation, reward, terminated, truncated, _ = env.step(int(action[index]))
                step_rewards[index] = reward
                if terminated or truncated:
                    step_dones[index] = True
                    completed_episodes += 1
                    reset_counter[index] += 1
                    observation = reset_episode(
                        env, train_seed_base + index * 100_000 + reset_counter[index], start_mode
                    )
                next_observations.append(observation)
            images.append(image)
            directions.append(direction)
            actions.append(action)
            log_probs.append(distribution.log_prob(action))
            values.append(value)
            rewards.append(torch.as_tensor(step_rewards, device=device))
            dones.append(torch.as_tensor(step_dones, device=device))
            resets.append(previous_done)
            previous_done = torch.as_tensor(step_dones, device=device)
            state = next_state * (~previous_done).unsqueeze(1)
            observations = next_observations
        final_image, final_direction = tensor_observation(observations, device)
        _, final_value, _ = model.step(final_image, final_direction, state, matrix)
    rollout = {
        "images": torch.stack(images), "directions": torch.stack(directions),
        "actions": torch.stack(actions), "old_log_probs": torch.stack(log_probs),
        "values": torch.stack(values), "rewards": torch.stack(rewards),
        "dones": torch.stack(dones), "reset_before": torch.stack(resets),
        "initial_state": start_state, "final_value": final_value,
    }
    return rollout, observations, state.detach(), completed_episodes


def advantages_and_returns(rollout: dict, gamma: float = 0.99,
                           gae_lambda: float = 0.95) -> tuple[torch.Tensor, torch.Tensor]:
    rewards, values, dones = rollout["rewards"], rollout["values"], rollout["dones"]
    advantages = torch.zeros_like(rewards)
    following_value = rollout["final_value"]
    following_advantage = torch.zeros_like(following_value)
    for step in reversed(range(rewards.shape[0])):
        continuity = (~dones[step]).float()
        delta = rewards[step] + gamma * continuity * following_value - values[step]
        following_advantage = delta + gamma * gae_lambda * continuity * following_advantage
        advantages[step] = following_advantage
        following_value = values[step]
    return advantages, advantages + values


def update_ppo(model: MemoryPolicy, optimizer: torch.optim.Optimizer, rollout: dict,
               advantages: torch.Tensor, returns: torch.Tensor, *, epochs: int,
               minibatch_envs: int, entropy_bonus: float) -> dict:
    model.train()
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    batch = rollout["actions"].shape[1]
    losses, entropies = [], []
    for _ in range(epochs):
        order = torch.randperm(batch, device=advantages.device)
        for indices in order.split(minibatch_envs):
            logits, predicted_values = model.trajectory(
                rollout["images"][:, indices], rollout["directions"][:, indices],
                rollout["reset_before"][:, indices], rollout["initial_state"][indices],
            )
            distribution = Categorical(logits=logits)
            new_log_probs = distribution.log_prob(rollout["actions"][:, indices])
            ratio = (new_log_probs - rollout["old_log_probs"][:, indices]).exp()
            advantage = advantages[:, indices]
            policy_loss = -torch.minimum(
                ratio * advantage, ratio.clamp(0.8, 1.2) * advantage
            ).mean()
            value_loss = 0.5 * F.mse_loss(predicted_values, returns[:, indices])
            entropy = distribution.entropy().mean()
            loss = policy_loss + 0.5 * value_loss - entropy_bonus * entropy
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            losses.append(float(loss.detach()))
            entropies.append(float(entropy.detach()))
    return {"loss": float(np.mean(losses)), "policy_entropy": float(np.mean(entropies))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("fly", "rewired", "rnn", "gru"), required=True)
    parser.add_argument("--neurons", type=int, choices=PILOT_NEURON_COUNTS, default=1000)
    parser.add_argument("--size", type=int, choices=(7, 11, 13), default=11)
    parser.add_argument("--max-episode-steps", type=int, default=100)
    parser.add_argument("--start-mode", choices=("fixed_cue", "default"), default="fixed_cue")
    parser.add_argument("--steps", type=int, default=10240)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--rollout-steps", type=int, default=32)
    parser.add_argument("--ppo-epochs", type=int, default=2)
    parser.add_argument("--minibatch-envs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--entropy-bonus", type=float, default=0.01)
    parser.add_argument("--eval-episodes", type=int, default=64)
    parser.add_argument("--eval-every", type=int, default=2560)
    parser.add_argument("--seed", type=int, default=301)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "minigrid_pilot")
    parser.add_argument("--init-checkpoint", type=Path, default=None,
                        help="사전 학습 체크포인트의 모델 가중치에서 PPO를 시작합니다")
    parser.add_argument("--select-best-checkpoint", action="store_true",
                        help="학습과 분리한 검증 시드의 성공률·보상으로 초기/중간/최종 가중치를 선택")
    args = parser.parse_args()
    if min(args.steps, args.num_envs, args.rollout_steps, args.ppo_epochs,
           args.minibatch_envs, args.eval_episodes, args.eval_every) < 1:
        parser.error("카운트 인자는 양수여야 합니다")
    if args.num_envs % args.minibatch_envs:
        parser.error("num-envs는 minibatch-envs의 배수여야 합니다")
    stride = args.num_envs * args.rollout_steps
    if args.steps % stride:
        parser.error(f"steps는 한 rollout의 {stride} 상호작용의 배수여야 합니다")
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    graph_name = f"malecns_cx_{args.neurons}"
    graph_path = GRAPH_DIR / f"{graph_name}{'_rewired_seed0' if args.model == 'rewired' else ''}.npz"
    # 기준 모델의 크기는 같은 실제 Fly 그래프의 전체 학습 파라미터 수에 맞춘다.
    initial_rng_state = torch.random.get_rng_state()
    reference = MemoryPolicy("fly", args.neurons, GRAPH_DIR / f"{graph_name}.npz")
    target_parameters = parameter_count(reference)
    if args.model == "fly":
        model = reference
    elif args.model == "rewired":
        # 동일 시드의 실제 그래프와 인코더·출력층·간선 초기값을 맞춘다.
        # fork_rng로 이후 PPO 행동 샘플링에 쓰일 RNG 흐름도 유지한다.
        with torch.random.fork_rng(devices=[]):
            torch.random.set_rng_state(initial_rng_state)
            model = MemoryPolicy("fly", args.neurons, graph_path)
    else:
        hidden = matched_hidden_size(args.model, target_parameters)
        model = MemoryPolicy(args.model, args.neurons, hidden_size=hidden)
    model.to(device)
    if args.init_checkpoint is not None:
        initial_checkpoint = torch.load(args.init_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(initial_checkpoint["model_state"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    envs = [create_env(args.size, args.max_episode_steps) for _ in range(args.num_envs)]
    train_seed_base = args.seed * 1_000_000
    reset_counter = [0] * args.num_envs
    observations = [reset_episode(env, train_seed_base + index * 100_000, args.start_mode)
                    for index, env in enumerate(envs)]
    state = model.initial_state(args.num_envs, device)
    eval_seed = 500_000 + args.seed * 1_000
    history = []
    start_time = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    try:
        initial = evaluate(model, size=args.size, max_steps=args.max_episode_steps,
                           episodes=args.eval_episodes, seed=eval_seed, device=device,
                           start_mode=args.start_mode)
        selected_step = 0
        selected_validation = initial
        selected_state = copy.deepcopy(model.state_dict()) if args.select_best_checkpoint else None
        print(json.dumps({"step": 0, **initial}, ensure_ascii=False), flush=True)
        for step in range(stride, args.steps + 1, stride):
            rollout, observations, state, completed_episodes = collect_rollout(
                model, envs, observations, state, length=args.rollout_steps, device=device,
                reset_counter=reset_counter, train_seed_base=train_seed_base,
                start_mode=args.start_mode,
            )
            advantages, returns = advantages_and_returns(rollout)
            losses = update_ppo(model, optimizer, rollout, advantages, returns,
                                epochs=args.ppo_epochs, minibatch_envs=args.minibatch_envs,
                                entropy_bonus=args.entropy_bonus)
            if step % args.eval_every == 0 or step == args.steps:
                measured = evaluate(model, size=args.size, max_steps=args.max_episode_steps,
                                    episodes=args.eval_episodes, seed=eval_seed, device=device,
                                    start_mode=args.start_mode)
                row = {"interaction_steps": step, **measured, **losses,
                       "completed_train_episodes_in_rollout": completed_episodes}
                history.append(row)
                print(json.dumps(row, ensure_ascii=False), flush=True)
                if args.select_best_checkpoint and (
                    measured["success_rate"], measured["mean_reward"]
                ) > (selected_validation["success_rate"], selected_validation["mean_reward"]):
                    selected_step = step
                    selected_validation = measured
                    selected_state = copy.deepcopy(model.state_dict())
        if selected_state is not None:
            model.load_state_dict(selected_state)
        final = evaluate(model, size=args.size, max_steps=args.max_episode_steps,
                         episodes=args.eval_episodes, seed=eval_seed + 100_000, device=device,
                         start_mode=args.start_mode)
        masked = evaluate(model, size=args.size, max_steps=args.max_episode_steps,
                          episodes=args.eval_episodes, seed=eval_seed + 100_000,
                          device=device, mask_cue=True, start_mode=args.start_mode)
        default_start = evaluate(model, size=args.size, max_steps=args.max_episode_steps,
                                 episodes=args.eval_episodes, seed=eval_seed + 100_000,
                                 device=device, start_mode="default")
    finally:
        for env in envs:
            env.close()
    elapsed = time.perf_counter() - start_time
    result = {
        "environment": f"MiniGrid-MemoryS{args.size}-v0", "environment_version": minigrid.__version__,
        "gymnasium_version": gym.__version__, "model": args.model,
        "neurons_reference": args.neurons, "hidden_size": model.hidden_size,
        "parameters": parameter_count(model), "fly_reference_parameters": target_parameters,
        "graph_sha256": sha256(graph_path) if args.model in ("fly", "rewired") else None,
        "paired_rewire_initialization": args.model == "rewired",
        "connection_direction": "pre_index -> post_index; W[post, pre]",
        "observation": "partial 7x7 image (one-hot object/color/state) and direction; constant mission omitted",
        "actions": "left, right, forward; original success reward",
        "algorithm": "recurrent PPO with GAE; full rollout BPTT",
        "rnn_recurrent_init": "orthogonal" if args.model == "rnn" else None,
        "max_episode_steps": args.max_episode_steps, "steps": args.steps,
        "init_checkpoint": str(args.init_checkpoint) if args.init_checkpoint is not None else None,
        "select_best_checkpoint": args.select_best_checkpoint,
        "selected_checkpoint_step": selected_step if args.select_best_checkpoint else None,
        "selected_validation": selected_validation if args.select_best_checkpoint else None,
        "start_mode": args.start_mode,
        "num_envs": args.num_envs, "rollout_steps": args.rollout_steps,
        "ppo_epochs": args.ppo_epochs, "minibatch_envs": args.minibatch_envs,
        "learning_rate": args.learning_rate, "entropy_bonus": args.entropy_bonus,
        "eval_episodes": args.eval_episodes, "seed": args.seed,
        "evaluation_seed": eval_seed, "device": str(device),
        "duration_seconds": elapsed,
        "peak_vram_bytes_pytorch": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        "initial_evaluation": initial, "history": history,
        "final_fresh_evaluation": final, "final_fresh_cue_masked_evaluation": masked,
        "final_fresh_default_start_evaluation": default_start,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"memory_s{args.size}_{args.model}_{args.neurons}_seed{args.seed}"
    json_path = args.output_dir / f"{stem}.json"
    checkpoint_path = args.output_dir / f"{stem}.pt"
    torch.save({"model_state": model.state_dict(), "result": result}, checkpoint_path)
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {json_path}", flush=True)


if __name__ == "__main__":
    main()

"""Reward-only recurrent PPO pilot against a common random opponent.

Run from the workspace root: .venv/Scripts/python.exe -m gomoku9.train
All policies are new; previous memory-task weights are never loaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .env import LearnerEnv
from .models import KINDS, Policy, count_parameters, masked_distribution, matched_hidden

ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = ROOT / "runs" / "pilot_20261006"


def atomic_json(path: Path, value: dict) -> None:
    temp = path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def atomic_torch(path: Path, value: dict) -> None:
    temp = path.with_suffix(".pt.tmp")
    torch.save(value, temp)
    temp.replace(path)


def advantages(rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor,
               bootstrap: torch.Tensor, gamma: float = 0.99, lam: float = 0.95
               ) -> tuple[torch.Tensor, torch.Tensor]:
    adv = torch.zeros_like(rewards)
    carry = torch.zeros_like(bootstrap)
    for t in reversed(range(len(rewards))):
        next_v = bootstrap if t == len(rewards) - 1 else values[t + 1]
        alive = (~dones[t]).float()
        delta = rewards[t] + gamma * next_v * alive - values[t]
        carry = delta + gamma * lam * alive * carry
        adv[t] = carry
    return adv, adv + values


def train_one(kind: str, seed: int, args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    stem = args.output / f"{kind}_seed{seed}"
    ckpt, result_path = stem.with_suffix(".pt"), stem.with_suffix(".json")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    config = dict(kind=kind, seed=seed, hidden=matched_hidden(kind), size=9,
                  rewire_seed=seed % 5, updates=args.updates, envs=args.envs,
                  rollout=args.rollout, epochs=args.epochs, lr=args.lr,
                  opponent="random", gamma=0.99, gae_lambda=0.95, ppo_clip=0.2,
                  entropy_coefficient=0.01, value_coefficient=0.5,
                  observation="full board: own, opponent, last move, black-to-play color",
                  reward="terminal +1/-1, draw 0; no shaping, teacher or search",
                  initialization="fresh; archived task checkpoints not loaded")
    model = Policy(kind, seed, config["hidden"], rewire_seed=seed % 5).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0)
    env = LearnerEnv(args.envs, seed=1_000_000 + seed)
    state = model.initial(args.envs, device)
    resets = torch.ones(args.envs, dtype=torch.bool, device=device)
    start, curve, outcomes, plies, elapsed_before = 0, [], [], [], 0.0
    if ckpt.exists():
        saved = torch.load(ckpt, map_location="cpu", weights_only=True)
        if saved["config"] != config:
            raise ValueError(f"Existing checkpoint has a different configuration: {ckpt}")
        model.load_state_dict(saved["model"])
        if model.graph_sha256 != saved["graph_sha256"]:
            raise ValueError("Graph hash changed")
        start = saved["completed_updates"]
        if start == args.updates:
            return json.loads(result_path.read_text(encoding="utf-8"))
        optimizer.load_state_dict(saved["optimizer"])
        state, resets = saved["actor_state"].to(device), saved["resets"].to(device)
        for name in ("board", "line_counts", "moves", "last", "finished", "winner"):
            setattr(env.games, name, saved["environment"][name].numpy().copy())
        env.colors = saved["environment"]["colors"].numpy().copy()
        env.rng.bit_generator.state = saved["environment"]["rng"]
        torch.set_rng_state(saved["torch_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(saved["cuda_rng"], device)
        curve, outcomes, plies = saved["curve"], saved["outcomes"], saved["plies"]
        elapsed_before = saved["elapsed_seconds"]
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()

    def save(update: int) -> dict:
        elapsed = elapsed_before + time.perf_counter() - started
        recent = np.asarray(outcomes[-1000:])
        result = dict(config=config, parameters=count_parameters(model),
                      graph_sha256=model.graph_sha256, completed_updates=update,
                      learner_decisions=update * args.envs * args.rollout,
                      completed_training_games=len(outcomes),
                      last_1000_training_win_rate=float(np.mean(recent == 1)) if len(recent) else None,
                      last_1000_training_draw_rate=float(np.mean(recent == 0)) if len(recent) else None,
                      last_1000_training_mean_plies=float(np.mean(plies[-1000:])) if plies else None,
                      elapsed_seconds=elapsed, curve=curve,
                      peak_allocated_gib=torch.cuda.max_memory_allocated(device) / 2**30
                      if device.type == "cuda" else None,
                      peak_reserved_gib=torch.cuda.max_memory_reserved(device) / 2**30
                      if device.type == "cuda" else None,
                      torch_version=str(torch.__version__),
                      device=torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
                      tf32=False,
                      source_hashes={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                     for p in ROOT.glob("*.py")})
        environment = {name: torch.from_numpy(getattr(env.games, name).copy())
                       for name in ("board", "line_counts", "moves", "last", "finished", "winner")}
        environment.update(colors=torch.from_numpy(env.colors.copy()), rng=env.rng.bit_generator.state)
        atomic_torch(ckpt, dict(model={k: v.detach().cpu() for k, v in model.state_dict().items()},
                               optimizer=optimizer.state_dict(), graph_sha256=model.graph_sha256,
                               config=config, completed_updates=update, actor_state=state.detach().cpu(),
                               resets=resets.cpu(), environment=environment,
                               torch_rng=torch.get_rng_state(),
                               cuda_rng=torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
                               curve=curve, outcomes=outcomes, plies=plies, elapsed_seconds=elapsed))
        atomic_json(result_path, result)
        return result

    print(json.dumps(dict(event="training_start", kind=kind, seed=seed,
                          hidden=model.hidden, parameters=count_parameters(model), resumed_at=start)), flush=True)
    for update in range(start + 1, args.updates + 1):
        initial_state = state.detach().clone()
        observations, legal_masks, reset_masks = [], [], []
        actions, old_logs, values, rewards, dones = [], [], [], [], []
        with torch.no_grad():
            matrix = model.matrix()
            for _ in range(args.rollout):
                obs_np, legal_np = env.observe()
                obs = torch.from_numpy(obs_np).to(device)
                legal = torch.from_numpy(legal_np).to(device)
                logits, value, next_state = model.step(obs, state, resets, matrix)
                distribution = masked_distribution(logits, legal)
                action = distribution.sample()
                reward_np, done_np, info = env.step(action.cpu().numpy())
                observations.append(obs)
                legal_masks.append(legal)
                reset_masks.append(resets)
                actions.append(action)
                old_logs.append(distribution.log_prob(action))
                values.append(value)
                rewards.append(torch.from_numpy(reward_np).to(device))
                dones.append(torch.from_numpy(done_np).to(device))
                outcomes.extend(info["results"].tolist())
                plies.extend(info["plies"].tolist())
                state, resets = next_state, dones[-1]
            obs_end, _ = env.observe()
            _, bootstrap, _ = model.step(torch.from_numpy(obs_end).to(device), state, resets, matrix)
            old_value = torch.stack(values)
            done_tensor = torch.stack(dones)
            adv, targets = advantages(torch.stack(rewards), old_value, done_tensor, bootstrap)
            adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
        obs_tensor, legal_tensor = torch.stack(observations), torch.stack(legal_masks)
        reset_tensor, action_tensor = torch.stack(reset_masks), torch.stack(actions)
        old_log = torch.stack(old_logs)
        for _ in range(args.epochs):
            logits, new_value = model.sequence(obs_tensor, initial_state, reset_tensor)
            distribution = masked_distribution(logits, legal_tensor)
            log_prob = distribution.log_prob(action_tensor)
            ratio = (log_prob - old_log).exp()
            policy_loss = -torch.minimum(ratio * adv, ratio.clamp(0.8, 1.2) * adv).mean()
            value_loss = F.mse_loss(new_value, targets)
            entropy = distribution.entropy().mean()
            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite PPO loss at {kind}/{seed}/{update}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
        # Collection state is retained across chunks and reset at episode boundaries.
        state = state.detach()
        if update == 1 or update % 25 == 0 or update == args.updates:
            recent = np.asarray(outcomes[-1000:])
            row = dict(update=update, training_win_rate=float(np.mean(recent == 1)) if len(recent) else None,
                       games=len(outcomes), policy_loss=float(policy_loss.detach()),
                       value_loss=float(value_loss.detach()), entropy=float(entropy.detach()),
                       seconds=round(elapsed_before + time.perf_counter() - started, 2))
            curve.append(row)
            print(json.dumps(dict(kind=kind, seed=seed, **row)), flush=True)
        if update % 100 == 0 or update == args.updates:
            result = save(update)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=KINDS, default=list(KINDS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[4101, 4102, 4103])
    parser.add_argument("--updates", type=int, default=400)
    parser.add_argument("--envs", type=int, default=32)
    parser.add_argument("--rollout", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if min(args.updates, args.envs, args.rollout, args.epochs) < 1:
        parser.error("All budgets must be positive")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    rows = []
    for seed in args.seeds:
        for kind in args.models:
            rows.append(train_one(kind, seed, args))
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()
    atomic_json(args.output / "training_summary.json", {"runs": rows})


if __name__ == "__main__":
    main()

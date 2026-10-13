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
from .tactics import TacticalData
from .history_aux import HistoryTacticalData

ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = ROOT / "runs" / "pilot_20261006"


def replace_with_retry(temp: Path, path: Path) -> None:
    # Windows refuses replacing a checkpoint during a concurrent short read.
    # Retain atomic replacement; never truncate the previous valid checkpoint.
    for attempt in range(40):
        try:
            temp.replace(path)
            return
        except PermissionError:
            if attempt == 39:
                raise
            time.sleep(0.05)


def atomic_json(path: Path, value: dict) -> None:
    temp = path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    replace_with_retry(temp, path)


def atomic_torch(path: Path, value: dict) -> None:
    temp = path.with_suffix(".pt.tmp")
    torch.save(value, temp)
    replace_with_retry(temp, path)


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
    initial_checkpoint = args.initialize_from / f"{kind}_seed{seed}.pt" if args.initialize_from else None
    defense_path = args.defense_data_dir / f"{kind}_seed{seed}_dagger.npz" if args.defense_data_dir else None
    attack_path = args.attack_data_dir / f"{kind}_seed{seed}_attack.npz" if args.attack_data_dir else None
    preparation_path = (args.preparation_data_dir / f"{kind}_seed{seed}_prep12.npz"
                        if args.preparation_data_dir else None)
    if preparation_path:
        fixed_quotas = {
            "base": args.fixed_base_batch or 32,
            "defense": args.fixed_defense_batch or 32,
            "attack": args.fixed_attack_batch or 32,
            "preparation": args.fixed_preparation_batch or 32,
        }
    else:
        fixed_quotas = {"base": 64, "defense": 32, "attack": 32}
    if defense_path and (any(value < 1 for value in fixed_quotas.values()) or
                         sum(fixed_quotas.values()) != args.auxiliary_batch):
        raise ValueError(f"Fixed quotas must be positive and sum to auxiliary batch: {fixed_quotas}")
    ckpt, result_path = stem.with_suffix(".pt"), stem.with_suffix(".json")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    config = dict(kind=kind, seed=seed, hidden=matched_hidden(kind, args.size), size=args.size,
                  rewire_seed=seed % 5, updates=args.updates, envs=args.envs,
                  rollout=args.rollout, epochs=args.epochs, lr=args.lr,
                  opponent=args.opponent, warmup_updates=args.warmup_updates,
                  auxiliary_batch=args.auxiliary_batch, auxiliary_weight=args.auxiliary_weight,
                  tactical_data_sha256=hashlib.sha256(args.tactical_data.read_bytes()).hexdigest()
                  if args.tactical_data else None,
                  gamma=0.99, gae_lambda=0.95, ppo_clip=0.2,
                  entropy_coefficient=0.01, value_coefficient=0.5,
                  observation="full board: own, opponent, last move, black-to-play color",
                  reward="terminal +1/-1, draw 0; no shaping or search; auxiliary tactical labels if configured",
                  initialization=(
                      f"fine-tune weights from {initial_checkpoint.name}; optimizer reset; "
                      f"sha256={hashlib.sha256(initial_checkpoint.read_bytes()).hexdigest()}"
                      if initial_checkpoint else "fresh; archived task checkpoints not loaded"
                  ))
    if args.auxiliary_mode != "cold" or args.architecture != "global" or args.curriculum:
        config.update(auxiliary_mode=args.auxiliary_mode, architecture=args.architecture,
                      curriculum=args.curriculum, history_gradient="detached full-history burn-in; target-step gradient")
    if args.start_data or args.balanced_auxiliary:
        config.update(start_data_sha256=hashlib.sha256(args.start_data.read_bytes()).hexdigest() if args.start_data else None,
                      empty_probability=args.empty_probability,balanced_auxiliary=args.balanced_auxiliary,
                      midgame_prefix="current-weight detached burn-in at every reset in collection and PPO")
    if defense_path or attack_path:
        mixture = fixed_quotas
        config.update(
            fixed_auxiliary_mixture=mixture,
            defense_data_sha256=hashlib.sha256(defense_path.read_bytes()).hexdigest(),
            attack_data_sha256=hashlib.sha256(attack_path.read_bytes()).hexdigest(),
            attack_collection="missed first move plus tactical continuation from policy-visited curriculum starts",
        )
        if preparation_path:
            with np.load(preparation_path) as preparation_saved:
                grouped_preparation = "root_keys" in preparation_saved
            config.update(
                preparation_data_sha256=hashlib.sha256(preparation_path.read_bytes()).hexdigest(),
                preparation_teacher="verified multi-response tree" if grouped_preparation else "beam search through six own setup moves against tactical replies",
                preparation_sampling="uniform root then uniform record" if grouped_preparation else "uniform record",
            )
    if args.dagger_games:
        config.update(
            dagger_games=args.dagger_games,
            dagger_rounds=args.dagger_rounds,
            dagger_fraction=args.dagger_fraction,
            dagger_holdout_sha256={
                str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in args.dagger_holdouts
            },
            dagger_policy="deterministic current policy; first certified unsafe action from each loss",
        )
    if args.threat_data:
        config.update(
            threat_data_sha256=hashlib.sha256(args.threat_data.read_bytes()).hexdigest(),
            threat_batch=args.threat_batch,
            threat_weight=args.threat_weight,
            threat_target="candidate-conditioned opponent immediate-win and double-threat maps",
        )
    cls = Policy
    if args.architecture in ("spatial", "spatial_threat"):
        from .spatial import SpatialPolicy
        cls = SpatialPolicy
    model_kwargs = {"threat_aux": True} if args.architecture == "spatial_threat" else {}
    model = cls(kind, seed, config["hidden"], size=args.size, rewire_seed=seed % 5,
                **model_kwargs).to(device)
    if initial_checkpoint:
        source = torch.load(initial_checkpoint, map_location="cpu", weights_only=True)
        source_config = source["config"]
        if (source_config["kind"] != kind or source_config["size"] != args.size or
                source_config.get("architecture", "global") != args.architecture):
            raise ValueError(f"Incompatible initialization checkpoint: {initial_checkpoint}")
        if source["graph_sha256"] != model.graph_sha256:
            raise ValueError(f"Graph hash differs from initialization checkpoint: {initial_checkpoint}")
        model.load_state_dict(source["model"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0)
    env = LearnerEnv(args.envs, seed=1_000_000 + seed, size=args.size, opponent=args.opponent)
    if args.start_data:
        from .midgame_env import MidgameEnv, restore_prefix_state, prefix_sequence
        env = MidgameEnv(args.envs,1_000_000+seed,args.size,args.opponent,args.start_data,args.empty_probability)
    if defense_path and attack_path:
        from .offense_aux import FixedMixtureData
        auxiliary = FixedMixtureData(
            args.tactical_data, defense_path, attack_path, device, args.auxiliary_mode,
            base_batch=fixed_quotas["base"],
            defense_batch=fixed_quotas["defense"],
            attack_batch=fixed_quotas["attack"],
            preparation_path=preparation_path,
            preparation_batch=fixed_quotas.get("preparation", 0),
        )
    else:
        auxiliary = HistoryTacticalData(args.tactical_data, device, args.auxiliary_mode) if args.tactical_data else None
    if auxiliary:
        auxiliary.balanced = args.balanced_auxiliary
    dagger_path = stem.with_name(stem.name + "_dagger").with_suffix(".npz")
    if args.dagger_games and dagger_path.exists():
        auxiliary.append_file(dagger_path, args.dagger_fraction)
    threat = None
    if args.threat_data:
        from .threat_aux import ThreatData
        threat = ThreatData(args.threat_data, device)
    dagger_stats = []
    midgame_outcomes = {str(i):dict(games=0,wins=0,within_horizon_wins=0) for i in range(4)}
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
        if args.start_data:
            for name in ("histories","start_horizon","start_plies","start_ids"):
                setattr(env,name,saved["environment"][name].numpy().copy())
            env.start_counts=saved["environment"]["start_counts"]
            env.prefix_frames=saved.get("rollout_prefix_frames",0)
            midgame_outcomes=saved["midgame_outcomes"]
        torch.set_rng_state(saved["torch_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(saved["cuda_rng"], device)
        curve, outcomes, plies = saved["curve"], saved["outcomes"], saved["plies"]
        if auxiliary:
            auxiliary.frames_used = saved.get("auxiliary_history_frames", 0)
            auxiliary.samples_by_category = saved.get("auxiliary_samples_by_category",auxiliary.samples_by_category)
        if threat:
            threat.frames_used = saved.get("threat_history_frames", 0)
            threat.samples = saved.get("threat_samples", 0)
        dagger_stats = saved.get("dagger_stats", [])
        elapsed_before = saved["elapsed_seconds"]
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()

    if start == 0 and args.warmup_updates:
        if auxiliary is None:
            raise ValueError("Warmup requires tactical training data")
        for step in range(1, args.warmup_updates + 1):
            loss = auxiliary.loss(model, args.auxiliary_batch)
            if threat is not None:
                loss = loss + args.threat_weight * threat.loss(model, args.threat_batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            if step % 200 == 0 or step == args.warmup_updates:
                print(json.dumps(dict(event="tactical_warmup", kind=kind, seed=seed,
                                      step=step, loss=float(loss.detach()),
                                      seconds=round(time.perf_counter() - started, 2))), flush=True)
        atomic_torch(stem.with_name(stem.name + "_warmup").with_suffix(".pt"),
                     dict(model={k: v.detach().cpu() for k, v in model.state_dict().items()},
                          graph_sha256=model.graph_sha256, config=config, completed_updates=0))

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
                      auxiliary_history_frames=auxiliary.frames_used if auxiliary else 0,
                      threat_history_frames=threat.frames_used if threat else 0,
                      threat_samples=threat.samples if threat else 0,
                      dagger_stats=dagger_stats,
                      dagger_records=(auxiliary.dagger_records if hasattr(auxiliary, "dagger_records") else
                                      len(auxiliary.obs) - auxiliary.base_count) if auxiliary else 0,
                      source_hashes={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                     for p in ROOT.glob("*.py")})
        result.update(auxiliary_samples_by_category=auxiliary.samples_by_category if auxiliary else {},
                      midgame_outcomes=midgame_outcomes if args.start_data else {},
                      episode_starts=env.start_counts if args.start_data else {},
                      rollout_prefix_frames=env.prefix_frames if args.start_data else 0)
        environment = {name: torch.from_numpy(getattr(env.games, name).copy())
                       for name in ("board", "line_counts", "moves", "last", "finished", "winner")}
        environment.update(colors=torch.from_numpy(env.colors.copy()), rng=env.rng.bit_generator.state)
        if args.start_data:
            environment.update({name:torch.from_numpy(getattr(env,name).copy())
                                for name in ("histories","start_horizon","start_plies","start_ids")})
            environment["start_counts"]=env.start_counts
        atomic_torch(ckpt, dict(model={k: v.detach().cpu() for k, v in model.state_dict().items()},
                               optimizer=optimizer.state_dict(), graph_sha256=model.graph_sha256,
                               config=config, completed_updates=update, actor_state=state.detach().cpu(),
                               resets=resets.cpu(), environment=environment,
                               torch_rng=torch.get_rng_state(),
                               cuda_rng=torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
                               auxiliary_history_frames=auxiliary.frames_used if auxiliary else 0,
                               auxiliary_samples_by_category=auxiliary.samples_by_category if auxiliary else {},
                               threat_history_frames=threat.frames_used if threat else 0,
                               threat_samples=threat.samples if threat else 0,
                               dagger_stats=dagger_stats,
                               midgame_outcomes=midgame_outcomes,
                               rollout_prefix_frames=env.prefix_frames if args.start_data else 0,
                               curve=curve, outcomes=outcomes, plies=plies, elapsed_seconds=elapsed))
        atomic_json(result_path, result)
        return result

    print(json.dumps(dict(event="training_start", kind=kind, seed=seed,
                          hidden=model.hidden, parameters=count_parameters(model), resumed_at=start)), flush=True)
    for update in range(start + 1, args.updates + 1):
        if args.dagger_games:
            interval = args.updates // args.dagger_rounds
            round_index = (update - 1) // interval
            boundary = 1 + round_index * interval
            completed_rounds = {row["round"] for row in dagger_stats}
            if update == boundary and round_index not in completed_rounds:
                from .dagger import (collect as collect_dagger, heldout_canonical_keys,
                                     record_canonical_keys, merge)
                excluded = heldout_canonical_keys(args.dagger_holdouts)
                excluded |= record_canonical_keys(dagger_path)
                records, stats = collect_dagger(
                    model, args.dagger_games,
                    220_000_000 + seed + 10_000 * round_index,
                    device, excluded, round_index,
                )
                aggregate = merge(dagger_path, records)
                auxiliary.append_file(dagger_path, args.dagger_fraction)
                stats["aggregate_records"] = len(aggregate)
                dagger_stats.append(stats)
                print(json.dumps(dict(event="dagger_collection", kind=kind, seed=seed, **stats)),
                      flush=True)
        if args.curriculum:
            fraction = update / args.updates
            env.opponent = "weak" if fraction <= 1 / 3 else ("medium" if fraction <= 2 / 3 else "hard")
        if args.start_data:
            env.level = min(3,1+(update-1)*3//args.updates)
        initial_state = state.detach().clone()
        prefix_histories=[]
        observations, legal_masks, reset_masks = [], [], []
        actions, old_logs, values, rewards, dones = [], [], [], [], []
        with torch.no_grad():
            matrix = model.matrix()
            for _ in range(args.rollout):
                obs_np, legal_np = env.observe()
                obs = torch.from_numpy(obs_np).to(device)
                legal = torch.from_numpy(legal_np).to(device)
                effective_resets=resets
                if args.start_data:
                    hist=torch.from_numpy(env.histories.astype(np.int64)).to(device)
                    prefix_histories.append(hist)
                    env.prefix_frames += int(((hist[resets]>=0).sum(1)//2).sum())
                    state=restore_prefix_state(model,state,resets,hist)
                    effective_resets=torch.zeros_like(resets)
                logits, value, next_state = model.step(obs, state, effective_resets, matrix)
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
                if args.start_data:
                    for d,r,n in zip(info["start_horizon"],info["results"],info["episode_decisions"]):
                        stats=midgame_outcomes[str(int(d))]
                        stats["games"]+=1
                        stats["wins"]+=int(r==1)
                        stats["within_horizon_wins"]+=int(r==1 and n<=d) if d else 0
                outcomes.extend(info["results"].tolist())
                plies.extend(info["plies"].tolist())
                state, resets = next_state, dones[-1]
            obs_end, _ = env.observe()
            bootstrap_state=state
            bootstrap_resets=resets
            if args.start_data:
                bootstrap_state=restore_prefix_state(model,state,resets,torch.from_numpy(env.histories.astype(np.int64)).to(device))
                env.prefix_frames += int(((env.histories[resets.cpu().numpy()]>=0).sum(1)//2).sum())
                bootstrap_resets=torch.zeros_like(resets)
            _, bootstrap, _ = model.step(torch.from_numpy(obs_end).to(device), bootstrap_state, bootstrap_resets, matrix)
            old_value = torch.stack(values)
            done_tensor = torch.stack(dones)
            adv, targets = advantages(torch.stack(rewards), old_value, done_tensor, bootstrap)
            adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
        obs_tensor, legal_tensor = torch.stack(observations), torch.stack(legal_masks)
        reset_tensor, action_tensor = torch.stack(reset_masks), torch.stack(actions)
        old_log = torch.stack(old_logs)
        for _ in range(args.epochs):
            if args.start_data:
                logits,new_value=prefix_sequence(model,obs_tensor,initial_state,reset_tensor,torch.stack(prefix_histories))
            else:
                logits, new_value = model.sequence(obs_tensor, initial_state, reset_tensor)
            distribution = masked_distribution(logits, legal_tensor)
            log_prob = distribution.log_prob(action_tensor)
            ratio = (log_prob - old_log).exp()
            policy_loss = -torch.minimum(ratio * adv, ratio.clamp(0.8, 1.2) * adv).mean()
            value_loss = F.mse_loss(new_value, targets)
            entropy = distribution.entropy().mean()
            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
            if auxiliary is not None:
                auxiliary_loss = auxiliary.loss(model, args.auxiliary_batch)
                loss = loss + args.auxiliary_weight * auxiliary_loss
            if threat is not None:
                threat_loss = threat.loss(model, args.threat_batch)
                loss = loss + args.threat_weight * threat_loss
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
    parser.add_argument("--size", type=int, default=9)
    parser.add_argument("--opponent", choices=["random", "weak", "tactical"], default="random")
    parser.add_argument("--tactical-data", type=Path)
    parser.add_argument("--warmup-updates", type=int, default=0)
    parser.add_argument("--auxiliary-batch", type=int, default=128)
    parser.add_argument("--auxiliary-weight", type=float, default=1.0)
    parser.add_argument("--auxiliary-mode", choices=["cold", "history"], default="cold")
    parser.add_argument("--architecture", choices=["global", "spatial", "spatial_threat"], default="global")
    parser.add_argument("--curriculum", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--start-data",type=Path)
    parser.add_argument("--empty-probability",type=float,default=.5)
    parser.add_argument("--balanced-auxiliary",action="store_true")
    parser.add_argument("--dagger-games",type=int,default=0)
    parser.add_argument("--dagger-rounds",type=int,default=3)
    parser.add_argument("--dagger-fraction",type=float,default=.5)
    parser.add_argument("--dagger-holdouts",type=Path,nargs="*",default=[])
    parser.add_argument("--threat-data",type=Path)
    parser.add_argument("--threat-batch",type=int,default=64)
    parser.add_argument("--threat-weight",type=float,default=1.0)
    parser.add_argument("--initialize-from", type=Path)
    parser.add_argument("--defense-data-dir", type=Path)
    parser.add_argument("--attack-data-dir", type=Path)
    parser.add_argument("--preparation-data-dir", type=Path)
    parser.add_argument("--fixed-base-batch", type=int)
    parser.add_argument("--fixed-defense-batch", type=int)
    parser.add_argument("--fixed-attack-batch", type=int)
    parser.add_argument("--fixed-preparation-batch", type=int)
    args = parser.parse_args()
    if not 0<=args.empty_probability<=1 or (args.start_data and args.architecture not in ("spatial", "spatial_threat")):
        parser.error("Invalid midgame mixture or architecture")
    if args.dagger_games and (not args.tactical_data or not args.start_data or
                              args.dagger_rounds < 1 or args.updates % args.dagger_rounds):
        parser.error("DAgger requires tactical/start data and evenly divided positive rounds")
    if not 0 <= args.dagger_fraction <= 1:
        parser.error("Invalid DAgger mixture")
    if bool(args.defense_data_dir) != bool(args.attack_data_dir):
        parser.error("Defense and attack data directories must be provided together")
    if args.defense_data_dir and (not args.tactical_data or args.auxiliary_batch != 128 or
                                  args.architecture != "spatial"):
        parser.error("Fixed E mixture requires tactical data, batch 128, and spatial architecture")
    if args.initialize_from and not args.initialize_from.exists():
        parser.error("Initialization directory does not exist")
    if args.preparation_data_dir and not (args.defense_data_dir and args.attack_data_dir):
        parser.error("Preparation data requires the complete fixed E mixture")
    if args.threat_data and (args.architecture != "spatial_threat" or args.threat_batch < 1):
        parser.error("Threat data requires spatial_threat and a positive batch")
    if min(args.updates, args.envs, args.rollout, args.epochs) < 1:
        parser.error("All budgets must be positive")
    if args.size < 5 or args.warmup_updates < 0 or args.auxiliary_batch < 1 or args.auxiliary_weight < 0:
        parser.error("Invalid size or auxiliary budget")
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









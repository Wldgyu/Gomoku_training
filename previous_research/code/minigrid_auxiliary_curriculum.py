"""단서 기억·분기점 비교 보조 손실의 S9 길이 전이 효과를 대조."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from minigrid_length_curriculum import HOLDOUT_SIZE, TRAIN_SIZES, evaluate_sizes
from minigrid_memory_diagnostic import build_model
from minigrid_memory_pilot import (
    ROOT,
    create_env,
    masked_cue_observation,
    parameter_count,
    reset_episode,
    tensor_observation,
)
from minigrid_random_start_curriculum import teacher_episode


def parameter_hash(model) -> str:
    digest = hashlib.sha256()
    for name, value in model.named_parameters():
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def topology_hash(model) -> str | None:
    if model.kind != "fly":
        return None
    digest = hashlib.sha256()
    digest.update(model.pre.cpu().numpy().tobytes())
    digest.update(model.post.cpu().numpy().tobytes())
    return digest.hexdigest()


def make_aux_dataset(first_seed: int, count: int, size: int) -> list[dict]:
    rows = []
    for seed in range(first_seed, first_seed + count):
        observations, actions, _ = teacher_episode(seed, size)
        env = create_env(size, 100)
        try:
            reset_episode(env, seed, "default")
            base = env.unwrapped
            center = base.height // 2
            cue_type = type(base.grid.get(1, center - 1)).__name__
            upper_type = type(base.grid.get(size - 2, center - 2)).__name__
            cue_label = int(cue_type == "Key")
            upper_match = int(cue_type == upper_type)
            visible = []
            for index, observation in enumerate(observations):
                visible.append(not np.array_equal(observation["image"], masked_cue_observation(env)["image"]))
                if index < len(actions) - 1:
                    env.step(actions[index])
            if not any(visible) or upper_match != int(actions[-2] == 0):
                raise RuntimeError("단서 또는 마지막 비교 라벨이 교사 경로와 일치하지 않습니다")
        finally:
            env.close()
        images, directions = tensor_observation(observations, torch.device("cpu"))
        first_visible = visible.index(True)
        rows.append({"images": images, "directions": directions,
                     "actions": torch.tensor(actions), "cue_label": cue_label,
                     "cue_supervised_from": first_visible, "upper_match": upper_match})
    return rows


def train_batch(model, cue_head: nn.Module, comparison_head: nn.Module,
                rows: list[dict], device: torch.device,
                cue_weight: float, comparison_weight: float) -> tuple[torch.Tensor, dict]:
    length, batch = max(len(row["actions"]) for row in rows), len(rows)
    images = torch.zeros((length, batch, 7, 7, 3), dtype=torch.long, device=device)
    directions = torch.zeros((length, batch), dtype=torch.long, device=device)
    actions = torch.zeros((length, batch), dtype=torch.long, device=device)
    actor_weights = torch.zeros((length, batch), device=device)
    cue_weights = torch.zeros((length, batch), device=device)
    cue_targets = torch.tensor([row["cue_label"] for row in rows], device=device)
    comparison_targets = torch.tensor([row["upper_match"] for row in rows], device=device)
    branch_indices = []
    for index, row in enumerate(rows):
        count = len(row["actions"])
        images[:count, index] = row["images"].to(device)
        directions[:count, index] = row["directions"].to(device)
        actions[:count, index] = row["actions"].to(device)
        actor_weights[:count, index] = 1
        actor_weights[count - 2, index] = 8
        cue_weights[row["cue_supervised_from"]:count, index] = 1
        branch_indices.append(count - 2)
    matrix = model.recurrent_matrix()
    state = model.initial_state(batch, device)
    actor_logits, states = [], []
    for step in range(length):
        logits, _, state = model.step(images[step], directions[step], state, matrix)
        actor_logits.append(logits)
        states.append(state)
    actor_logits = torch.stack(actor_logits)
    states = torch.stack(states)
    action_loss = F.cross_entropy(actor_logits.reshape(-1, 3), actions.reshape(-1), reduction="none")
    action_loss = (action_loss.reshape(length, batch) * actor_weights).sum() / actor_weights.sum()
    loss = action_loss
    metrics = {"actor_loss": float(action_loss.detach())}
    if cue_weight:
        cue_logits = cue_head(states)
        cue_labels = cue_targets.unsqueeze(0).expand(length, -1)
        cue_loss = F.cross_entropy(cue_logits.reshape(-1, 2), cue_labels.reshape(-1), reduction="none")
        cue_loss = (cue_loss.reshape(length, batch) * cue_weights).sum() / cue_weights.sum()
        loss = loss + cue_weight * cue_loss
        metrics["cue_loss"] = float(cue_loss.detach())
    if comparison_weight:
        final_states = states[torch.tensor(branch_indices, device=device), torch.arange(batch, device=device)]
        comparison_loss = F.cross_entropy(comparison_head(final_states), comparison_targets)
        loss = loss + comparison_weight * comparison_loss
        metrics["comparison_loss"] = float(comparison_loss.detach())
    return loss, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("fly", "rewired", "rnn", "gru"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--source-checkpoint", type=Path)
    parser.add_argument("--train-episodes-per-size", type=int, default=512)
    parser.add_argument("--eval-episodes", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    parser.add_argument("--cue-weight", type=float, default=0.5)
    parser.add_argument("--comparison-weight", type=float, default=0.5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "minigrid_auxiliary_curriculum")
    args = parser.parse_args()
    if min(args.train_episodes_per_size, args.eval_episodes, args.epochs, args.batch_size) < 1:
        parser.error("에피소드·epoch·batch 인자는 양수여야 합니다")
    device = torch.device(args.device)
    model = build_model(args.model, 1000, args.seed, device)
    if args.source_checkpoint:
        checkpoint = torch.load(args.source_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
    initial_parameter_hash = parameter_hash(model)
    graph_hash = topology_hash(model)
    # Separate generator keeps minibatch order identical in the control and auxiliary conditions.
    order_generator = torch.Generator().manual_seed(args.seed + 71_000)
    torch.manual_seed(args.seed + 72_000)
    cue_head = nn.Linear(model.hidden_size, 2).to(device)
    comparison_head = nn.Linear(model.hidden_size, 2).to(device)
    train = []
    for size in TRAIN_SIZES:
        train.extend(make_aux_dataset(17_000_000 + size * 100_000,
                                      args.train_episodes_per_size, size))
    start = time.perf_counter()
    initial = evaluate_sizes(model, episodes=args.eval_episodes, seed=18_000_000,
                             device=device, sizes=TRAIN_SIZES + (HOLDOUT_SIZE,), ablations=False)
    parameters = list(model.parameters()) + list(cue_head.parameters()) + list(comparison_head.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate)
    best_state, best_row, history = None, None, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(len(train), generator=order_generator).tolist()
        metrics = []
        for offset in range(0, len(train), args.batch_size):
            batch = [train[index] for index in order[offset:offset + args.batch_size]]
            loss, batch_metrics = train_batch(model, cue_head, comparison_head, batch,
                                              device, args.cue_weight, args.comparison_weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 0.5)
            optimizer.step()
            metrics.append(batch_metrics)
        if epoch % 5 == 0 or epoch == args.epochs:
            validated = evaluate_sizes(model, episodes=args.eval_episodes, seed=18_000_000,
                                       device=device, sizes=TRAIN_SIZES, ablations=False)
            scores = [validated[str(size)]["normal"]["success_rate"] for size in TRAIN_SIZES]
            row = {"epoch": epoch, "mean_train_success": float(np.mean(scores)),
                   "min_train_success": float(min(scores)),
                   "losses": {key: float(np.mean([item[key] for item in metrics]))
                              for key in metrics[0]}, "by_size": validated}
            history.append(row)
            print(json.dumps({"epoch": epoch, "mean_train_success": row["mean_train_success"],
                              "min_train_success": row["min_train_success"],
                              "losses": row["losses"]}), flush=True)
            if best_row is None or (row["mean_train_success"], row["min_train_success"]) > (
                best_row["mean_train_success"], best_row["min_train_success"]
            ):
                best_row = row
                best_state = copy.deepcopy(model.state_dict())
    assert best_row is not None and best_state is not None
    model.load_state_dict(best_state)
    final = evaluate_sizes(model, episodes=args.eval_episodes * 2, seed=19_000_000,
                           device=device, sizes=TRAIN_SIZES + (HOLDOUT_SIZE,), ablations=True)
    result = {"model": args.model, "seed": args.seed, "parameters": parameter_count(model),
              "initial_parameter_sha256": initial_parameter_hash, "topology_sha256": graph_hash,
              "source_checkpoint": str(args.source_checkpoint) if args.source_checkpoint else None,
              "train_sizes": TRAIN_SIZES, "holdout_size": HOLDOUT_SIZE,
              "train_episodes_per_size": args.train_episodes_per_size,
              "eval_episodes_final_per_size": args.eval_episodes * 2,
              "epochs": args.epochs, "learning_rate": args.learning_rate,
              "cue_weight": args.cue_weight, "comparison_weight": args.comparison_weight,
              "initial": initial, "history": history, "selected_epoch": best_row["epoch"],
              "selection_rule": "mean and minimum train-size success; S9 excluded",
              "final_fresh": final, "duration_seconds": time.perf_counter() - start}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    condition = "aux" if args.cue_weight or args.comparison_weight else "control"
    source_name = "finetune" if args.source_checkpoint else "scratch"
    stem = f"{source_name}_{condition}_{args.model}_1000_seed{args.seed}"
    (args.output_dir / f"{stem}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    torch.save({"model_state": model.state_dict(), "result": result}, args.output_dir / f"{stem}.pt")
    print(json.dumps({"selected_epoch": best_row["epoch"], "s9": final[str(HOLDOUT_SIZE)]}), flush=True)


if __name__ == "__main__":
    main()

"""관측에서 단서·분기·목표를 읽고 일치 규칙으로 마지막 선택을 하는 정책."""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
from minigrid.core.world_object import Ball
from torch import nn
from torch.nn import functional as F

from minigrid_memory_diagnostic import build_model
from minigrid_memory_pilot import (
    ROOT,
    ObservationEncoder,
    masked_cue_observation,
    tensor_observation,
)
from minigrid_relation_data import VARIANTS, make_env, reset_variant, teacher_episode


class VisualRelations(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = ObservationEncoder()
        self.cue = nn.Linear(64, 3)  # 안 보임 / Ball / Key
        self.branch = nn.Linear(64, 2)
        self.upper = nn.Linear(64, 2)  # Ball / Key

    def forward(self, images: torch.Tensor, directions: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.encoder(images, directions)
        return self.cue(features), self.branch(features), self.upper(features)


def labeled_frames(seeds: range, size: int, *, variants=VARIANTS,
                   random_length: bool = False, excluded_junction_x=()) -> dict:
    observations, cue_labels, branch_labels, upper_labels = [], [], [], []
    trajectories = 0
    for seed in seeds:
        for cue_flip, target_flip in variants:
            sequence, actions, meta = teacher_episode(
                seed, size, cue_flip=cue_flip, target_flip=target_flip,
                random_length=random_length
            )
            if meta["junction_x"] in excluded_junction_x:
                break
            env = make_env(size, random_length=random_length)
            try:
                reset_variant(env, seed, cue_flip=cue_flip, target_flip=target_flip)
                base = env.unwrapped
                upper = base.grid.get(meta["junction_x"], base.height // 2 - 2)
                upper_label = int(not isinstance(upper, Ball))
                cue_label = 1 if meta["cue_type"] == "Ball" else 2
                for index, observation in enumerate(sequence):
                    visible = not np.array_equal(
                        observation["image"], masked_cue_observation(env)["image"]
                    )
                    observations.append(observation)
                    cue_labels.append(cue_label if visible else 0)
                    branch_labels.append(int(index == len(actions) - 2))
                    upper_labels.append(upper_label)
                    if index < len(actions) - 1:
                        env.step(actions[index])
            finally:
                env.close()
            trajectories += 1
    images, directions = tensor_observation(observations, torch.device("cpu"))
    return {"images": images, "directions": directions,
            "cue": torch.tensor(cue_labels), "branch": torch.tensor(branch_labels),
            "upper": torch.tensor(upper_labels), "trajectories": trajectories}


@torch.no_grad()
def classify(detector: VisualRelations, dataset: dict, device: torch.device,
             batch_size: int = 1024) -> dict:
    detector.eval()
    correct_cue = correct_branch = correct_upper = positive_cue = predicted_cue = 0
    positive_branch = predicted_branch = 0
    count = len(dataset["cue"])
    for start in range(0, count, batch_size):
        stop = start + batch_size
        cue, branch, upper = detector(dataset["images"][start:stop].to(device),
                                      dataset["directions"][start:stop].to(device))
        cue_prediction = cue.argmax(dim=-1).cpu()
        branch_prediction = branch.argmax(dim=-1).cpu()
        upper_prediction = upper.argmax(dim=-1).cpu()
        cue_true = dataset["cue"][start:stop]
        branch_true = dataset["branch"][start:stop]
        upper_true = dataset["upper"][start:stop]
        correct_cue += int((cue_prediction == cue_true).sum())
        correct_branch += int((branch_prediction == branch_true).sum())
        correct_upper += int(((upper_prediction == upper_true) & (branch_true == 1)).sum())
        positive_cue += int((cue_true > 0).sum())
        predicted_cue += int((cue_prediction > 0).sum())
        positive_branch += int((branch_true > 0).sum())
        predicted_branch += int((branch_prediction > 0).sum())
    return {"frames": count, "cue_accuracy": correct_cue / count,
            "branch_accuracy": correct_branch / count,
            "upper_accuracy_at_branch": correct_upper / max(1, positive_branch),
            "cue_positive_count": positive_cue, "cue_predicted_count": predicted_cue,
            "branch_positive_count": positive_branch, "branch_predicted_count": predicted_branch}


@torch.no_grad()
def evaluate_policy(actor, detector: VisualRelations, *, size: int, seeds: range,
                    device: torch.device, cue_threshold: float = 0.5,
                    branch_threshold: float = 0.5) -> dict:
    actor.eval()
    detector.eval()
    success = cue_latched = branch_override = branch_reached = all_four = 0
    variant_success = {f"{int(cue)}{int(target)}": 0 for cue, target in VARIANTS}
    for seed in seeds:
        four = []
        for cue_flip, target_flip in VARIANTS:
            env = make_env(size)
            try:
                observation, meta = reset_variant(env, seed, cue_flip=cue_flip, target_flip=target_flip)
                base = env.unwrapped
                state = actor.initial_state(1, device)
                matrix = actor.recurrent_matrix()
                remembered, did_override, did_reach = None, False, False
                while True:
                    image, direction = tensor_observation([observation], device)
                    action_logits, _, state = actor.step(image, direction, state, matrix)
                    cue_logits, branch_logits, upper_logits = detector(image, direction)
                    cue_probability = cue_logits.softmax(dim=-1)[0]
                    branch_probability = branch_logits.softmax(dim=-1)[0, 1]
                    if remembered is None:
                        candidate = int(cue_probability.argmax().item())
                        if candidate and float(cue_probability[candidate]) >= cue_threshold:
                            remembered = candidate - 1  # Ball=0, Key=1
                    action = int(action_logits.argmax(dim=-1).item())
                    if float(branch_probability) >= branch_threshold and remembered is not None:
                        upper_type = int(upper_logits.argmax(dim=-1).item())
                        action = int(upper_type != remembered)  # 일치하면 북쪽(좌회전)
                        did_override = True
                    if tuple(base.agent_pos) == (meta["junction_x"], base.height // 2):
                        did_reach = True
                    observation, reward, terminated, truncated, _ = env.step(action)
                    if terminated or truncated:
                        ok = reward > 0
                        success += ok
                        variant_success[f"{int(cue_flip)}{int(target_flip)}"] += ok
                        four.append(ok)
                        cue_latched += remembered is not None
                        branch_override += did_override
                        branch_reached += did_reach
                        break
            finally:
                env.close()
        all_four += all(four)
    total = len(seeds) * 4
    return {"base_episodes": len(seeds), "success_rate": success / total,
            "all_four_success": all_four / len(seeds),
            "per_variant_success": {key: value / len(seeds) for key, value in variant_success.items()},
            "cue_latched_rate": cue_latched / total,
            "branch_override_rate": branch_override / total,
            "branch_reach_rate": branch_reached / total}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor-model", choices=("fly", "rewired", "rnn", "gru"), default="rnn")
    parser.add_argument("--actor-seed", type=int, default=401)
    parser.add_argument("--actor-checkpoint", type=Path, required=True)
    parser.add_argument("--detector-seed", type=int, default=501)
    parser.add_argument("--train-base-episodes-per-size", type=int, default=256)
    parser.add_argument("--validation-base-episodes", type=int, default=64)
    parser.add_argument("--final-base-episodes", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "minigrid_compositional")
    args = parser.parse_args()
    device = torch.device(args.device)
    actor = build_model(args.actor_model, 1000, args.actor_seed, device)
    actor.load_state_dict(torch.load(args.actor_checkpoint, map_location=device, weights_only=False)["model_state"])
    torch.manual_seed(args.detector_seed)
    detector = VisualRelations().to(device)
    start = time.perf_counter()
    train_parts = [labeled_frames(range(25_000_000 + size * 100_000,
                                       25_000_000 + size * 100_000 + args.train_base_episodes_per_size), size)
                   for size in (7, 11, 13)]
    train = {key: torch.cat([part[key] for part in train_parts])
             for key in ("images", "directions", "cue", "branch", "upper")}
    validation = labeled_frames(range(26_000_000, 26_000_000 + args.validation_base_episodes), 11)
    optimizer = torch.optim.AdamW(detector.parameters(), lr=args.learning_rate)
    generator = torch.Generator().manual_seed(args.detector_seed + 1000)
    best_state, best_score, selected_epoch = None, -1.0, 0
    history = []
    for epoch in range(1, args.epochs + 1):
        detector.train()
        order = torch.randperm(len(train["cue"]), generator=generator)
        losses = []
        for indices in order.split(args.batch_size):
            cue_true = train["cue"][indices].to(device)
            branch_true = train["branch"][indices].to(device)
            upper_true = train["upper"][indices].to(device)
            cue, branch, upper = detector(train["images"][indices].to(device),
                                          train["directions"][indices].to(device))
            cue_loss = F.cross_entropy(cue, cue_true, weight=torch.tensor([1.0, 4.0, 4.0], device=device))
            branch_loss = F.cross_entropy(branch, branch_true,
                                          weight=torch.tensor([1.0, 8.0], device=device))
            branch_frames = branch_true == 1
            upper_loss = (F.cross_entropy(upper[branch_frames], upper_true[branch_frames])
                          if branch_frames.any() else upper.sum() * 0)
            loss = cue_loss + branch_loss + upper_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        metrics = classify(detector, validation, device)
        score = (metrics["cue_accuracy"] + metrics["branch_accuracy"]
                 + metrics["upper_accuracy_at_branch"]) / 3
        row = {"epoch": epoch, "loss": float(np.mean(losses)), "validation": metrics, "score": score}
        history.append(row)
        print(json.dumps({"epoch": epoch, "loss": row["loss"], "validation": metrics}), flush=True)
        if score > best_score:
            best_state, best_score, selected_epoch = copy.deepcopy(detector.state_dict()), score, epoch
    assert best_state is not None
    detector.load_state_dict(best_state)
    teacher_diagnostic = {str(size): classify(
        detector, labeled_frames(range(27_000_000, 27_000_000 + args.final_base_episodes), size), device
    ) for size in (9, 15, 19)}
    autonomous = {str(size): evaluate_policy(actor, detector, size=size,
                                             seeds=range(28_000_000, 28_000_000 + args.final_base_episodes),
                                             device=device)
                  for size in (9, 15, 19)}
    result = {"method": "visual cue latch and explicit cue-target equality at detected branch",
              "actor_model": args.actor_model, "actor_seed": args.actor_seed,
              "actor_checkpoint": str(args.actor_checkpoint), "detector_seed": args.detector_seed,
              "train_base_episodes_per_size": args.train_base_episodes_per_size,
              "train_frames": len(train["cue"]), "epochs": args.epochs,
              "validation_base_episodes": args.validation_base_episodes,
              "final_base_episodes": args.final_base_episodes,
              "selected_epoch": selected_epoch, "history": history,
              "teacher_frame_diagnostic": teacher_diagnostic,
              "autonomous": autonomous, "duration_seconds": time.perf_counter() - start}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"compositional_{args.actor_model}_seed{args.actor_seed}_detector{args.detector_seed}"
    (args.output_dir / f"{stem}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    torch.save({"detector_state": detector.state_dict(), "result": result}, args.output_dir / f"{stem}.pt")
    print(json.dumps({"selected_epoch": selected_epoch, "autonomous": autonomous}), flush=True)


if __name__ == "__main__":
    main()

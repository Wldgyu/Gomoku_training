"""단서·목표 짝 과제와 가변 복도 학습의 미사용 길이 전이 실험."""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

from minigrid_memory_diagnostic import build_model
from minigrid_memory_pilot import (
    ROOT,
    masked_cue_observation,
    parameter_count,
    tensor_observation,
)
from minigrid_random_start_curriculum import train_batch
from minigrid_relation_data import (
    EXCLUDED_JUNCTION_X,
    VARIANTS,
    build_dataset,
    make_env,
    reset_variant,
)

TRAIN_SIZES = (7, 11, 13)
HOLDOUT_SIZES = (9, 15)


def build_train(mode: str, *, episodes: int) -> list[dict]:
    if episodes % 12:
        raise ValueError("episodes는 12의 배수여야 합니다")
    factorial = mode.endswith("factorial")
    variable = mode.startswith("variable")
    variants = VARIANTS if factorial else ((False, False),)
    if variable:
        needed = episodes // len(variants)
        rows, seed = [], 21_000_000
        while len(rows) < episodes:
            chunk_size = min(256, max(16, needed - len(rows) // len(variants) + 8))
            chunk = build_dataset(range(seed, seed + chunk_size), 17, variants=variants,
                                  random_length=True, excluded_junction_x=EXCLUDED_JUNCTION_X)
            rows.extend(chunk)
            seed += chunk_size
        rows = rows[:episodes]
    else:
        count = episodes // (len(TRAIN_SIZES) * len(variants))
        rows = []
        for size in TRAIN_SIZES:
            first = 21_000_000 + size * 100_000
            rows.extend(build_dataset(range(first, first + count), size, variants=variants))
    if len(rows) != episodes or any(row["meta"]["junction_x"] in EXCLUDED_JUNCTION_X for row in rows):
        raise RuntimeError("훈련 에피소드 수 또는 제외 길이가 잘못됐습니다")
    return rows


@torch.no_grad()
def evaluate_variants(model, *, size: int, seeds: range, device: torch.device,
                      random_length: bool = False, excluded_junction_x=()) -> dict:
    model.eval()
    variant_success = {f"{int(cue)}{int(target)}": 0 for cue, target in VARIANTS}
    all_four, cue_seen, reached, used = 0, 0, 0, 0
    for seed in seeds:
        env_probe = make_env(size, random_length=random_length)
        try:
            _, meta = reset_variant(env_probe, seed)
        finally:
            env_probe.close()
        if meta["junction_x"] in excluded_junction_x:
            continue
        used += 1
        successes = []
        for cue_flip, target_flip in VARIANTS:
            env = make_env(size, random_length=random_length)
            try:
                observation, _ = reset_variant(env, seed, cue_flip=cue_flip, target_flip=target_flip)
                base = env.unwrapped
                state = model.initial_state(1, device)
                matrix = model.recurrent_matrix()
                seen, did_reach = False, False
                while True:
                    seen |= not np.array_equal(observation["image"], masked_cue_observation(env)["image"])
                    if tuple(base.agent_pos) == (meta["junction_x"], base.height // 2):
                        did_reach = True
                    image, direction = tensor_observation([observation], device)
                    logits, _, state = model.step(image, direction, state, matrix)
                    observation, reward, terminated, truncated, _ = env.step(int(logits.argmax(dim=-1).item()))
                    if terminated or truncated:
                        success = reward > 0
                        variant_success[f"{int(cue_flip)}{int(target_flip)}"] += success
                        successes.append(success)
                        cue_seen += seen
                        reached += did_reach
                        break
            finally:
                env.close()
        all_four += all(successes)
    if not used:
        raise RuntimeError("평가할 복도 길이가 없습니다")
    return {"base_episodes": used, "per_variant_success": {key: value / used
            for key, value in variant_success.items()}, "all_four_success": all_four / used,
            "mean_success": sum(variant_success.values()) / (4 * used),
            "cue_seen_rate": cue_seen / (4 * used), "branch_reach_rate": reached / (4 * used)}


def evaluate_lengths(model, *, sizes: tuple[int, ...], seed: int,
                     episodes: int, device: torch.device) -> dict:
    return {str(size): evaluate_variants(model, size=size,
                                        seeds=range(seed, seed + episodes), device=device)
            for size in sizes}


def evaluate_validation(model, *, seed: int, episodes: int, device: torch.device) -> dict:
    rows = evaluate_lengths(model, sizes=TRAIN_SIZES, seed=seed,
                            episodes=episodes, device=device)
    rows["17Random_heldin"] = evaluate_variants(
        model, size=17, seeds=range(seed, seed + episodes * 2), device=device,
        random_length=True, excluded_junction_x=EXCLUDED_JUNCTION_X
    )
    return rows


def validation_score(rows: dict) -> tuple[float, float, float]:
    scores = [row["all_four_success"] for row in rows.values()]
    normal = [row["per_variant_success"]["00"] for row in rows.values()]
    return float(np.mean(scores)), float(min(scores)), float(np.mean(normal))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("fly", "rewired", "rnn", "gru"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--mode", choices=("fixed_control", "fixed_factorial",
                                            "variable_control", "variable_factorial"), required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--train-episodes", type=int, default=3072)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    parser.add_argument("--validation-base-episodes", type=int, default=24)
    parser.add_argument("--final-base-episodes", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "minigrid_relation_curriculum")
    args = parser.parse_args()
    if min(args.train_episodes, args.epochs, args.batch_size,
           args.validation_base_episodes, args.final_base_episodes) < 1:
        parser.error("에피소드·epoch·batch 인자는 양수여야 합니다")
    device = torch.device(args.device)
    model = build_model(args.model, 1000, args.seed, device)
    checkpoint = torch.load(args.source_checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    start = time.perf_counter()
    train = build_train(args.mode, episodes=args.train_episodes)
    train_junction_counts = {str(x): sum(row["meta"]["junction_x"] == x for row in train)
                             for x in sorted({row["meta"]["junction_x"] for row in train})}
    order_generator = torch.Generator().manual_seed(args.seed + 81_000)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    initial_validation = evaluate_validation(model, seed=22_000_000,
                                             episodes=args.validation_base_episodes, device=device)
    history = [{"epoch": 0, "by_size": initial_validation,
                "mean_all_four": validation_score(initial_validation)[0]}]
    best_state = copy.deepcopy(model.state_dict())
    best_score, selected_epoch = validation_score(initial_validation), 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(len(train), generator=order_generator).tolist()
        losses = []
        for offset in range(0, len(train), args.batch_size):
            batch = [train[index] for index in order[offset:offset + args.batch_size]]
            loss = train_batch(model, batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            losses.append(float(loss.detach()))
        if epoch % 5 == 0 or epoch == args.epochs:
            validated = evaluate_validation(model, seed=22_000_000,
                                            episodes=args.validation_base_episodes, device=device)
            score = validation_score(validated)
            row = {"epoch": epoch, "teacher_loss": float(np.mean(losses)),
                   "mean_all_four": score[0], "min_all_four": score[1],
                   "mean_normal": score[2], "by_size": validated}
            history.append(row)
            print(json.dumps({key: row[key] for key in ("epoch", "teacher_loss",
                                                      "mean_all_four", "mean_normal")}), flush=True)
            if score > best_score:
                best_score, best_state, selected_epoch = score, copy.deepcopy(model.state_dict()), epoch
    model.load_state_dict(best_state)
    final = evaluate_lengths(model, sizes=TRAIN_SIZES + HOLDOUT_SIZES, seed=23_000_000,
                             episodes=args.final_base_episodes, device=device)
    result = {"model": args.model, "seed": args.seed, "mode": args.mode,
              "method": "teacher imitation with factorial cue/target and/or variable-length training",
              "source_checkpoint": str(args.source_checkpoint), "parameters": parameter_count(model),
              "train_episodes": len(train), "train_junction_x_counts": train_junction_counts,
              "excluded_train_junction_x": EXCLUDED_JUNCTION_X,
              "epochs": args.epochs, "selected_epoch": selected_epoch,
              "selection_rule": "S7/S11/S13 and S17Random held-in all-four success; S9/S15 corridor lengths excluded",
              "learning_rate": args.learning_rate,
              "validation_base_episodes": args.validation_base_episodes,
              "final_base_episodes": args.final_base_episodes,
              "history": history, "final_fresh": final,
              "duration_seconds": time.perf_counter() - start}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"relation_{args.mode}_{args.model}_1000_seed{args.seed}"
    (args.output_dir / f"{stem}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    torch.save({"model_state": model.state_dict(), "result": result}, args.output_dir / f"{stem}.pt")
    print(json.dumps({"selected_epoch": selected_epoch,
                      "s9": final["9"], "s15": final["15"]}), flush=True)


if __name__ == "__main__":
    main()

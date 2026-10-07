"""직교 초기화와 기존 초기화를 같은 새 카드 게임에서 비교한다."""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path

import torch

from pilot_card_game import BaselinePolicy, evaluate

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "game_pilot"
OLD_SEEDS = (201, 202, 203, 204, 205)
NEW_SEEDS = (206, 207, 208, 209, 210)


def path_for(size: int, seed: int, init: str) -> Path:
    name = f"card_stage1_mlphead_reinforce_rnn_{size}_delay4_seed{seed}.json"
    if init == "orthogonal":
        directory = OUT / "rnn_recovery" / "orthogonal_controlled"
    elif seed in OLD_SEEDS:
        directory = OUT / "heldout_tuned"
    else:
        directory = OUT / "rnn_recovery" / "default_fresh"
    return directory / name


def evaluate_one(size: int, seed: int, init: str, games: int, device: torch.device) -> dict:
    path = path_for(size, seed, init)
    result = json.loads(path.read_text(encoding="utf-8"))
    if result["model"] != "rnn" or result["blank_steps"] != 4:
        raise ValueError(f"4스텝 RNN 결과가 아닙니다: {path}")
    model = BaselinePolicy("rnn", result["hidden_size"]).to(device)
    checkpoint = torch.load(path.with_suffix(".pt"), map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    success, no_memory = evaluate(
        model, count=games, blank_steps=4, device=device,
        seed=70_000 + seed, stage=1,
    )
    row = {
        "neurons_reference": size,
        "seed": seed,
        "group": "diagnostic_old" if seed in OLD_SEEDS else "independent_new",
        "initialization": init,
        "games": games,
        "success_rate": success,
        "no_memory_success_rate": no_memory,
        "source": str(path.relative_to(ROOT)).replace("\\", "/"),
    }
    print(f"EVAL n={size} seed={seed} init={init} success={success:.3f}", flush=True)
    return row


def paired_summary(rows: list[dict], size: int, seeds: tuple[int, ...]) -> dict:
    lookup = {(row["seed"], row["initialization"]): row["success_rate"] for row in rows
              if row["neurons_reference"] == size and row["seed"] in seeds}
    if len(lookup) != 2 * len(seeds):
        raise ValueError(f"평가 행이 부족합니다: size={size}, seeds={seeds}")
    default = [lookup[(seed, "default")] for seed in seeds]
    orthogonal = [lookup[(seed, "orthogonal")] for seed in seeds]
    differences = [b - a for a, b in zip(default, orthogonal, strict=True)]
    rng = random.Random(20261001 + size + min(seeds))
    draws = [statistics.mean(rng.choices(differences, k=len(seeds))) for _ in range(10000)]
    draws.sort()
    return {
        "seeds": seeds,
        "default_mean": statistics.mean(default),
        "orthogonal_mean": statistics.mean(orthogonal),
        "paired_difference_mean": statistics.mean(differences),
        "paired_difference_95pct_bootstrap": [draws[250], draws[9749]],
        "default_success_over_75pct": sum(value >= 0.75 for value in default),
        "orthogonal_success_over_75pct": sum(value >= 0.75 for value in orthogonal),
        "default_by_seed": dict(zip(map(str, seeds), default, strict=True)),
        "orthogonal_by_seed": dict(zip(map(str, seeds), orthogonal, strict=True)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-games", type=int, default=4096)
    args = parser.parse_args()
    if args.eval_games < 1:
        parser.error("eval-games must be positive")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = [
        evaluate_one(size, seed, init, args.eval_games, device)
        for size in (1000, 2000)
        for seed in OLD_SEEDS + NEW_SEEDS
        for init in ("default", "orthogonal")
    ]
    summary = {
        "protocol": "same new 4096 games per seed for default and controlled orthogonal init",
        "rows": rows,
        "by_size_and_group": {
            f"{size}_{group}": paired_summary(rows, size, seeds)
            for size in (1000, 2000)
            for group, seeds in (("old", OLD_SEEDS), ("new", NEW_SEEDS))
        },
    }
    output = OUT / "rnn_recovery_test_4096.json"
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {output}", flush=True)
    for key, value in summary["by_size_and_group"].items():
        print(key, json.dumps(value, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

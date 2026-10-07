"""독립 평가 결과를 그래프·학습 시드별로 집계하고 불확실성을 계산한다."""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SEEDS = (201, 202, 203, 204, 205)
GRAPH_SEEDS = (0, 1, 2, 3, 4)


def mean(values: list[float]) -> float:
    return statistics.mean(values)


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


def bootstrap_paired(values: list[float], rng: random.Random, repeats: int) -> tuple[float, float]:
    draws = [mean([rng.choice(values) for _ in values]) for _ in range(repeats)]
    return percentile(draws, 0.025), percentile(draws, 0.975)


def topology_summary(rows: list[dict], size: int, rng: random.Random, repeats: int) -> dict:
    actual = {
        row["seed"]: row["success_rate"] for row in rows
        if row["neurons_reference"] == size and row["model"] == "fly"
        and row["graph_variant"] == "real"
    }
    rewired = {
        (row["rewire_seed"], row["seed"]): row["success_rate"] for row in rows
        if row["neurons_reference"] == size and row["model"] == "fly"
        and row["graph_variant"] == "rewired"
    }
    if len(actual) != 5 or len(rewired) != 25:
        raise ValueError(f"크기 {size}의 실제/재배선 결과가 부족합니다")
    actual_mean = mean([actual[seed] for seed in SEEDS])
    rewired_mean = mean([rewired[(graph, seed)] for graph in GRAPH_SEEDS for seed in SEEDS])
    by_graph = {
        str(graph): mean([rewired[(graph, seed)] for seed in SEEDS])
        for graph in GRAPH_SEEDS
    }
    by_seed = {
        str(seed): actual[seed] - mean([rewired[(graph, seed)] for graph in GRAPH_SEEDS])
        for seed in SEEDS
    }
    # 학습 시드와 재배선 그래프를 각각 복원추출해 두 변동원을 반영한다.
    draws = []
    for _ in range(repeats):
        sampled_seeds = [rng.choice(SEEDS) for _ in SEEDS]
        sampled_graphs = [rng.choice(GRAPH_SEEDS) for _ in GRAPH_SEEDS]
        delta = mean([actual[seed] for seed in sampled_seeds])
        delta -= mean([rewired[(graph, seed)] for graph in sampled_graphs for seed in sampled_seeds])
        draws.append(delta)
    return {
        "actual_mean": actual_mean,
        "rewired_mean": rewired_mean,
        "difference_actual_minus_rewired": actual_mean - rewired_mean,
        "difference_95pct_two_way_bootstrap": [percentile(draws, 0.025), percentile(draws, 0.975)],
        "actual_by_training_seed": {str(seed): actual[seed] for seed in SEEDS},
        "rewired_mean_by_graph_seed": by_graph,
        "difference_by_training_seed": by_seed,
        "note": "one actual graph, five rewired graphs, five paired training seeds; exploratory bootstrap interval",
    }


def baseline_summary(rows: list[dict], size: int, kind: str,
                     rng: random.Random, repeats: int) -> dict:
    variants = {}
    for variant in ("default", "tuned"):
        candidates = {
            row["seed"]: row for row in rows
            if row["neurons_reference"] == size and row["model"] == kind
            and row.get("baseline_variant") == variant
        }
        if len(candidates) != 5:
            raise ValueError(f"{size} {kind} {variant} 결과 부족")
        variants[variant] = candidates
    default = [variants["default"][seed]["success_rate"] for seed in SEEDS]
    tuned = [variants["tuned"][seed]["success_rate"] for seed in SEEDS]
    diffs = [b - a for a, b in zip(default, tuned, strict=True)]
    return {
        "default_learning_rate": 0.001,
        "selected_learning_rate": variants["tuned"][SEEDS[0]]["learning_rate"],
        "default_mean": mean(default),
        "tuned_mean": mean(tuned),
        "paired_difference_mean": mean(diffs),
        "paired_difference_95pct_bootstrap": bootstrap_paired(diffs, rng, repeats),
        "default_by_seed": dict(zip(map(str, SEEDS), default, strict=True)),
        "tuned_by_seed": dict(zip(map(str, SEEDS), tuned, strict=True)),
        "default_success_over_75pct": sum(value >= 0.75 for value in default),
        "tuned_success_over_75pct": sum(value >= 0.75 for value in tuned),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/game_pilot/followup_test_4096.json")
    parser.add_argument("--output", type=Path, default=ROOT / "data/game_pilot/followup_analysis.json")
    parser.add_argument("--bootstrap-repeats", type=int, default=10000)
    args = parser.parse_args()
    rows = json.loads(args.input.read_text(encoding="utf-8"))["rows"]
    if len(rows) != 100:
        raise ValueError(f"평가 행 수가 100이 아닙니다: {len(rows)}")
    rng = random.Random(20261001)
    summary = {
        "source": str(args.input),
        "test_seeds": SEEDS,
        "rewire_seeds": GRAPH_SEEDS,
        "bootstrap_repeats": args.bootstrap_repeats,
        "topology": {},
        "baselines": {},
    }
    for size in (1000, 2000):
        summary["topology"][str(size)] = topology_summary(rows, size, rng, args.bootstrap_repeats)
        for kind in ("rnn", "gru"):
            summary["baselines"][f"{size}_{kind}"] = baseline_summary(
                rows, size, kind, rng, args.bootstrap_repeats
            )
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

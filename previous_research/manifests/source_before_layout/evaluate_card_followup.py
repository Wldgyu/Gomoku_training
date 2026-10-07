"""후속 4스텝 실험의 저장 모델을 같은 새 카드 문제로 재평가한다."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from pilot_card_game import BaselinePolicy, FlyPolicy, evaluate

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "game_pilot"
SIZES = (1000, 2000)
SEEDS = (201, 202, 203, 204, 205)
GRAPH_SEEDS = (0, 1, 2, 3, 4)


def result_file(kind: str, size: int, seed: int, *, graph_seed: int | None = None,
                baseline_variant: str | None = None) -> Path:
    variant = "" if graph_seed is None else f"_rewired{graph_seed}"
    name = f"card_stage1_mlphead_reinforce_{kind}{variant}_{size}_delay4_seed{seed}.json"
    directory = OUT if baseline_variant is None else OUT / f"heldout_{baseline_variant}"
    return directory / name


def evaluate_one(path: Path, device: torch.device, games: int) -> dict:
    result = json.loads(path.read_text(encoding="utf-8"))
    if result["stage"] != 1 or result["blank_steps"] != 4:
        raise ValueError(f"4스텝 1단계 결과가 아닙니다: {path}")
    kind = result["model"]
    if kind == "fly":
        suffix = (
            "" if result["graph_variant"] == "real"
            else f"_rewired_seed{result['rewire_seed']}"
        )
        graph = ROOT / "data" / "subgraphs" / f"malecns_cx_{result['neurons_reference']}{suffix}.npz"
        model = FlyPolicy(graph)
    else:
        model = BaselinePolicy(kind, result["hidden_size"])
    checkpoint = torch.load(path.with_suffix(".pt"), map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    # 동일 학습 시드에 대해 모든 그래프와 모델에 같은 새 문제를 제공한다.
    success, no_memory = evaluate(
        model, count=games, blank_steps=4, device=device,
        seed=60_000 + result["seed"], stage=1,
    )
    row = {
        "model": kind,
        "neurons_reference": result["neurons_reference"],
        "seed": result["seed"],
        "graph_variant": result["graph_variant"],
        "rewire_seed": result["rewire_seed"],
        "learning_rate": result.get("learning_rate", 0.001),
        "success_rate": success,
        "no_memory_success_rate": no_memory,
        "training_choices": result["choice_interactions"],
        "evaluation_games": games,
        "source_result": str(path.relative_to(ROOT)).replace("\\", "/"),
    }
    print(
        f"EVAL n={row['neurons_reference']} model={kind} "
        f"graph={row['graph_variant']}{row['rewire_seed']} "
        f"seed={row['seed']} lr={row['learning_rate']} success={success:.3f}",
        flush=True,
    )
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-games", type=int, default=4096)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=OUT / "followup_test_4096.json")
    args = parser.parse_args()
    if args.eval_games < 1:
        parser.error("eval-games must be positive")
    device = torch.device(args.device)
    rows = []
    for size in SIZES:
        for seed in SEEDS:
            rows.append(evaluate_one(result_file("fly", size, seed), device, args.eval_games))
            for graph_seed in GRAPH_SEEDS:
                rows.append(evaluate_one(result_file("fly", size, seed, graph_seed=graph_seed), device, args.eval_games))
            for kind in ("rnn", "gru"):
                for variant in ("default", "tuned"):
                    path = result_file(kind, size, seed, baseline_variant=variant)
                    # 조정에서 기본 학습률이 선택되면 체크포인트가 정확히
                    # 같으므로 중복 학습하지 않고 기본 결과를 함께 사용한다.
                    if variant == "tuned" and not path.exists():
                        path = result_file(kind, size, seed, baseline_variant="default")
                    rows.append(evaluate_one(
                        path,
                        device, args.eval_games,
                    ) | {"baseline_variant": variant})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "protocol": "new common card games, 4096 per training seed; training seeds 201-205",
        "rows": rows,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {args.output} ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()

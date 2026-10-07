"""1스텝 지연으로 학습한 카드 게임 모델을 더 긴 지연에서 다시 평가한다.

이 평가는 새로운 지연에 대한 일반화 검사다. 긴 지연으로 다시 학습한
결과와 혼동하지 않도록 학습 지연과 평가 지연을 모두 기록한다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from pilot_card_game import BaselinePolicy, FlyPolicy, evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("data/game_pilot"))
    parser.add_argument("--graph-dir", type=Path, default=Path("data/subgraphs"))
    parser.add_argument("--delays", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--eval-games", type=int, default=1024)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=Path("data/game_pilot/stage1_delay_generalization.json"))
    args = parser.parse_args()
    if not args.delays or min(args.delays) < 1 or args.eval_games < 1:
        parser.error("delays and eval-games must be positive")

    device = torch.device(args.device)
    rows = []
    # 이전 1단계 결과 파일만 읽는다. 새 지연 학습 결과와 섞지 않는다.
    paths = sorted(args.input_dir.glob("card_stage1_mlphead_reinforce_*.json"))
    paths = [path for path in paths if "_delay" not in path.stem]
    for path in paths:
        result = json.loads(path.read_text(encoding="utf-8"))
        size = result["neurons_reference"]
        kind = result["model"]
        variant = result.get("graph_variant", "real")
        seed = result["seed"]
        if kind == "fly":
            suffix = "" if variant == "real" else f"_rewired_seed{result['rewire_seed']}"
            graph_path = args.graph_dir / f"malecns_cx_{size}{suffix}.npz"
            model = FlyPolicy(graph_path)
        else:
            model = BaselinePolicy(kind, result["hidden_size"])
        checkpoint = torch.load(path.with_suffix(".pt"), map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["model_state"])
        model.to(device)
        for delay in args.delays:
            success, no_memory = evaluate(
                model, count=args.eval_games, blank_steps=delay,
                device=device, seed=seed + 30_000, stage=1,
            )
            row = {
                "model": kind,
                "graph_variant": variant,
                "neurons_reference": size,
                "seed": seed,
                "training_delay": result["blank_steps"],
                "evaluation_delay": delay,
                "eval_games": args.eval_games,
                "success_rate": success,
                "no_memory_success_rate": no_memory,
                "source_result": path.name,
            }
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({
            "protocol": "train at delay 1; evaluate same checkpoints at new delays without retraining",
            "rows": rows,
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Saved: {args.output} ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()

"""4스텝 카드 게임의 구조 재현과 RNN·GRU 학습률 점검을 순차 실행한다.

실험 단계를 독립적으로 다시 실행할 수 있으며, 완료된 결과 파일은
조건을 검증한 뒤 재사용한다. 모든 학습은 16GB GPU에서 한 번에 하나씩
실행한다.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PILOT = ROOT / "pilot_card_game.py"
OUT = ROOT / "data" / "game_pilot"
SIZES = (1000, 2000)
TOPOLOGY_SEEDS = (201, 202, 203, 204, 205)
REWIRE_SEEDS = (0, 1, 2, 3, 4)
TUNING_SEEDS = (101, 102, 103)
LEARNING_RATES = (0.0003, 0.001, 0.003)
BASELINES = ("rnn", "gru")


def lr_tag(rate: float) -> str:
    return f"lr_{rate:.4f}".replace(".", "p")


def result_path(output_dir: Path, model: str, size: int, seed: int,
                graph_variant: str = "real", rewire_seed: int = 0) -> Path:
    variant = "" if graph_variant == "real" else f"_rewired{rewire_seed}"
    stem = f"card_stage1_mlphead_reinforce_{model}{variant}_{size}_delay4_seed{seed}.json"
    return output_dir / stem


def run_one(*, model: str, size: int, seed: int, output_dir: Path,
            learning_rate: float = 0.001, graph_variant: str = "real",
            rewire_seed: int = 0) -> dict:
    path = result_path(output_dir, model, size, seed, graph_variant, rewire_seed)
    if path.exists() and path.with_suffix(".pt").exists():
        result = json.loads(path.read_text(encoding="utf-8"))
        expected = {
            "model": model,
            "stage": 1,
            "neurons_reference": size,
            "seed": seed,
            "blank_steps": 4,
            "graph_variant": graph_variant,
            "choice_interactions": 307200,
            "eval_games": 1024,
        }
        for key, value in expected.items():
            if result.get(key) != value:
                raise ValueError(f"기존 결과의 {key} 불일치: {path}")
        if abs(result.get("learning_rate", 0.001) - learning_rate) > 1e-12:
            raise ValueError(f"기존 결과의 학습률 불일치: {path}")
        print(f"SKIP {path.relative_to(ROOT)} success={result['final_fresh_success_rate']:.3f}", flush=True)
        return result

    command = [
        sys.executable, str(PILOT),
        "--stage", "1", "--blank-steps", "4", "--model", model,
        "--neurons", str(size), "--seed", str(seed),
        "--algorithm", "reinforce", "--updates", "1200", "--batch", "256",
        "--eval-every", "200", "--eval-games", "1024",
        "--learning-rate", str(learning_rate), "--entropy-bonus", "0.01",
        "--graph-variant", graph_variant, "--output-dir", str(output_dir),
    ]
    if graph_variant == "rewired":
        command.extend(("--rewire-seed", str(rewire_seed)))
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    if completed.returncode:
        print(completed.stdout, flush=True)
        print(completed.stderr, file=sys.stderr, flush=True)
        raise RuntimeError(f"학습 실패: {path}")
    result = json.loads(path.read_text(encoding="utf-8"))
    print(f"DONE {path.relative_to(ROOT)} success={result['final_fresh_success_rate']:.3f}", flush=True)
    return result


def run_topology() -> None:
    """독립된 학습 시드 5개와 차수 보존 재배선 그래프 5개를 비교한다."""
    for size in SIZES:
        for seed in TOPOLOGY_SEEDS:
            run_one(model="fly", size=size, seed=seed, output_dir=OUT)
        for graph_seed in REWIRE_SEEDS:
            graph = ROOT / "data" / "subgraphs" / f"malecns_cx_{size}_rewired_seed{graph_seed}.npz"
            if not graph.exists():
                raise FileNotFoundError(graph)
            for seed in TOPOLOGY_SEEDS:
                run_one(model="fly", size=size, seed=seed, output_dir=OUT,
                        graph_variant="rewired", rewire_seed=graph_seed)


def run_tuning() -> None:
    """기준 모델의 학습률만 바꾸고 조정용 시드에서 성공률을 잰다."""
    for size in SIZES:
        for model in BASELINES:
            for rate in LEARNING_RATES:
                output_dir = OUT / "lr_tuning" / lr_tag(rate)
                for seed in TUNING_SEEDS:
                    run_one(model=model, size=size, seed=seed,
                            output_dir=output_dir, learning_rate=rate)


def selected_rates() -> dict[tuple[int, str], float]:
    """조정 시드 평균 최종 성공률로 학습률을 고른다. 평가 시드는 쓰지 않는다."""
    choices = {}
    for size in SIZES:
        for model in BASELINES:
            candidates = []
            for rate in LEARNING_RATES:
                output_dir = OUT / "lr_tuning" / lr_tag(rate)
                scores = []
                for seed in TUNING_SEEDS:
                    path = result_path(output_dir, model, size, seed)
                    if not path.exists():
                        raise FileNotFoundError(path)
                    scores.append(json.loads(path.read_text(encoding="utf-8"))["final_fresh_success_rate"])
                candidates.append((sum(scores) / len(scores), rate))
            # 동점이면 더 작은 학습률을 택한다.
            _, selected = max(candidates, key=lambda item: (item[0], -item[1]))
            choices[(size, model)] = selected
            print(f"SELECT size={size} model={model} rate={selected} candidates={candidates}", flush=True)
    return choices


def run_heldout() -> None:
    """조정에 쓰지 않은 시드에서 선택 학습률의 성능을 확인한다."""
    choices = selected_rates()
    for size in SIZES:
        for model in BASELINES:
            rate = choices[(size, model)]
            for seed in TOPOLOGY_SEEDS:
                # 공통 학습률 0.001도 같은 시드에 돌려 설정 변경의 효과를
                # 시드별로 직접 비교한다.
                run_one(model=model, size=size, seed=seed,
                        output_dir=OUT / "heldout_default", learning_rate=0.001)
                if rate != 0.001:
                    run_one(model=model, size=size, seed=seed,
                            output_dir=OUT / "heldout_tuned", learning_rate=rate)
                else:
                    print(f"SAME size={size} model={model} seed={seed}: selected rate is default 0.001", flush=True)
    selected_path = OUT / "heldout_tuned" / "selected_rates.json"
    selected_path.write_text(json.dumps({
        "selection": "highest mean final success on tuning seeds 101,102,103; tie favors lower learning rate",
        "tuning_seeds": TUNING_SEEDS,
        "heldout_seeds": TOPOLOGY_SEEDS,
        "entropy_bonus": 0.01,
        "selected_rates": {f"{size}_{model}": choices[(size, model)] for size in SIZES for model in BASELINES},
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("topology", "tune", "heldout"))
    args = parser.parse_args()
    if args.phase == "topology":
        run_topology()
    elif args.phase == "tune":
        run_tuning()
    else:
        run_heldout()


if __name__ == "__main__":
    main()

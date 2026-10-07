"""직교 초기화의 실패 시드 복구와 새 시드 일반화를 순차 학습한다."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "game_pilot" / "rnn_recovery"
SIZES = (1000, 2000)
OLD_SEEDS = (201, 202, 203, 204, 205)
NEW_SEEDS = (206, 207, 208, 209, 210)


def run_one(size: int, seed: int, initialization: str, output_dir: Path) -> None:
    name = f"card_stage1_mlphead_reinforce_rnn_{size}_delay4_seed{seed}.json"
    path = output_dir / name
    if path.exists() and path.with_suffix(".pt").exists():
        result = json.loads(path.read_text(encoding="utf-8"))
        if (result["seed"] != seed or result["neurons_reference"] != size
                or result["rnn_init"] != initialization or result["blank_steps"] != 4
                or abs(result["learning_rate"] - 0.0003) > 1e-12):
            raise ValueError(f"기존 결과의 조건이 다릅니다: {path}")
        print(f"SKIP n={size} seed={seed} init={initialization} success={result['final_fresh_success_rate']:.3f}", flush=True)
        return
    command = [
        sys.executable, str(ROOT / "pilot_card_game.py"),
        "--stage", "1", "--blank-steps", "4", "--model", "rnn",
        "--neurons", str(size), "--seed", str(seed),
        "--learning-rate", "0.0003", "--entropy-bonus", "0.01",
        "--rnn-init", initialization, "--updates", "1200", "--batch", "256",
        "--eval-games", "1024", "--output-dir", str(output_dir),
    ]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    if completed.returncode:
        print(completed.stdout, flush=True)
        print(completed.stderr, file=sys.stderr, flush=True)
        raise RuntimeError(f"학습 실패: {path}")
    result = json.loads(path.read_text(encoding="utf-8"))
    print(f"DONE n={size} seed={seed} init={initialization} success={result['final_fresh_success_rate']:.3f}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("old", "new"))
    args = parser.parse_args()
    for size in SIZES:
        seeds = OLD_SEEDS if args.phase == "old" else NEW_SEEDS
        for seed in seeds:
            if args.phase == "new":
                run_one(size, seed, "default", OUT / "default_fresh")
            run_one(size, seed, "orthogonal", OUT / "orthogonal_controlled")


if __name__ == "__main__":
    main()

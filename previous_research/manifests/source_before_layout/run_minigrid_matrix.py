"""MiniGrid MemoryS11의 4모델×2크기×3시드 파일럿을 재시작 가능하게 실행한다."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "data" / "minigrid_pilot" / "fixed_cue"
MODELS = ("fly", "rewired", "rnn", "gru")
SIZES = (1000, 2000)
SEEDS = (301, 302, 303)


def run_one(model: str, size: int, seed: int) -> None:
    result_path = OUTPUT / f"memory_s11_{model}_{size}_seed{seed}.json"
    if result_path.exists() and result_path.with_suffix(".pt").exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if (result["model"] != model or result["neurons_reference"] != size
                or result["seed"] != seed or result["steps"] != 10240
                or result["start_mode"] != "fixed_cue"):
            raise ValueError(f"기존 결과와 실행 조건이 다릅니다: {result_path}")
        if model != "rewired" or result.get("paired_rewire_initialization", False):
            print(f"SKIP {model} {size} {seed}: {result['final_fresh_evaluation']['success_rate']:.3f}", flush=True)
            return
        # 초기 구현에서 재배선 그래프만 다른 난수 순서로 생성됐다.
        # 이전 원자료는 보존하고, 동일 초기 파라미터로 다시 학습한다.
        archive = OUTPUT / "unpaired_rewire_initialization"
        archive.mkdir(parents=True, exist_ok=True)
        for old_path in (result_path, result_path.with_suffix(".pt")):
            old_path.replace(archive / old_path.name)
        print(f"ARCHIVED prior unpaired rewire: {model} {size} {seed}", flush=True)
    command = [
        sys.executable, str(ROOT / "minigrid_memory_pilot.py"),
        "--model", model, "--neurons", str(size), "--size", "11",
        "--steps", "10240", "--seed", str(seed), "--output-dir", str(OUTPUT),
    ]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    if completed.returncode:
        print(completed.stdout, flush=True)
        print(completed.stderr, file=sys.stderr, flush=True)
        raise RuntimeError(f"학습 실패: {model} {size} {seed}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    print(f"DONE {model} {size} {seed}: "
          f"success={result['final_fresh_evaluation']['success_rate']:.3f}, "
          f"cue_masked={result['final_fresh_cue_masked_evaluation']['success_rate']:.3f}, "
          f"seconds={result['duration_seconds']:.1f}", flush=True)


def main() -> None:
    for seed in SEEDS:
        for size in SIZES:
            for model in MODELS:
                run_one(model, size, seed)


if __name__ == "__main__":
    main()

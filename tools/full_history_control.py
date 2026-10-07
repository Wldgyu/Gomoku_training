"""Check history-only training against the original cold-state budget."""
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "gomoku" if (ROOT / "gomoku").exists() else "gomoku9"
RUN = PACKAGE + "/runs/advanced_20261007"
OLD = PACKAGE + "/runs/defense12_20261006"


def run(*args):
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True)


def main():
    deadline = time.monotonic() + 7200
    while not any((ROOT / RUN / name).exists() for name in
                  ("parallel_seed4603/training_summary.json", "parallel_training_summary.json")):
        if time.monotonic() > deadline:
            raise TimeoutError("Independent runs not complete")
        time.sleep(5)
    run("-m", PACKAGE + ".train", "--size", "12", "--models", "fly", "gru", "--seeds", "4201",
        "--opponent", "weak", "--warmup-updates", "1800", "--updates", "1200",
        "--auxiliary-mode", "history", "--architecture", "global", "--tactical-data", OLD + "/data/train.npz",
        "--output", RUN + "/step1_full_history")
    run("-m", PACKAGE + ".advanced_evaluate", "--input", RUN + "/step1_full_history",
        "--data", OLD + "/data/validation.npz", "--models", "fly", "gru", "--seeds", "4201",
        "--games", "256", "--device", "cpu")
    print("Original-budget history control completed.", flush=True)


if __name__ == "__main__":
    main()

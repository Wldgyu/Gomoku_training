"""Finish fixed post-training checks after the ongoing evaluator exits."""
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "gomoku" if (ROOT / "gomoku").exists() else "gomoku9"
RUN = PACKAGE + "/runs/advanced_20261007"


def run(*args):
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True)


def main():
    deadline = time.monotonic() + 7200
    while not (ROOT / RUN / "final/evaluation.json").exists():
        if time.monotonic() > deadline:
            raise TimeoutError("Final fixed evaluation not available")
        time.sleep(5)
    run("-m", PACKAGE + ".advanced_evaluate", "--input", RUN + "/final", "--data",
        PACKAGE + "/runs/defense12_20261006/data/test.npz", "--models", "fly", "rewired", "rnn", "gru",
        "--seeds", "4601", "4602", "4603", "--tactical-only", "--device", "cpu",
        "--output-name", "evaluation_original_test.json")
    run("-m", PACKAGE + ".advanced_audit", "--input", RUN,
        "--previous", PACKAGE + "/runs/defense12_20261006/data")
    run("-m", PACKAGE + ".advanced_match_audit", "--input", RUN + "/final")
    run("-m", PACKAGE + ".advanced_report", "--input", RUN)
    print("Fixed evaluations, audit and report completed.", flush=True)


if __name__ == "__main__":
    main()

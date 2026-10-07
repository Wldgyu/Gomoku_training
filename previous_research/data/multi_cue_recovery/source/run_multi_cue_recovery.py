"""Ordered, resumable controller for the second three-stage study."""

import argparse
import subprocess
import sys

from multi_cue_memory import ROOT
from multi_cue_recovery import OUT

STAGES = (
    ("readout", "multi_cue_recovery.py", ("readout",)),
    ("visual", "multi_cue_observation_recovery.py", ()),
    ("repeat", "multi_cue_recovery.py", ("repeat",)),
    ("summary", "summarize_multi_cue_recovery.py", ()),
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--from-stage", choices=[s[0] for s in STAGES], default="readout"
    )
    parser.add_argument("--through", choices=[s[0] for s in STAGES], default="summary")
    args = parser.parse_args()
    first = next(i for i, s in enumerate(STAGES) if s[0] == args.from_stage)
    last = next(i for i, s in enumerate(STAGES) if s[0] == args.through)
    if first > last:
        parser.error("from-stage comes after through")
    OUT.mkdir(parents=True, exist_ok=True)
    for label, script, params in STAGES[first : last + 1]:
        print(f"START {label}", flush=True)
        with (OUT / f"{label}.log").open("a", encoding="utf-8") as log:
            subprocess.run(
                [sys.executable, "-u", str(ROOT / script), *params],
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
            if label == "visual":
                subprocess.run(
                    [sys.executable, "-u", str(ROOT / script), "--refresh-perception"],
                    cwd=ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
        print(f"COMPLETE {label}", flush=True)


if __name__ == "__main__":
    main()

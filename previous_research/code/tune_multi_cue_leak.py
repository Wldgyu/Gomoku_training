"""Diagnostic on tuning seeds: does the 65% plateau of graph-based memories depend on the
leak constant (0.9 in earlier modules) rather than on the edges? Validation accuracy only."""

from __future__ import annotations

import json

import torch

from multi_cue_memory import MemoryModel, ROOT, TaskConfig, train_one

LEAKS = (0.9, 0.5, 0.0)
KINDS = ("fly", "leaky")
RATES = (0.001, 0.003)
SEEDS = (901, 902)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = TaskConfig(2, 2, 4, 2)
    rows = []
    out = ROOT / "data" / "multi_cue" / "tuning_leak.json"
    for leak in LEAKS:
        for kind in KINDS:
            for rate in RATES:
                for seed in SEEDS:
                    result = train_one(kind, config, seed, 1000, updates=4000, batch_size=128,
                                       learning_rate=rate, eval_every=500, validation_size=1024,
                                       test_size=16, device=device, leak=leak)
                    rows.append({"leak": leak, "kind": kind, "lr": rate, "seed": seed,
                                 "val": result["final_validation"],
                                 "curve": result["curve"]})
                    print(json.dumps({"leak": leak, "kind": kind, "lr": rate, "seed": seed,
                                      "val": round(result["final_validation"], 3)}),
                          flush=True)
                    out.write_text(json.dumps(rows, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

"""Learning-rate / budget check on separate tuning seeds (validation accuracy only).

Conditions are the easy end of the load grid. Final comparison seeds are never used here.
"""

from __future__ import annotations

import json
import sys

import torch

from multi_cue_memory import (
    MemoryModel,
    ROOT,
    TaskConfig,
    matched_hidden,
    parameter_count,
    train_one,
)

RATES = (0.0003, 0.001, 0.003, 0.01)
CONDITIONS = (TaskConfig(1, 1, 0, 0), TaskConfig(2, 2, 4, 2))
KINDS = ("fly", "rnn", "gru", "leaky")
SEEDS = (901, 902)
UPDATES = 4000


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    target = parameter_count(MemoryModel("fly", 1000))
    hidden = {kind: matched_hidden(kind, target) for kind in KINDS}
    out = ROOT / "data" / "multi_cue" / "tuning.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for config in CONDITIONS:
        for kind in KINDS:
            for rate in RATES:
                for seed in SEEDS:
                    result = train_one(kind, config, seed, hidden[kind], updates=UPDATES,
                                       batch_size=128, learning_rate=rate, eval_every=500,
                                       validation_size=1024, test_size=16, device=device)
                    rows.append({key: result[key] for key in (
                        "task", "kind", "seed", "learning_rate", "final_validation", "curve")})
                    print(json.dumps({"task": config.name, "kind": kind, "lr": rate,
                                      "seed": seed,
                                      "val": round(result["final_validation"], 4)}),
                          flush=True)
                    out.write_text(json.dumps(rows, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())

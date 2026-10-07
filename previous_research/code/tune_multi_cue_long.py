"""Longer-budget check for the plateau at ~65% (tuning seeds only, validation accuracy)."""

from __future__ import annotations

import json

import torch

from multi_cue_memory import (
    MemoryModel,
    ROOT,
    TaskConfig,
    matched_hidden,
    parameter_count,
    train_one,
)

UPDATES = 16000
RATES = (0.0003, 0.003)
KINDS = ("fly", "leaky", "gru")


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    target = parameter_count(MemoryModel("fly", 1000))
    config = TaskConfig(2, 2, 4, 2)
    out = ROOT / "data" / "multi_cue" / "tuning_long.json"
    rows = []
    for kind in KINDS:
        for rate in RATES:
            result = train_one(kind, config, 901, matched_hidden(kind, target),
                               updates=UPDATES, batch_size=128, learning_rate=rate,
                               eval_every=1000, validation_size=1024, test_size=16,
                               device=device)
            rows.append({key: result[key] for key in (
                "task", "kind", "seed", "learning_rate", "final_validation", "curve")})
            print(json.dumps({"kind": kind, "lr": rate,
                              "val": [round(p["validation"], 3) for p in result["curve"]]}),
                  flush=True)
            out.write_text(json.dumps(rows, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

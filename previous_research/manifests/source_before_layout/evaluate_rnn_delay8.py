"""8스텝 RNN 초기화 비교를 같은 새 게임에서 재평가한다."""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import torch

from pilot_card_game import BaselinePolicy, evaluate

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "game_pilot" / "rnn_recovery"


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for size in (1000, 2000):
        for seed in (206, 207, 208):
            for initialization in ("default", "orthogonal"):
                folder = OUT / f"delay8_{initialization}"
                path = folder / f"card_stage1_mlphead_reinforce_rnn_{size}_delay8_seed{seed}.json"
                result = json.loads(path.read_text(encoding="utf-8"))
                model = BaselinePolicy("rnn", result["hidden_size"]).to(device)
                checkpoint = torch.load(path.with_suffix(".pt"), map_location=device, weights_only=True)
                model.load_state_dict(checkpoint["model_state"])
                success, no_memory = evaluate(
                    model, count=4096, blank_steps=8, device=device,
                    seed=90_000 + seed, stage=1,
                )
                rows.append({
                    "neurons_reference": size,
                    "seed": seed,
                    "initialization": initialization,
                    "success_rate": success,
                    "no_memory_success_rate": no_memory,
                    "eval_games": 4096,
                    "source": str(path.relative_to(ROOT)).replace("\\", "/"),
                })
    summary = {
        str(size): {
            initialization: statistics.mean(
                row["success_rate"] for row in rows
                if row["neurons_reference"] == size and row["initialization"] == initialization
            )
            for initialization in ("default", "orthogonal")
        }
        for size in (1000, 2000)
    }
    output = OUT / "rnn_delay8_test_4096.json"
    output.write_text(json.dumps({"rows": rows, "means": summary}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rows": rows, "means": summary}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

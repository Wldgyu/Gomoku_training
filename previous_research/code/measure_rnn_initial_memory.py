"""학습 전 RNN에서 첫 카드 정보가 공백 4스텝 동안 얼마나 남는지 측정한다."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from pilot_card_game import (
    BaselinePolicy,
    FlyPolicy,
    make_games,
    matched_hidden_size,
    parameter_count,
)

ROOT = Path(__file__).resolve().parent.parent


def measure(size: int, seed: int, initialization: str) -> dict:
    torch.manual_seed(seed)
    fly = FlyPolicy(ROOT / "data" / "subgraphs" / f"malecns_cx_{size}.npz")
    hidden = matched_hidden_size("rnn", parameter_count(fly))
    model = BaselinePolicy("rnn", hidden, rnn_init=initialization)
    model.eval()
    rng = torch.Generator().manual_seed(80_000 + seed)
    observations, _ = make_games(512, 4, torch.device("cpu"), rng)
    wiped = observations.clone()
    wiped[:, 0, :32] = 0
    with torch.no_grad():
        states, _ = model.core(observations)
        wiped_states, _ = model.core(wiped)
        difference = (states - wiped_states).norm(dim=2).mean(dim=0)
    values = difference.tolist()
    return {
        "neurons_reference": size,
        "seed": seed,
        "initialization": initialization,
        "first_card_state_distance_by_timestep": values,
        "final_to_initial_distance_ratio": values[-1] / values[0],
    }


def main() -> None:
    rows = [
        measure(size, seed, initialization)
        for size in (1000, 2000)
        for seed in (203, 204, 208, 209)
        for initialization in ("default", "orthogonal")
    ]
    output = ROOT / "data" / "game_pilot" / "rnn_initial_memory_decay.json"
    output.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for row in rows:
        print(row["neurons_reference"], row["seed"], row["initialization"],
              [round(value, 6) for value in row["first_card_state_distance_by_timestep"]],
              "retention", round(row["final_to_initial_distance_ratio"], 6))


if __name__ == "__main__":
    main()

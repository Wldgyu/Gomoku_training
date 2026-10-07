"""4스텝 RNN 실패 시드의 행동 분포와 과거 카드 의존성을 점검한다."""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F

from pilot_card_game import BaselinePolicy, make_games

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "game_pilot"


def diagnose(size: int, seed: int, count: int, device: torch.device) -> dict:
    path = OUT / "heldout_tuned" / f"card_stage1_mlphead_reinforce_rnn_{size}_delay4_seed{seed}.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    model = BaselinePolicy("rnn", result["hidden_size"]).to(device)
    checkpoint = torch.load(path.with_suffix(".pt"), map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    rng = torch.Generator(device=device).manual_seed(60_000 + seed)
    actions = torch.zeros(4, dtype=torch.int64, device=device)
    entropy_sum = 0.0
    first_card_effect = 0.0
    changed_decisions = 0
    correct = 0
    with torch.no_grad():
        for start in range(0, count, 256):
            batch = min(256, count - start)
            observations, answer = make_games(batch, 4, device, rng)
            logits = model(observations)
            probability = F.softmax(logits, dim=1)
            predicted = logits.argmax(1)
            actions += torch.bincount(predicted, minlength=4)
            entropy_sum += float((-(probability * probability.clamp_min(1e-12).log()).sum(1)).sum().item())
            correct += int((predicted == answer).sum().item())
            observations[:, 0, :32] = 0
            wiped = model(observations)
            first_card_effect += float((logits - wiped).abs().mean(1).sum().item())
            changed_decisions += int((predicted != wiped.argmax(1)).sum().item())
    return {
        "neurons_reference": size,
        "seed": seed,
        "games": count,
        "success_rate": correct / count,
        "greedy_action_fraction": (actions.float() / count).tolist(),
        "mean_policy_entropy_nats": entropy_sum / count,
        "max_entropy_nats": math.log(4),
        "mean_absolute_logit_change_after_wiping_first_cards": first_card_effect / count,
        "greedy_action_change_fraction_after_wiping_first_cards": changed_decisions / count,
    }


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = [diagnose(size, seed, 4096, device) for size in (1000, 2000) for seed in (203, 204)]
    output = OUT / "rnn_seed_diagnostics.json"
    output.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(rows, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

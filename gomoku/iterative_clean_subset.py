"""Common post-hoc test subset excluding any known training-position overlap."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .midgame_clean_subset import game_counters
from .train import atomic_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--prior", type=Path, required=True)
    args = parser.parse_args()

    audit = json.loads((args.root / "audit.json").read_text(encoding="utf-8"))
    excluded = {41}  # Previous A/B/Curriculum audit exact D4 training overlap.
    for row in audit["test_full_game_training_overlap"].values():
        excluded.update(row["opening_indexes"])

    locations = {
        "baseline": args.prior,
        "step1": args.prior,
        "c_dagger25": args.root,
        "d_threat_spatial": args.root,
    }
    rows = []
    for arm, location in locations.items():
        for kind in ("fly", "gru"):
            path = location / f"{arm}_{kind}_test_full_replays.json"
            games = json.loads(path.read_text(encoding="utf-8"))["games"]
            kept = [game for game in games if int(game["opening_index"]) not in excluded]
            stats = [game_counters(game) for game in kept]
            block_positions = sum(row["counters"]["block"]["positions"] for row in stats)
            block_correct = sum(row["counters"]["block"]["correct"] for row in stats)
            wins = sum(row["result"] == 1 for row in stats)
            losses = sum(row["result"] == -1 for row in stats)
            draws = sum(row["result"] == 0 for row in stats)
            rows.append({
                "arm": arm,
                "kind": kind,
                "games": len(kept),
                "wins": wins,
                "losses": losses,
                "draws": draws,
                "defense_opportunities": block_positions,
                "defense_omissions": block_positions - block_correct,
                "defense_omission_rate": (
                    (block_positions - block_correct) / block_positions if block_positions else None
                ),
                "games_reaching_multiple_threat": sum(row["reached"] for row in stats),
                "multiple_threat_game_rate": sum(row["reached"] for row in stats) / len(kept),
                "actual_safe_forks_created": sum(row["forks"] for row in stats),
            })

    summary = {}
    for arm in locations:
        selected = [row for row in rows if row["arm"] == arm]
        games = sum(row["games"] for row in selected)
        opportunities = sum(row["defense_opportunities"] for row in selected)
        omissions = sum(row["defense_omissions"] for row in selected)
        summary[arm] = {
            "games": games,
            "wins": sum(row["wins"] for row in selected),
            "defense_opportunities": opportunities,
            "defense_omissions": omissions,
            "defense_omission_rate": omissions / opportunities,
            "games_reaching_multiple_threat": sum(
                row["games_reaching_multiple_threat"] for row in selected
            ),
            "multiple_threat_game_rate": sum(
                row["games_reaching_multiple_threat"] for row in selected
            ) / games,
            "actual_safe_forks_created": sum(
                row["actual_safe_forks_created"] for row in selected
            ),
        }
    output = {
        "posthoc_supplementary": True,
        "common_excluded_opening_indexes": sorted(excluded),
        "excluded_openings": len(excluded),
        "retained_games_per_model": 256 - len(excluded),
        "runs": rows,
        "summary": summary,
        "interpretation": (
            "The same opening identifiers with an exact D4 training-position match in any "
            "A/B/C/D model are excluded from every model. Raw preregistered test results "
            "remain primary; this common subset checks robustness and was selected post hoc."
        ),
    }
    atomic_json(args.root / "clean_common_test.json", output)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

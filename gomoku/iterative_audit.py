"""Audit iterative DAgger/threat runs, budgets, labels, and held-out overlap."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from .env import BatchBoard
from .midgame_data import canonical, all_prefix_keys, read_records
from .multistep_tactics import multi_targets
from .train import atomic_json


def obs_cells(obs):
    color = 1 if obs[3, 0, 0] else -1
    cells = ((obs[0] - obs[1]) * color).reshape(-1).astype(np.int8)
    return cells, color


def canonical_from_npz(path):
    with np.load(path) as saved:
        observations = saved["obs"]
    return {canonical(*obs_cells(obs)) for obs in observations}


def audit_dagger(path, reserved):
    with np.load(path) as saved:
        observations = saved["obs"].copy()
        labels = saved["labels"].copy()
        categories = saved["category"].copy()
        histories = saved["histories"].copy()
        rounds = saved["round"].copy()
        keys = saved["keys"].tolist()
    assert len(keys) == len(set(keys))
    canonical_keys = set()
    for obs, label, category, history in zip(observations, labels, categories, histories):
        board = BatchBoard(1, 12)
        moves = history[history >= 0]
        for ply, action in enumerate(moves):
            board.place(np.array([action]), 1 if ply % 2 == 0 else -1)
        cells, color = obs_cells(obs)
        np.testing.assert_array_equal(board.board[0], cells)
        np.testing.assert_array_equal(board.observe(color)[0], obs)
        target, found_category = multi_targets(board, np.array([color], dtype=np.int8))
        assert int(found_category[0]) == int(category)
        np.testing.assert_array_equal(target[0], label)
        canonical_keys.add(canonical(cells, color))
    overlap = canonical_keys & reserved
    assert not overlap
    return {
        "records": len(keys),
        "categories": dict(sorted(Counter(map(int, categories)).items())),
        "rounds": dict(sorted(Counter(map(int, rounds)).items())),
        "canonical_records": len(canonical_keys),
        "heldout_canonical_overlap": len(overlap),
    }, canonical_keys


def full_game_overlap(path, training):
    payload = json.loads(path.read_text(encoding="utf-8"))
    games_with_overlap = 0
    positions = 0
    overlaps = 0
    indexes = []
    opening_indexes = []
    for index, game in enumerate(payload["games"]):
        board = BatchBoard(1, 12)
        touched = False
        for ply, action in enumerate(game["moves"]):
            color = 1 if ply % 2 == 0 else -1
            if color == int(game["a_color"]):
                positions += 1
                if canonical(board.board[0], color) in training:
                    overlaps += 1
                    touched = True
            board.place(np.array([action]), color)
        if touched:
            games_with_overlap += 1
            indexes.append(index)
            opening_indexes.append(int(game["opening_index"]))
    return {
        "games": len(payload["games"]),
        "learner_positions": positions,
        "overlapping_positions": overlaps,
        "games_with_overlap": games_with_overlap,
        "game_indexes": indexes,
        "opening_indexes": opening_indexes,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--prior", type=Path, required=True)
    args = parser.parse_args()

    validation = read_records(args.prior / "data" / "validation_starts.npz")
    test = read_records(args.prior / "data" / "test_starts.npz")
    reserved = all_prefix_keys(validation + test)
    base = canonical_from_npz(args.prior / "data" / "baseline_aux.npz")
    threat_train = canonical_from_npz(args.root / "data" / "threat_train.npz")
    threat_validation = canonical_from_npz(args.root / "data" / "threat_validation.npz")
    threat_test = canonical_from_npz(args.root / "data" / "threat_test.npz")
    assert not (threat_train & threat_validation)
    assert not (threat_train & threat_test)
    assert not (threat_validation & threat_test)

    runs = {}
    replay_overlap = {}
    expected = {
        "c_dagger25": {"auxiliary_batch": 128, "threat_batch": 0},
        "d_threat_spatial": {"auxiliary_batch": 64, "threat_batch": 64},
    }
    for arm in expected:
        for kind in ("fly", "gru"):
            stem = args.root / arm / f"{kind}_seed4810"
            result = json.loads(stem.with_suffix(".json").read_text(encoding="utf-8"))
            checkpoint = torch.load(stem.with_suffix(".pt"), map_location="cpu", weights_only=True)
            config = result["config"]
            assert result["completed_updates"] == config["updates"] == checkpoint["completed_updates"] == 300
            assert result["learner_decisions"] == 300 * 32 * 16 == 153600
            assert config["warmup_updates"] == 600 and config["epochs"] == 2
            assert config["auxiliary_batch"] == expected[arm]["auxiliary_batch"]
            threat_batch = config.get("threat_batch", 0)
            assert threat_batch == expected[arm]["threat_batch"]
            optimizer_steps = config["warmup_updates"] + config["updates"] * config["epochs"]
            total_auxiliary_examples = optimizer_steps * (config["auxiliary_batch"] + threat_batch)
            assert total_auxiliary_examples == 153600
            dagger, dagger_keys = audit_dagger(
                stem.with_name(stem.name + "_dagger").with_suffix(".npz"), reserved
            )
            assert len(result["dagger_stats"]) == 3
            assert result["dagger_records"] == dagger["records"]
            runs[f"{arm}_{kind}"] = {
                "parameters": result["parameters"],
                "completed_updates": result["completed_updates"],
                "learner_decisions": result["learner_decisions"],
                "optimizer_steps": optimizer_steps,
                "policy_auxiliary_examples": optimizer_steps * config["auxiliary_batch"],
                "threat_auxiliary_examples": optimizer_steps * threat_batch,
                "total_auxiliary_examples": total_auxiliary_examples,
                "dagger_games": sum(row["games"] for row in result["dagger_stats"]),
                "dagger": dagger,
                "checkpoint_sha256": hashlib.sha256(stem.with_suffix(".pt").read_bytes()).hexdigest(),
            }
            training = base | dagger_keys
            if arm == "d_threat_spatial":
                training |= threat_train
            replay_overlap[f"{arm}_{kind}"] = full_game_overlap(
                args.root / f"{arm}_{kind}_test_full_replays.json", training
            )

    output = {
        "status": "passed",
        "reserved_midgame_canonical_prefixes": len(reserved),
        "base_training_canonical_positions": len(base),
        "threat_split_canonical_positions": {
            "train": len(threat_train),
            "validation": len(threat_validation),
            "test": len(threat_test),
        },
        "threat_split_overlap": {
            "train_validation": len(threat_train & threat_validation),
            "train_test": len(threat_train & threat_test),
            "validation_test": len(threat_validation & threat_test),
        },
        "runs": runs,
        "test_full_game_training_overlap": replay_overlap,
    }
    atomic_json(args.root / "audit.json", output)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


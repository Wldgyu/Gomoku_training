"""동일 단계 Fly·재배선 Fly 5시드의 학습 경로와 S9 결과를 감사·요약."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from minigrid_auxiliary_curriculum import parameter_hash, topology_hash
from minigrid_memory_diagnostic import build_model
from minigrid_memory_pilot import ROOT

BASE = ROOT / "data" / "minigrid_matched_topology"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def stage_paths(model: str, seed: int) -> dict[str, Path]:
    old = seed <= 403
    return {
        "diagnostic": (ROOT / "data" / "minigrid_diagnostic" if old else BASE / "diagnostic")
        / f"choice_s11_{model}_1000_seed{seed}.json",
        "full": (ROOT / "data" / "minigrid_curriculum" if old else BASE / "full")
        / f"full_s11_{model}_1000_seed{seed}.json",
        "random": (ROOT / "data" / "minigrid_random_start" / "from_full"
                   if model == "fly" and old else BASE / "random")
        / f"random_s11_{model}_1000_seed{seed}.json",
        "length": BASE / "length" / f"mixed_s7_11_13_{model}_1000_seed{seed}.json",
    }


def audit_row(model: str, seed: int) -> dict:
    paths = stage_paths(model, seed)
    data = {stage: load(path) for stage, path in paths.items()}
    diagnostic, full, random, length = (data[stage] for stage in ("diagnostic", "full", "random", "length"))
    assert (diagnostic["train_episodes"], diagnostic["epochs"],
            diagnostic["learning_rate"], diagnostic["cue_loss_weight"]) == (1024, 120, 0.0003, 0.5)
    assert (full["train_episodes"], full["epochs"]) == (1024, 30)
    assert (random["train_episodes"], random["epochs"], random["learning_rate"]) == (2048, 30, 0.0001)
    assert (length["train_episodes_per_size"], length["epochs"],
            length["learning_rate"], length["eval_episodes_final_per_size"]) == (512, 15, 0.0001, 128)
    assert list(length["train_sizes"]) == [7, 11, 13] and length["holdout_size"] == 9
    assert "S9 excluded" in length["selection_rule"]
    assert Path(full["source_checkpoint"]).stem == paths["diagnostic"].stem
    assert Path(random["source_checkpoint"]).stem == paths["full"].stem
    assert Path(length["source_checkpoint"]).stem == paths["random"].stem
    final = length["final_fresh"]
    return {"model": model, "seed": seed,
            "stage_paths": {stage: str(path) for stage, path in paths.items()},
            "diagnostic_s11_accuracy": diagnostic["history"][-1]["heldout_accuracy"],
            "full_s11_success": full["final_fresh"]["success_rate"],
            "random_s11_success": random["final_fresh"]["success_rate"],
            "random_s11_swap_success": random["final_fresh_cue_swapped"]["success_rate"],
            "selected_length_epoch": length["selected_epoch"],
            "train_length_success": {size: final[size]["normal"]["success_rate"]
                                     for size in ("7", "11", "13")},
            "s9_success": final["9"]["normal"]["success_rate"],
            "s9_branch_reach": final["9"]["normal"]["branch_reach_rate"],
            "s9_cue_seen": final["9"]["normal"]["cue_seen_rate"],
            "s9_mask_success": final["9"]["masked"]["success_rate"],
            "s9_swap_success": final["9"]["swapped"]["success_rate"]}


def main() -> None:
    rows = [audit_row(model, seed) for model in ("fly", "rewired") for seed in range(401, 406)]
    initial = []
    device = torch.device("cpu")
    for seed in range(401, 406):
        fly = build_model("fly", 1000, seed, device)
        rewired = build_model("rewired", 1000, seed, device)
        assert parameter_hash(fly) == parameter_hash(rewired)
        assert topology_hash(fly) != topology_hash(rewired)
        initial.append({"seed": seed, "equal_trainable_initial_weights": True,
                        "different_topology": True, "initial_parameter_sha256": parameter_hash(fly),
                        "fly_topology_sha256": topology_hash(fly),
                        "rewired_topology_sha256": topology_hash(rewired)})
    paired = []
    for seed in range(401, 406):
        fly = next(row for row in rows if row["model"] == "fly" and row["seed"] == seed)
        rewired = next(row for row in rows if row["model"] == "rewired" and row["seed"] == seed)
        paired.append({"seed": seed, "fly_s9": fly["s9_success"],
                       "rewired_s9": rewired["s9_success"],
                       "fly_minus_rewired": fly["s9_success"] - rewired["s9_success"]})
    summary = {"rows": rows, "initialization_audit": initial, "paired_s9": paired,
               "fly_mean_s9": float(np.mean([row["s9_success"] for row in rows if row["model"] == "fly"])),
               "rewired_mean_s9": float(np.mean([row["s9_success"] for row in rows if row["model"] == "rewired"])),
               "mean_paired_difference": float(np.mean([row["fly_minus_rewired"] for row in paired]))}
    destination = BASE / "summary.json"
    destination.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()

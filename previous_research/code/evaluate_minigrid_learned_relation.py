"""Evaluate learned cue memory and comparison during autonomous MiniGrid navigation."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from minigrid_compositional_policy import VisualRelations
from minigrid_learned_relation import LearnedRelation
from minigrid_memory_diagnostic import build_model
from minigrid_memory_pilot import ROOT, tensor_observation
from minigrid_relation_data import VARIANTS, make_env, reset_variant


@torch.no_grad()
def evaluate(actor, detector, memory, *, size: int, start_seed: int,
             base_count: int, device: torch.device, ablation: str = "none") -> dict:
    actor.eval()
    detector.eval()
    memory.eval()
    actor_matrix = actor.recurrent_matrix()
    memory_matrix = memory.matrix()
    envs, observations, metas = [], [], []
    for seed in range(start_seed, start_seed + base_count):
        for cue_flip, target_flip in VARIANTS:
            env = make_env(size)
            observation, meta = reset_variant(env, seed, cue_flip=cue_flip,
                                              target_flip=target_flip)
            envs.append(env)
            observations.append(observation)
            metas.append(meta)
    count = len(envs)
    actor_states = actor.initial_state(count, device)
    memory_states = torch.zeros(count, memory.hidden, device=device)
    done = [False] * count
    outcomes = [False] * count
    did_override = [False] * count
    did_reach = [False] * count
    cue_correct = [False] * count
    cue_correct_at_junction = [False] * count
    early_override = [False] * count
    branch_steps = [0] * count
    time_steps = [0] * count
    try:
        while not all(done):
            active = [index for index, finished in enumerate(done) if not finished]
            indices = torch.tensor(active, device=device)
            image, direction = tensor_observation([observations[index] for index in active], device)
            actor_logits, _, next_actor = actor.step(
                image, direction, actor_states[indices], actor_matrix)
            cue_logits, branch_logits, upper_logits = detector(image, direction)
            cue_input = cue_logits.softmax(dim=-1)[:, 1:3]
            if ablation == "mask_cue":
                cue_input = torch.zeros_like(cue_input)
            previous_memory = memory_states[indices]
            override = branch_logits.argmax(dim=-1) == 1
            if ablation == "reset_at_branch":
                previous_memory = torch.where(override.unsqueeze(1),
                                              torch.zeros_like(previous_memory), previous_memory)
            next_memory = memory.memory_step(
                cue_input, previous_memory, memory_matrix)
            action = actor_logits.argmax(dim=-1)
            if override.any():
                upper = upper_logits.softmax(dim=-1)
                learned_turn = memory.turn_from_state(next_memory, upper).argmax(dim=-1)
                action = torch.where(override, learned_turn, action)
            actions = action.cpu().tolist()
            override_flags = override.cpu().tolist()
            actor_states[indices] = next_actor
            memory_states[indices] = next_memory
            for local, index in enumerate(active):
                if override_flags[local] and not did_override[index]:
                    cue_correct[index] = (int(memory.cue_readout(next_memory[local:local + 1]).argmax(
                        dim=-1).item()) == int(metas[index]["cue_type"] == "Key"))
                    branch_steps[index] = time_steps[index]
                did_override[index] |= override_flags[local]
                base = envs[index].unwrapped
                at_junction = tuple(base.agent_pos) == (metas[index]["junction_x"], base.height // 2)
                if override_flags[local] and not at_junction:
                    early_override[index] = True
                if at_junction and not did_reach[index]:
                    cue_correct_at_junction[index] = (int(memory.cue_readout(
                        next_memory[local:local + 1]).argmax(dim=-1).item())
                        == int(metas[index]["cue_type"] == "Key"))
                did_reach[index] |= at_junction
                observation, reward, terminated, truncated, _ = envs[index].step(actions[local])
                observations[index] = observation
                time_steps[index] += 1
                if terminated or truncated:
                    done[index] = True
                    outcomes[index] = reward > 0
    finally:
        for env in envs:
            env.close()
    successes = sum(outcomes)
    all_four = sum(all(outcomes[index:index + 4]) for index in range(0, count, 4))
    overrides = sum(did_override)
    reached = sum(did_reach)
    episodes = base_count * 4
    return {"base_episodes": base_count, "success_rate": successes / episodes,
            "all_four_success": all_four / base_count,
            "branch_override_rate": overrides / episodes,
            "branch_reach_rate": reached / episodes,
            "cue_accuracy_at_override": sum(cue_correct) / episodes,
            "cue_accuracy_at_junction": sum(cue_correct_at_junction) / episodes,
            "early_override_rate": sum(early_override) / episodes,
            "mean_branch_step": sum(branch_steps) / episodes}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=("fly", "rewired", "rnn", "gru"),
                        default=["fly", "rewired", "rnn", "gru"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[601, 602, 603, 604, 605])
    parser.add_argument("--sizes", nargs="+", type=int, default=[9, 15, 19])
    parser.add_argument("--base-per-size", type=int, default=32)
    parser.add_argument("--start-seed", type=int, default=37_000_000)
    parser.add_argument("--ablation", choices=("none", "mask_cue", "reset_at_branch"),
                        default="none")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint-dir", type=Path,
                        default=ROOT / "data/minigrid_learned_relation/main")
    parser.add_argument("--actor-checkpoint", type=Path,
                        default=ROOT / "data/minigrid_relation_curriculum/relation_variable_control_rnn_1000_seed401.pt")
    parser.add_argument("--detector-checkpoint", type=Path,
                        default=ROOT / "data/minigrid_compositional/compositional_rnn_seed401_detector501.pt")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "data/minigrid_learned_relation/autonomous.json")
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.set_num_threads(min(torch.get_num_threads(), 8))
    actor = build_model("rnn", 1000, 401, device)
    actor.load_state_dict(torch.load(args.actor_checkpoint, map_location=device,
                                     weights_only=False)["model_state"])
    detector = VisualRelations().to(device)
    detector.load_state_dict(torch.load(args.detector_checkpoint, map_location=device,
                                        weights_only=False)["detector_state"])
    started = time.perf_counter()
    rows = []
    for seed in args.seeds:
        for kind in args.models:
            checkpoint = torch.load(args.checkpoint_dir / f"learned_relation_{kind}_seed{seed}.pt",
                                    map_location=device, weights_only=False)
            memory = LearnedRelation(kind, checkpoint["result"]["hidden"], seed,
                                     checkpoint["result"].get("rewire_seed", 0)).to(device)
            memory.load_state_dict(checkpoint["model_state"])
            result = {str(size): evaluate(actor, detector, memory, size=size,
                                          start_seed=args.start_seed + size * 100_000,
                                          base_count=args.base_per_size, device=device,
                                          ablation=args.ablation)
                      for size in args.sizes}
            row = {"kind": kind, "seed": seed, "sizes": result}
            rows.append(row)
            print(json.dumps(row), flush=True)
    summary = {"navigation_actor": str(args.actor_checkpoint),
               "detector": str(args.detector_checkpoint), "checkpoint_dir": str(args.checkpoint_dir),
               "start_seed": args.start_seed, "base_per_size": args.base_per_size,
               "ablation": args.ablation,
               "duration_seconds": time.perf_counter() - started, "runs": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

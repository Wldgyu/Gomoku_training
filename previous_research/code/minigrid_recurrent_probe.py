"""분기점 순환 상태에 남은 단서 종류를 읽어 미사용 길이에서 검증."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from minigrid_memory_diagnostic import build_model
from minigrid_memory_pilot import ROOT, tensor_observation
from minigrid_relation_data import VARIANTS, teacher_episode


@torch.no_grad()
def branch_states(actor, *, size: int, seeds: range,
                  device: torch.device, reset_at_branch: bool = False
                  ) -> tuple[torch.Tensor, torch.Tensor, list[dict]]:
    actor.eval()
    states, labels, metadata = [], [], []
    matrix = actor.recurrent_matrix()
    for seed in seeds:
        for cue_flip, target_flip in VARIANTS:
            observations, actions, meta = teacher_episode(
                seed, size, cue_flip=cue_flip, target_flip=target_flip
            )
            images, directions = tensor_observation(observations, device)
            state = actor.initial_state(1, device)
            for index in range(len(actions) - 1):
                if reset_at_branch and index == len(actions) - 2:
                    state = actor.initial_state(1, device)
                _, _, state = actor.step(images[index:index + 1], directions[index:index + 1], state, matrix)
            states.append(state[0].detach().cpu())
            labels.append(int(meta["cue_type"] == "Key"))
            metadata.append(meta)
    return torch.stack(states), torch.tensor(labels), metadata


@torch.no_grad()
def accuracy(probe: nn.Module, states: torch.Tensor, labels: torch.Tensor,
             device: torch.device) -> float:
    probe.eval()
    correct = 0
    for start in range(0, len(labels), 512):
        stop = start + 512
        logits = probe(states[start:stop].to(device))
        correct += int((logits.argmax(dim=-1).cpu() == labels[start:stop]).sum())
    return correct / len(labels)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("fly", "rewired", "rnn", "gru"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train-base-episodes-per-size", type=int, default=256)
    parser.add_argument("--validation-base-episodes", type=int, default=64)
    parser.add_argument("--final-base-episodes", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "minigrid_recurrent_probe")
    args = parser.parse_args()
    device = torch.device(args.device)
    actor = build_model(args.model, 1000, args.seed, device)
    actor.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=False)["model_state"])
    train_parts = [branch_states(actor, size=size,
                                 seeds=range(30_000_000 + size * 100_000,
                                             30_000_000 + size * 100_000 + args.train_base_episodes_per_size),
                                 device=device)
                   for size in (7, 11, 13)]
    train_states = torch.cat([part[0] for part in train_parts])
    train_labels = torch.cat([part[1] for part in train_parts])
    validation_parts = [branch_states(actor, size=size,
                                      seeds=range(31_000_000, 31_000_000 + args.validation_base_episodes),
                                      device=device)
                        for size in (7, 11, 13)]
    validation_states = torch.cat([part[0] for part in validation_parts])
    validation_labels = torch.cat([part[1] for part in validation_parts])
    torch.manual_seed(args.seed + 31_000)
    probe = nn.Linear(actor.hidden_size, 2).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=0.01)
    generator = torch.Generator().manual_seed(args.seed + 32_000)
    best_state, best_accuracy, selected_epoch, history = None, -1.0, 0, []
    for epoch in range(1, args.epochs + 1):
        probe.train()
        order = torch.randperm(len(train_labels), generator=generator)
        for indices in order.split(256):
            logits = probe(train_states[indices].to(device))
            loss = F.cross_entropy(logits, train_labels[indices].to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            score = accuracy(probe, validation_states, validation_labels, device)
            history.append({"epoch": epoch, "validation_accuracy": score})
            print(json.dumps(history[-1]), flush=True)
            if score > best_accuracy:
                best_state, best_accuracy, selected_epoch = copy.deepcopy(probe.state_dict()), score, epoch
    assert best_state is not None
    probe.load_state_dict(best_state)
    final = {}
    for size in (9, 15, 19):
        seeds = range(32_000_000, 32_000_000 + args.final_base_episodes)
        states, labels, _ = branch_states(actor, size=size, seeds=seeds, device=device)
        reset_states, _, _ = branch_states(actor, size=size, seeds=seeds,
                                           device=device, reset_at_branch=True)
        final[str(size)] = {"accuracy": accuracy(probe, states, labels, device),
                            "reset_at_branch_accuracy": accuracy(probe, reset_states, labels, device),
                            "episodes": len(labels)}
    result = {"model": args.model, "seed": args.seed, "actor_checkpoint": str(args.checkpoint),
              "probe": "linear classifier of frozen recurrent state at junction",
              "train_sizes": (7, 11, 13), "holdout_sizes": (9, 15, 19),
              "train_base_episodes_per_size": args.train_base_episodes_per_size,
              "validation_base_episodes": args.validation_base_episodes,
              "final_base_episodes": args.final_base_episodes,
              "selected_epoch": selected_epoch, "history": history,
              "final": final}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"cue_probe_{args.model}_seed{args.seed}_{args.checkpoint.stem}"
    (args.output_dir / f"{stem}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                                                 encoding="utf-8")
    torch.save({"probe_state": probe.state_dict(), "result": result}, args.output_dir / f"{stem}.pt")
    print(json.dumps({"selected_epoch": selected_epoch, "final": final}), flush=True)


if __name__ == "__main__":
    main()

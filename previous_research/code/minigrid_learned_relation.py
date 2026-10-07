"""Frozen visual detector followed by a trainable recurrent cue memory and turn comparator.

The detector supplies only current partial-observation predictions. The recurrent
module receives cue probabilities, and a bilinear layer learns the branch choice from its
state and the current upper target probabilities. No equality operation is used.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from minigrid_compositional_policy import VisualRelations
from minigrid_memory_pilot import GRAPH_DIR, ROOT, parameter_count
from minigrid_relation_data import build_dataset


class LearnedRelation(nn.Module):
    def __init__(self, kind: str, hidden: int, seed: int, rewire_seed: int = 0) -> None:
        super().__init__()
        self.kind = kind
        self.hidden = hidden
        if kind in ("fly", "rewired"):
            suffix = "" if kind == "fly" else f"_rewired_seed{rewire_seed}"
            with np.load(GRAPH_DIR / f"malecns_cx_1000{suffix}.npz") as graph:
                pre = torch.from_numpy(graph["pre_index"].astype(np.int64))
                post = torch.from_numpy(graph["post_index"].astype(np.int64))
                self.hidden = len(graph["body_ids"])
            degree = torch.bincount(post, minlength=self.hidden).clamp(min=1)
            self.register_buffer("pre", pre)
            self.register_buffer("post", post)
            self.register_buffer("edge_scale", degree[post].float().rsqrt())
            self.edge_values = nn.Parameter(torch.randn(len(pre)) * 0.2)
            self.input_layer = nn.Linear(2, self.hidden)
        else:
            cell = nn.RNNCell if kind == "rnn" else nn.GRUCell
            self.core = cell(2, hidden)
            if kind == "rnn":
                with torch.random.fork_rng(devices=[]):
                    nn.init.orthogonal_(self.core.weight_hh)
        self.cue_readout = nn.Linear(self.hidden, 2)
        # A two-value learned readout is the only cue information the comparator receives.
        self.comparator = nn.Bilinear(2, 2, 2)
        self.comparator_skip = nn.Linear(4, 2)

    def matrix(self) -> torch.Tensor | None:
        if self.kind not in ("fly", "rewired"):
            return None
        result = self.edge_values.new_zeros(self.hidden, self.hidden)
        return result.index_put((self.post, self.pre), self.edge_values * self.edge_scale,
                                accumulate=True)

    def memory_step(self, cue: torch.Tensor, state: torch.Tensor,
                    matrix: torch.Tensor | None = None) -> torch.Tensor:
        if self.kind not in ("fly", "rewired"):
            return self.core(cue, state)
        if matrix is None:
            matrix = self.matrix()
        return 0.9 * state + 0.1 * torch.tanh(self.input_layer(cue) + F.linear(state, matrix))

    def turn_from_state(self, state: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
        cue_probabilities = self.cue_readout(state).softmax(dim=-1)
        return (self.comparator(cue_probabilities, upper)
                + self.comparator_skip(torch.cat((cue_probabilities, upper), dim=1)))

    def forward(self, cues: torch.Tensor, branch_index: torch.Tensor,
                upper: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # cues: batch × time × (Ball, Key). Every padded step is inactive.
        state = cues.new_zeros(len(cues), self.hidden)
        picked = torch.zeros_like(state)
        matrix = self.matrix()
        for step in range(int(branch_index.max().item()) + 1):
            state = self.memory_step(cues[:, step], state, matrix)
            picked = torch.where((branch_index == step).unsqueeze(1), state, picked)
        cue_logits = self.cue_readout(picked)
        return self.turn_from_state(picked, upper), cue_logits


def matched_hidden(fly_parameters: int, kind: str) -> int:
    if kind in ("fly", "rewired"):
        return 1000
    def count(hidden: int) -> int:
        model = LearnedRelation(kind, hidden, 0)
        return parameter_count(model)
    low, high = 1, 512
    while count(high) < fly_parameters:
        high *= 2
    while low < high:
        middle = (low + high) // 2
        if count(middle) < fly_parameters:
            low = middle + 1
        else:
            high = middle
    return min((low - 1, low), key=lambda n: abs(count(n) - fly_parameters))


@torch.no_grad()
def features(rows: list[dict], detector: VisualRelations, device: torch.device,
             detector_batch: int = 1024) -> dict[str, torch.Tensor]:
    detector.eval()
    lengths = [len(row["actions"]) for row in rows]
    images = torch.cat([row["images"] for row in rows]).to(device)
    directions = torch.cat([row["directions"] for row in rows]).to(device)
    cue_parts, upper_parts = [], []
    for image, direction in zip(images.split(detector_batch), directions.split(detector_batch)):
        cue_logits, _, upper_logits = detector(image, direction)
        cue_parts.append(cue_logits.softmax(dim=-1)[:, 1:3].cpu())
        upper_parts.append(upper_logits.softmax(dim=-1).cpu())
    cue_frames, upper_frames = torch.cat(cue_parts), torch.cat(upper_parts)
    branch_indices = [row.get("branch_index", length - 2) for row, length in zip(rows, lengths)]
    padded = torch.zeros(len(rows), max(branch_indices) + 1, 2)
    upper = torch.zeros(len(rows), 2)
    offsets, branch = 0, []
    for index, length in enumerate(lengths):
        branch_at = branch_indices[index]
        padded[index, :branch_at + 1] = cue_frames[offsets:offsets + branch_at + 1]
        upper[index] = upper_frames[offsets + branch_at]
        branch.append(branch_at)
        offsets += length
    return {"cue": padded, "upper": upper,
            "branch": torch.tensor(branch),
            "turn": torch.tensor([row["meta"]["correct_turn"] for row in rows]),
            "cue_label": torch.tensor([int(row["meta"]["cue_type"] == "Key") for row in rows]),
            "seed": torch.tensor([row["seed"] for row in rows])}


def data_for_sizes(detector: VisualRelations, device: torch.device, sizes: tuple[int, ...],
                   start_seed: int, base_count: int) -> dict[int, dict]:
    result = {}
    for size in sizes:
        rows = build_dataset(range(start_seed + size * 100_000,
                                   start_seed + size * 100_000 + base_count), size)
        result[size] = features(rows, detector, device)
    return result


def combine(parts: dict[int, dict]) -> dict:
    max_length = max(item["cue"].shape[1] for item in parts.values())
    keys = next(iter(parts.values())).keys()
    return {key: torch.cat([
        F.pad(item[key], (0, 0, 0, max_length - item[key].shape[1])) if key == "cue" else item[key]
        for item in parts.values()
    ]) for key in keys}


def select(data: dict, indices: torch.Tensor, device: torch.device) -> dict:
    return {key: value[indices].to(device) for key, value in data.items()}


def pretrain_comparator(model: LearnedRelation, train: dict, device: torch.device,
                        steps: int = 200) -> float:
    """Learn the 2×2 relation from labeled training episodes, then freeze it."""
    cue = F.one_hot(train["cue_label"].to(device), 2).float()
    upper = train["upper"].to(device)
    labels = train["turn"].to(device)
    parameters = list(model.comparator.parameters()) + list(model.comparator_skip.parameters())
    optimizer = torch.optim.Adam(parameters, lr=0.03)
    for _ in range(steps):
        logits = model.comparator(cue, upper) + model.comparator_skip(torch.cat((cue, upper), dim=1))
        loss = F.cross_entropy(logits, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    accuracy = float((logits.argmax(dim=-1) == labels).float().mean())
    for parameter in parameters:
        parameter.requires_grad_(False)
    return accuracy


@torch.no_grad()
def evaluate(model: LearnedRelation, data: dict, device: torch.device,
             batch_size: int = 256) -> dict:
    model.eval()
    predictions, cue_predictions, losses = [], [], []
    for indices in torch.arange(len(data["turn"])).split(batch_size):
        batch = select(data, indices, device)
        turn, cue = model(batch["cue"], batch["branch"], batch["upper"])
        predictions.append(turn.argmax(dim=-1).cpu())
        cue_predictions.append(cue.argmax(dim=-1).cpu())
        losses.append(float(F.cross_entropy(turn, batch["turn"], reduction="sum")
                            + F.cross_entropy(cue, batch["cue_label"], reduction="sum")))
    prediction = torch.cat(predictions)
    cue_prediction = torch.cat(cue_predictions)
    groups = [prediction[i:i + 4].eq(data["turn"][i:i + 4]).all().item()
              for i in range(0, len(prediction), 4)]
    return {"episodes": len(prediction),
            "turn_accuracy": float(prediction.eq(data["turn"]).float().mean()),
            "all_four_success": float(np.mean(groups)),
            "cue_accuracy": float(cue_prediction.eq(data["cue_label"]).float().mean()),
            "loss": sum(losses) / len(prediction)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=("fly", "rewired", "rnn", "gru"),
                        default=["fly", "rewired", "rnn", "gru"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[601, 602, 603, 604, 605])
    parser.add_argument("--train-base-per-size", type=int, default=128)
    parser.add_argument("--validation-base-per-size", type=int, default=32)
    parser.add_argument("--final-base-per-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--detector-checkpoint", type=Path,
                        default=ROOT / "data/minigrid_compositional/compositional_rnn_seed401_detector501.pt")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/minigrid_learned_relation")
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.set_num_threads(min(torch.get_num_threads(), 8))
    started = time.perf_counter()
    detector = VisualRelations().to(device)
    detector.load_state_dict(torch.load(args.detector_checkpoint, map_location=device,
                                        weights_only=False)["detector_state"])
    train = combine(data_for_sizes(detector, device, (7, 11, 13), 34_000_000,
                                   args.train_base_per_size))
    validation = combine(data_for_sizes(detector, device, (7, 11, 13), 35_000_000,
                                        args.validation_base_per_size))
    final = data_for_sizes(detector, device, (9, 15, 19), 36_000_000,
                           args.final_base_per_size)
    torch.manual_seed(0)
    fly_count = parameter_count(LearnedRelation("fly", 1000, 0))
    hidden = {kind: matched_hidden(fly_count, kind) for kind in args.models}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for seed in args.seeds:
        for kind in args.models:
            torch.manual_seed(seed)
            model = LearnedRelation(kind, hidden[kind], seed).to(device)
            comparator_train_accuracy = pretrain_comparator(model, train, device)
            optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                          lr=args.learning_rate)
            generator = torch.Generator().manual_seed(seed + 1000)
            best_state, best_score, best_epoch = None, (-1.0, -1.0, -1.0), 0
            history = []
            for epoch in range(1, args.epochs + 1):
                model.train()
                losses = []
                order = torch.randperm(len(train["turn"]), generator=generator)
                for indices in order.split(args.batch_size):
                    batch = select(train, indices, device)
                    turn, cue = model(batch["cue"], batch["branch"], batch["upper"])
                    loss = F.cross_entropy(turn, batch["turn"]) + F.cross_entropy(
                        cue, batch["cue_label"])
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    optimizer.step()
                    losses.append(float(loss.detach()))
                valid = evaluate(model, validation, device)
                score = (valid["all_four_success"], valid["turn_accuracy"], valid["cue_accuracy"])
                history.append({"epoch": epoch, "loss": float(np.mean(losses)), "validation": valid})
                if score > best_score:
                    best_state, best_score, best_epoch = copy.deepcopy(model.state_dict()), score, epoch
                if epoch == 1 or epoch == args.epochs or epoch % 4 == 0:
                    print(json.dumps({"kind": kind, "seed": seed, "epoch": epoch,
                                      "loss": history[-1]["loss"], "validation": valid}), flush=True)
            model.load_state_dict(best_state)
            heldout = {str(size): evaluate(model, data, device) for size, data in final.items()}
            row = {"kind": kind, "seed": seed, "parameters": parameter_count(model),
                   "hidden": model.hidden, "best_epoch": best_epoch,
                   "comparator_train_accuracy": comparator_train_accuracy,
                   "validation": evaluate(model, validation, device), "heldout": heldout,
                   "history": history}
            summary.append(row)
            stem = f"learned_relation_{kind}_seed{seed}"
            (args.output_dir / f"{stem}.json").write_text(
                json.dumps(row, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            torch.save({"model_state": model.state_dict(), "result": row},
                       args.output_dir / f"{stem}.pt")
            print(json.dumps({"kind": kind, "seed": seed, "best_epoch": best_epoch,
                              "heldout": heldout}), flush=True)
    report = {"method": "frozen visual detector, learned recurrent cue memory and bilinear comparison",
              "train_sizes": [7, 11, 13], "heldout_sizes": [9, 15, 19],
              "train_base_per_size": args.train_base_per_size,
              "validation_base_per_size": args.validation_base_per_size,
              "final_base_per_size": args.final_base_per_size,
              "detector_checkpoint": str(args.detector_checkpoint),
              "seeds": args.seeds, "models": args.models, "epochs": args.epochs,
              "batch_size": args.batch_size, "learning_rate": args.learning_rate,
              "duration_seconds": time.perf_counter() - started, "runs": summary}
    (args.output_dir / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                                  encoding="utf-8")


if __name__ == "__main__":
    main()

"""Candidate-conditioned opponent threat-map supervision."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .history_aux import burn_in
from .multistep_tactics import winning_cells, safe_forks


def generate(path: Path, sources, seed: int, max_positions: int = 24000):
    rng = np.random.default_rng(seed)
    examples = []
    seen = set()
    for source in sources:
        if not Path(source).exists():
            continue
        with np.load(source) as saved:
            obs = saved["obs"].copy()
            labels = saved["labels"].copy()
            histories = saved["histories"].copy()
        order = rng.permutation(len(obs))
        for i in order:
            color = 1 if obs[i, 3, 0, 0] else -1
            cells = ((obs[i, 0] - obs[i, 1]) * color).reshape(-1).astype(np.int8)
            legal = np.flatnonzero(cells == 0)
            preferred = np.flatnonzero(labels[i] & (cells == 0))
            candidates = []
            if len(preferred):
                candidates.append(int(rng.choice(preferred)))
            alternatives = legal[~np.isin(legal, candidates)]
            if len(alternatives):
                candidates.append(int(rng.choice(alternatives)))
            for action in candidates:
                key = (cells.tobytes(), int(color), action)
                if key in seen:
                    continue
                seen.add(key)
                after = cells.copy()
                after[action] = color
                target = np.stack((
                    winning_cells(after, -color, 12),
                    safe_forks(after, -color, 12),
                )).astype(bool)
                examples.append((obs[i].astype(np.int8), histories[i], action, target))
                if len(examples) >= max_positions:
                    break
            if len(examples) >= max_positions:
                break
        if len(examples) >= max_positions:
            break
    if not examples:
        raise RuntimeError("No threat examples were generated")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        obs=np.array([x[0] for x in examples]),
        histories=np.array([x[1] for x in examples]),
        actions=np.array([x[2] for x in examples], dtype=np.int16),
        targets=np.array([x[3] for x in examples]),
        size=12,
        seed=seed,
    )
    return {
        "examples": len(examples),
        "immediate_positive": int(sum(x[3][0].sum() for x in examples)),
        "multiple_positive": int(sum(x[3][1].sum() for x in examples)),
    }


class ThreatData:
    def __init__(self, path: Path, device):
        with np.load(path) as saved:
            self.obs = torch.from_numpy(saved["obs"].astype(np.float32)).to(device)
            self.histories = torch.from_numpy(saved["histories"].astype(np.int64)).to(device)
            self.actions = torch.from_numpy(saved["actions"].astype(np.int64)).to(device)
            self.targets = torch.from_numpy(saved["targets"].astype(np.float32)).to(device)
        presence = self.targets.bool().any(2)
        immediate = torch.nonzero(presence[:, 0], as_tuple=False).flatten()
        multiple = torch.nonzero(presence[:, 1], as_tuple=False).flatten()
        quiet = torch.nonzero(~presence.any(1), as_tuple=False).flatten()
        self.groups = [group for group in (immediate, multiple, quiet) if len(group)]
        self.frames_used = 0
        self.samples = 0

    def loss(self, model, batch):
        group_ids = torch.randint(len(self.groups), (batch,), device=self.obs.device)
        ids = torch.empty(batch, dtype=torch.long, device=self.obs.device)
        for group_index, candidates in enumerate(self.groups):
            selected = group_ids == group_index
            count = int(selected.sum())
            ids[selected] = candidates[torch.randint(len(candidates), (count,), device=self.obs.device)]
        history = self.histories[ids]
        self.frames_used += int(((history >= 0).sum(1) // 2).sum())
        self.samples += batch
        state = burn_in(model, history)
        logits, _ = model.threat(
            self.obs[ids], state,
            torch.zeros(batch, dtype=torch.bool, device=self.obs.device),
            self.actions[ids],
        )
        targets = self.targets[ids].reshape(batch, 2, model.size, model.size)
        flat = targets.flatten(2)
        positives = flat.sum(2)
        negatives = flat.shape[2] - positives
        weights = torch.where(
            positives > 0,
            (negatives / positives.clamp(min=1)).clamp(max=32),
            torch.ones_like(positives),
        )
        raw = F.binary_cross_entropy_with_logits(logits, targets, reduction="none").flatten(2)
        weighted = raw * torch.where(flat > 0, weights[:, :, None], 1.0)
        per_map = weighted.mean(2)
        # Quiet maps remain useful negative evidence, but cannot dominate the
        # sparse maps that contain an actual opponent threat.
        per_map = torch.where(positives > 0, per_map, 0.25 * per_map)
        return per_map.mean()

    @torch.no_grad()
    def evaluate(self, model, batch=128):
        true_positive = np.zeros(2, dtype=np.int64)
        predicted = np.zeros(2, dtype=np.int64)
        positive = np.zeros(2, dtype=np.int64)
        exact = np.zeros(2, dtype=np.int64)
        top_hits = np.zeros(2, dtype=np.int64)
        positive_probability = np.zeros(2, dtype=np.float64)
        negative_probability = np.zeros(2, dtype=np.float64)
        negative = np.zeros(2, dtype=np.int64)
        total = 0
        for start in range(0, len(self.obs), batch):
            stop = min(start + batch, len(self.obs))
            history = self.histories[start:stop]
            state = burn_in(model, history)
            logits, _ = model.threat(
                self.obs[start:stop], state,
                torch.zeros(stop - start, dtype=torch.bool, device=self.obs.device),
                self.actions[start:stop],
            )
            pred = logits.sigmoid() >= 0.5
            target = self.targets[start:stop].reshape(stop - start, 2, model.size, model.size).bool()
            true_positive += (pred & target).sum((0, 2, 3)).cpu().numpy()
            predicted += pred.sum((0, 2, 3)).cpu().numpy()
            positive += target.sum((0, 2, 3)).cpu().numpy()
            exact += (pred == target).flatten(2).all(2).sum(0).cpu().numpy()
            probabilities = logits.sigmoid().flatten(2)
            target_flat = target.flatten(2)
            for channel in range(2):
                for row, count in enumerate(target_flat[:, channel].sum(1).tolist()):
                    if count:
                        order = probabilities[row, channel].argsort(descending=True)[:count]
                        top_hits[channel] += int(target_flat[row, channel, order].sum())
                positive_probability[channel] += float(
                    probabilities[:, channel][target_flat[:, channel]].sum())
                negative_probability[channel] += float(
                    probabilities[:, channel][~target_flat[:, channel]].sum())
                negative[channel] += int((~target_flat[:, channel]).sum())
            total += stop - start
        precision = true_positive / np.maximum(predicted, 1)
        recall = true_positive / np.maximum(positive, 1)
        return {
            name: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "exact_maps": int(exact[i]),
                "examples": total,
                "positive_cells": int(positive[i]),
                "top_k_recall": float(top_hits[i] / max(positive[i], 1)),
                "positive_mean_probability": float(positive_probability[i] / max(positive[i], 1)),
                "negative_mean_probability": float(negative_probability[i] / max(negative[i], 1)),
            }
            for i, name in enumerate(("immediate_win", "multiple_threat"))
        }

def main():
    import argparse
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sources", type=Path, nargs="+", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--max-positions", type=int, default=24000)
    args = parser.parse_args()
    print(json.dumps(generate(args.output, args.sources, args.seed, args.max_positions)))


if __name__ == "__main__":
    main()




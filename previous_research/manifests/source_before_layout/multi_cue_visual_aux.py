"""Paired base/aux room-corridor experiment with a learned forward/stop policy."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from grid_multi_cue import (FEATURES, CueCorridorEnv, GridSpec, build_dataset,
                            make_layout, one_hot_cells)
from multi_cue_memory import COLORS, KINDS, MemoryModel, parameter_count

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "multi_cue_visual_aux"
RATES = {"fly": 0.0003, "rewired": 0.0003, "rnn": 0.0003,
         "gru": 0.003, "leaky": 0.0003}
KINDS_TO_RUN = ("fly", "rewired", "rnn", "gru", "leaky")


def dataset(name, count, delays, seed, wait=0.0):
    path = OUT / "cache" / f"{name}.pt"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(build_dataset([GridSpec(2, delay, 2, wait) for delay in delays],
                                 count, seed), path)
    return torch.load(path, weights_only=True)


def matched_hidden(kind, target):
    if kind in ("fly", "rewired", "leaky"):
        return 1000
    def size(n):
        model = MemoryModel(kind, n, input_dim=FEATURES)
        return parameter_count(model)
    low, high = 1, 512
    while size(high) < target:
        high *= 2
    while low < high:
        mid = (low + high) // 2
        if size(mid) < target:
            low = mid + 1
        else:
            high = mid
    return min((low - 1, low), key=lambda n: abs(size(n) - target))


class VisualAuxModel(nn.Module):
    def __init__(self, kind, hidden, seed, aux):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(20, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.Flatten(), nn.Linear(32 * 49, FEATURES), nn.Tanh())
        self.memory = MemoryModel(kind, hidden, seed % 5, input_dim=FEATURES)
        self.stop_head = nn.Linear(FEATURES, 1)
        self.aux_head = nn.Linear(self.memory.hidden, COLORS * KINDS) if aux else None

    def forward(self, images, lengths):
        batch, steps = images.shape[:2]
        features = self.encoder(one_hot_cells(images.flatten(0, 1))).view(batch, steps, -1)
        logits, state = self.memory(features, lengths, return_state=True)
        aux = (self.aux_head(state).view(batch, COLORS, KINDS)
               if self.aux_head is not None else None)
        stop = self.stop_head(features).squeeze(-1)
        return logits, aux, stop

    @torch.no_grad()
    def stop_probability(self, image, device):
        x = torch.as_tensor(image, device=device).unsqueeze(0)
        features = self.encoder(one_hot_cells(x))
        return float(self.stop_head(features).sigmoid().item())


def batch_loss(model, data, ids, device):
    images = data["images"][ids].to(device)
    lengths = data["lengths"][ids].to(device)
    logits, aux, _ = model(images, lengths)
    loss = F.cross_entropy(logits, data["answers"][ids].to(device))
    if aux is not None:
        labels = data["cue_kinds"][ids].to(device)
        present = data["cue_present"][ids].to(device)
        per = F.cross_entropy(aux.flatten(0, 1), labels.flatten(), reduction="none")
        loss = loss + (per * present.flatten()).sum() / present.sum()
    return loss


@torch.no_grad()
def stop_features(model, data, device, episodes=2048):
    """Frozen visual features and forward/stop labels from teacher paths."""
    features, labels = [], []
    for start in range(0, min(episodes, len(data["answers"])), 64):
        images = data["images"][start:start + 64].to(device)
        lengths = data["lengths"][start:start + 64].to(device)
        batch, steps = images.shape[:2]
        encoded = model.encoder(one_hot_cells(images.flatten(0, 1))).view(batch, steps, -1)
        positions = torch.arange(steps, device=device).unsqueeze(0)
        valid = positions < lengths.unsqueeze(1)
        target = positions == lengths.unsqueeze(1) - 1
        features.append(encoded[valid])
        labels.append(target[valid].float())
    return torch.cat(features), torch.cat(labels)


def fit_stop_head(model, train, device):
    features, labels = stop_features(model, train, device)
    opt = torch.optim.AdamW(model.stop_head.parameters(), lr=0.01)
    generator = torch.Generator().manual_seed(1777)
    for _ in range(400):
        ids = torch.randint(len(labels), (512,), generator=generator).to(device)
        logits = model.stop_head(features[ids]).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(
            logits, labels[ids], pos_weight=torch.tensor(8., device=device))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()


@torch.no_grad()
def fixed_path_eval(model, data, device):
    model.eval()
    correct = 0
    aux_correct = aux_total = 0
    count = len(data["answers"])
    by_delay = {}
    for start in range(0, count, 128):
        ids = slice(start, start + 128)
        logits, aux, _ = model(data["images"][ids].to(device),
                               data["lengths"][ids].to(device))
        correct += int((logits.argmax(-1).cpu() == data["answers"][ids]).sum())
        if aux is not None:
            present = data["cue_present"][ids]
            aux_correct += int(((aux.argmax(-1).cpu() == data["cue_kinds"][ids])
                                & present).sum())
            aux_total += int(present.sum())
    for delay in sorted(set(data["delay"].tolist())):
        mask = data["delay"] == delay
        subset = {k: v[mask] for k, v in data.items()}
        total = len(subset["answers"])
        c = 0
        for start in range(0, total, 128):
            ids = slice(start, start + 128)
            logits = model(subset["images"][ids].to(device),
                           subset["lengths"][ids].to(device))[0]
            c += int((logits.argmax(-1).cpu() == subset["answers"][ids]).sum())
        by_delay[str(delay)] = c / total
    result = {"answer_accuracy": correct / count, "by_delay": by_delay, "count": count}
    if aux_total:
        result["aux_cue_accuracy"] = aux_correct / aux_total
    return result


@torch.no_grad()
def autonomous_eval(model, device, *, count=256, seed=73):
    model.eval()
    rng = np.random.default_rng(seed)
    reached = answered = answered_if_reached = 0
    for episode in range(count):
        delay = (8, 14)[episode % 2]
        layout = make_layout(GridSpec(2, delay, 2), rng)
        env = CueCorridorEnv(layout)
        try:
            obs, _ = env.reset(seed=seed * 1_000_003 + episode)
            frames = [obs["image"]]
            stopped = False
            for _ in range(layout["width"] + 5):
                if model.stop_probability(obs["image"], device) >= 0.5:
                    stopped = True
                    break
                obs, *_ = env.step(2)  # learned stop detector; otherwise forward
                frames.append(obs["image"])
            success = stopped and int(env.agent_pos[0]) == layout["width"] - 2
            reached += int(success)
            if success:
                images = torch.as_tensor(np.stack(frames), device=device).unsqueeze(0)
                lengths = torch.tensor([len(frames)], device=device)
                pred = int(model(images, lengths)[0].argmax(-1).item())
                answered_if_reached += int(pred == layout["answer"])
                answered += int(pred == layout["answer"])
        finally:
            env.close()
    return {"movement_success": reached / count,
            "answer_given_movement_success": answered_if_reached / reached if reached else None,
            "end_to_end_answer_success": answered / count, "episodes": count}


def fit(kind, seed, aux, train, val, tests, device, epochs=20):
    target = parameter_count(MemoryModel("fly", 1000, input_dim=FEATURES))
    hidden = matched_hidden(kind, target)
    torch.manual_seed(seed)
    model = VisualAuxModel(kind, hidden, seed, aux).to(device)
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=RATES[kind])
    generator = torch.Generator().manual_seed(seed + 7)
    best_score, best_state, best_epoch = -1., None, 0
    curve = []
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        for ids in torch.randperm(len(train["answers"]), generator=generator).split(128):
            loss = batch_loss(model, train, ids, device)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        score = fixed_path_eval(model, val, device)["answer_accuracy"]
        curve.append({"epoch": epoch, "validation": score})
        if score > best_score:
            best_score, best_epoch = score, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch % 5 == 0:
            print(json.dumps({"kind": kind, "seed": seed, "aux": aux,
                              "epoch": epoch, "validation": score}), flush=True)
    model.load_state_dict(best_state)
    fit_stop_head(model, train, device)
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    results = {name: fixed_path_eval(model, data, device) for name, data in tests.items()}
    results["autonomous_unseen"] = autonomous_eval(model, device, seed=80_000 + seed)
    path = OUT / "runs" / f"{kind}_{'aux' if aux else 'base'}_seed{seed}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": best_state, "kind": kind, "seed": seed, "aux": aux,
                "hidden": hidden}, path)
    result = {"kind": kind, "seed": seed, "aux": aux, "hidden": hidden,
              "learning_rate": RATES[kind], "best_epoch": best_epoch,
              "best_validation": best_score, "curve": curve, "tests": results,
              "seconds": time.perf_counter() - start, "checkpoint": str(path)}
    path.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"kind": kind, "seed": seed, "aux": aux,
                      "tests": {k: v.get("answer_accuracy", v.get("movement_success"))
                                for k, v in results.items()}}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=KINDS_TO_RUN, default=list(KINDS_TO_RUN))
    parser.add_argument("--seeds", nargs="+", type=int, default=[1101, 1102, 1103])
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    torch.set_num_threads(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train = dataset("train", 8192, (6, 10), 61)
    val = dataset("val", 1024, (6, 10), 62)
    tests = {"trained_lengths": dataset("test", 2048, (6, 10), 63),
             "unseen_lengths": dataset("unseen", 2048, (8, 14), 64),
             "unseen_wait": dataset("wait", 2048, (8, 14), 65, 0.5)}
    for seed in args.seeds:
        for kind in args.models:
            for aux in (False, True):
                result_path = OUT / "runs" / f"{kind}_{'aux' if aux else 'base'}_seed{seed}.json"
                if not result_path.exists():
                    fit(kind, seed, aux, train, val, tests, device, args.epochs)


if __name__ == "__main__":
    main()

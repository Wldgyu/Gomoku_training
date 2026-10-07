"""Frozen-state and counterfactual-query diagnostics for the aux replay."""

from __future__ import annotations

import argparse
import json

import torch
from torch import nn
from torch.nn import functional as F

from multi_cue_aux_transfer import OUT, TRAIN, load_model
from multi_cue_binding import CUE_ROLE, cue_targets
from multi_cue_memory import COLORS, KINDS, make_batch


@torch.no_grad()
def states(model, inputs):
    state = inputs.new_zeros(len(inputs), model.hidden)
    matrix = model.matrix() if model.kind in ("fly", "rewired") else None
    picked = {}
    for step in range(inputs.shape[1] - 1):
        x = inputs[:, step]
        if model.kind in ("fly", "rewired", "leaky"):
            drive = model.input_layer(x)
            if matrix is not None:
                drive = drive + F.linear(state, matrix)
            state = 0.9 * state + 0.1 * drive.tanh()
        else:
            state = model.core(x, state)
        if step == 1:
            picked["after_cues"] = state.clone()
    picked["before_query"] = state
    return picked


def collect(model, data, device):
    chunks = {"after_cues": [], "before_query": []}
    for start in range(0, len(data["answers"]), 256):
        part = data["inputs"][start:start + 256].to(device)
        for name, value in states(model, part).items():
            chunks[name].append(value.detach())
    return {name: torch.cat(values) for name, values in chunks.items()}


def fit_probe(features, targets, present, hidden, device):
    # Probe weights are fitted on independent examples; the memory stays frozen.
    torch.manual_seed(9841)
    probe = nn.Linear(hidden, COLORS * KINDS).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=0.01, weight_decay=0.01)
    generator = torch.Generator().manual_seed(9841)
    for _ in range(250):
        ids = torch.randint(len(features), (256,), generator=generator).to(device)
        logits = probe(features[ids]).view(-1, COLORS, KINDS)
        loss = F.cross_entropy(logits.flatten(0, 1), targets[ids].flatten(),
                               reduction="none").view(-1, COLORS)
        loss = (loss * present[ids]).sum() / present[ids].sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return probe


@torch.no_grad()
def probe_score(probe, features, targets, present):
    pred = probe(features).view(-1, COLORS, KINDS).argmax(-1)
    return float(((pred == targets) & present).sum() / present.sum())


@torch.no_grad()
def query_swap(model, data, device):
    inputs = data["inputs"].to(device)
    cue_rows = inputs[:, :, CUE_ROLE]
    # Use chronological cue order, independent of the original random query.
    slots = cue_rows.nonzero(as_tuple=False).view(len(inputs), 2, 2)[:, :, 1]
    colors = inputs.gather(1, slots.unsqueeze(-1).expand(-1, -1, inputs.shape[-1]))[
        :, :, :COLORS].argmax(-1)
    kinds = inputs.gather(1, slots.unsqueeze(-1).expand(-1, -1, inputs.shape[-1]))[
        :, :, COLORS:COLORS + KINDS].argmax(-1)
    predictions = []
    for slot in range(2):
        swapped = inputs.clone()
        swapped[:, -1, :COLORS] = F.one_hot(colors[:, slot], COLORS).float()
        outputs = []
        for start in range(0, len(swapped), 256):
            outputs.append(model(swapped[start:start + 256])[0].argmax(-1))
        predictions.append(torch.cat(outputs))
    p0, p1 = predictions
    different = kinds[:, 0] != kinds[:, 1]
    return {"both_correct": float(((p0 == kinds[:, 0]) & (p1 == kinds[:, 1])).float().mean()),
            "slot0_correct": float((p0 == kinds[:, 0]).float().mean()),
            "slot1_correct": float((p1 == kinds[:, 1]).float().mean()),
            "different_kind_fraction": float(different.float().mean()),
            "changes_when_kind_differs": float((p0[different] != p1[different]).float().mean())}


def diagnose(kind: str, seed: int, device: torch.device):
    model, meta = load_model(OUT / "checkpoints" / f"{kind}_aux_seed{seed}.pt", device)
    train = make_batch(TRAIN, 3072, 5_000_000_000 + seed)
    test = make_batch(TRAIN, 4096, 6_000_000_000 + seed)
    train_states, test_states = collect(model, train, device), collect(model, test, device)
    target_train, present_train = cue_targets(train["inputs"])
    target_test, present_test = cue_targets(test["inputs"])
    target_train, present_train = target_train.to(device), present_train.to(device)
    target_test, present_test = target_test.to(device), present_test.to(device)
    scores = {}
    for phase in ("after_cues", "before_query"):
        probe = fit_probe(train_states[phase], target_train, present_train, model.hidden, device)
        scores[phase] = {
            "linear_probe": probe_score(probe, test_states[phase], target_test, present_test),
            "trained_aux_head": probe_score(model.aux_head, test_states[phase],
                                             target_test, present_test),
        }
        if phase == "after_cues":
            scores[phase]["same_probe_after_delay"] = probe_score(
                probe, test_states["before_query"], target_test, present_test)
    scores["query_swap"] = query_swap(model, test, device)
    with torch.no_grad():
        aux_predictions = []
        direct_predictions = []
        for start in range(0, len(test["answers"]), 256):
            inputs = test["inputs"][start:start + 256].to(device)
            direct, aux = model(inputs)
            direct_predictions.append(direct.argmax(-1).cpu())
            query = inputs[:, -1, :COLORS].argmax(-1)
            aux_predictions.append(aux[torch.arange(len(query), device=device), query]
                                   .argmax(-1).cpu())
        scores["final_readout"] = {
            "trained_answer_head": float((torch.cat(direct_predictions)
                                          == test["answers"]).float().mean()),
            "aux_with_explicit_query_selection": float((torch.cat(aux_predictions)
                                                        == test["answers"]).float().mean()),
        }
    result = {"kind": kind, "seed": seed, "replayed_test": meta["replayed_test"],
              "diagnostics": scores}
    out = OUT / "diagnostic" / f"{kind}_seed{seed}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["fly", "rewired", "rnn", "gru", "leaky"])
    parser.add_argument("--seed", type=int, default=3104)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for kind in args.models:
        diagnose(kind, args.seed, device)


if __name__ == "__main__":
    main()

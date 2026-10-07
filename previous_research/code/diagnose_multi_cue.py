"""Frozen/learned-edge interventions and independent memory-state readout probes."""

from __future__ import annotations

import json

import torch
from torch import nn
from torch.nn import functional as F

from multi_cue_followup import (
    OUT,
    SEEDS,
    Run,
    atomic_json,
    atomic_torch,
    batched_forward,
    bundle,
    load_model,
    metadata,
    metrics,
    model_for,
    predict,
)
from multi_cue_memory import COLORS, KINDS, TaskConfig, make_batch


@torch.no_grad()
def query_switch(model, data, config, device):
    """Exploratory paired query intervention; the preceding stream stays unchanged."""
    if config.cues_max != 2:
        raise ValueError("two cues required")
    colors, kinds, slots = metadata(data, config)
    rows = torch.arange(len(slots))
    other = 1 - slots
    original = predict(model, data["inputs"], device)
    edited = data["inputs"].clone()
    edited[:, -1, :COLORS] = 0
    edited[rows, -1, colors[rows, other]] = 1
    changed = predict(model, edited, device)
    distinct = kinds[:, 0].ne(kinds[:, 1])
    return {
        "changed_query_accuracy": float(changed.eq(kinds[rows, other]).float().mean()),
        "both_queries_correct": float(
            (original.eq(data["answers"]) & changed.eq(kinds[rows, other]))
            .float()
            .mean()
        ),
        "prediction_changes_when_kinds_differ": float(
            changed[distinct].ne(original[distinct]).float().mean()
        ),
        "distinct_kind_count": int(distinct.sum()),
    }


@torch.no_grad()
def state_data(model, config, count, seed, device):
    data = make_batch(config, count, seed)
    colors, kinds, _ = metadata(data, config)
    targets = torch.full((count, COLORS), -1, dtype=torch.long)
    targets.scatter_(1, colors, kinds)
    p, b = bundle([model])
    phases = {
        "cue_end": config.cues_max - 1,
        "delay_end": config.cues_max + config.delay - 1,
        "query_end": config.length - 1,
    }
    states = {phase: [] for phase in phases}
    for x in data["inputs"].split(512):
        _, trace = batched_forward(p, b, model.kind, x.to(device)[None], trace=True)
        for phase, t in phases.items():
            states[phase].append(trace[0, :, t].cpu())
    return {
        "states": {k: torch.cat(v) for k, v in states.items()},
        "targets": targets,
        "colors": colors,
    }


def probe_one(train, val, test, phase, kind, seed, device):
    torch.manual_seed(seed)
    trainx = train["states"][phase].to(device)
    mean = trainx.mean(0)
    scale = trainx.std(0).clamp(min=1e-4)
    trainx = (trainx - mean) / scale
    valx = (val["states"][phase].to(device) - mean) / scale
    testx = (test["states"][phase].to(device) - mean) / scale
    y = train["targets"].to(device)
    vy = val["targets"].to(device)
    hidden = trainx.shape[1]
    probe = (
        nn.Linear(hidden, COLORS * KINDS)
        if kind == "linear"
        else nn.Sequential(
            nn.Linear(hidden, 128), nn.ReLU(), nn.Linear(128, COLORS * KINDS)
        )
    ).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=0.003)
    generator = torch.Generator(device=device).manual_seed(seed + 1)
    best = float("inf")
    best_state = None
    for update in range(1, 501):
        ids = torch.randint(len(y), (256,), generator=generator, device=device)
        logits = probe(trainx[ids]).view(-1, COLORS, KINDS)
        loss = F.cross_entropy(logits.flatten(0, 1), y[ids].flatten(), ignore_index=-1)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if update % 50 == 0:
            with torch.no_grad():
                validation = float(
                    F.cross_entropy(
                        probe(valx).view(-1, KINDS), vy.flatten(), ignore_index=-1
                    )
                )
            if validation < best:
                best = validation
                best_state = {
                    n: v.detach().clone() for n, v in probe.state_dict().items()
                }
    probe.load_state_dict(best_state)
    with torch.no_grad():
        pred = probe(testx).view(-1, COLORS, KINDS).argmax(-1).cpu()
    targets = test["targets"]
    correct = pred.eq(targets)
    mask = targets.ge(0)
    result = {
        "accuracy_all_present_colors": float(correct[mask].float().mean()),
        "by_cue_slot": {
            str(slot): float(
                correct.gather(1, test["colors"][:, slot, None]).float().mean()
            )
            for slot in range(test["colors"].shape[1])
        },
        "best_validation_loss": best,
        "phase": phase,
        "probe": kind,
        "updates": 500,
        "normalization_std_mean": float(scale.mean()),
        "normalization_min_std": 1e-4,
    }
    return result, {
        "probe_state": {n: v.detach().cpu() for n, v in probe.state_dict().items()},
        "mean": mean.cpu(),
        "scale": scale.cpu(),
        "result": result,
    }


def main():
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda")
    config = TaskConfig(2, 2, 4, 2)
    out = OUT / "diagnosis"
    for seed in SEEDS:
        for kind, frozen in (
            ("fly", False),
            ("fly", True),
            ("leaky", False),
            ("rnn", False),
            ("gru", False),
        ):
            run = Run(kind, seed, frozen=frozen)
            path = out / f"{run.name}.json"
            if path.exists():
                continue
            folder = OUT / ("frozen" if frozen else "baseline")
            model, record = load_model(folder / f"{config.name}_{run.name}.pt", device)
            switched_query = query_switch(
                model, make_batch(config, 4096, 3_000_000_000 + seed), config, device
            )
            datasets = [
                state_data(model, config, count, base + seed, device)
                for count, base in (
                    (4096, 4_100_000_000),
                    (1024, 4_200_000_000),
                    (4096, 4_300_000_000),
                )
            ]
            probes = []
            for phase in ("cue_end", "delay_end", "query_end"):
                for probe_kind in ("linear", "mlp"):
                    result, checkpoint = probe_one(
                        *datasets, phase, probe_kind, seed + 20000, device
                    )
                    probes.append(result)
                    atomic_torch(
                        out / f"{run.name}_{phase}_{probe_kind}.pt", checkpoint
                    )
            interventions = {}
            if kind == "fly" and not frozen:
                original = model.edge_values.detach().clone()
                test = make_batch(config, 4096, 3_000_000_000 + seed)
                with torch.no_grad():
                    model.edge_values.zero_()
                interventions["zero_edges_at_inference"] = metrics(
                    model, test, config, device
                )
                initial = model_for(run, device)
                with torch.no_grad():
                    model.edge_values.copy_(initial.edge_values)
                interventions["initial_edges_at_inference"] = metrics(
                    model, test, config, device
                )
                interventions["learned_edge_delta_rms"] = float(
                    (original - initial.edge_values).square().mean().sqrt()
                )
                with torch.no_grad():
                    model.edge_values.copy_(original)
            atomic_json(
                path,
                {
                    "run": record,
                    "probes": probes,
                    "query_switch": switched_query,
                    "interventions": interventions,
                    "probe_seed_ranges": [
                        4_100_000_000 + seed,
                        4_200_000_000 + seed,
                        4_300_000_000 + seed,
                    ],
                },
            )
            print(
                json.dumps(
                    {
                        "event": "diagnosis",
                        "run": run.name,
                        "probes": [
                            {
                                k: p[k]
                                for k in (
                                    "phase",
                                    "probe",
                                    "accuracy_all_present_colors",
                                    "by_cue_slot",
                                )
                            }
                            for p in probes
                        ],
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()

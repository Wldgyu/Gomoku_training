"""Prespecified multi-cue replication, independent batched training, and load grid."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from multi_cue_memory import (
    COLORS,
    INPUT_DIM,
    KINDS,
    ROOT,
    MemoryModel,
    TaskConfig,
    make_batch,
    parameter_count,
    updates_to_reach,
)

OUT = ROOT / "data" / "multi_cue_followup"
SEEDS = (1101, 1102, 1103)
RATES = {"fly": 0.003, "rewired": 0.003, "leaky": 0.003, "rnn": 0.0003, "gru": 0.01}
HIDDEN = {"fly": 1000, "rewired": 1000, "leaky": 1000, "rnn": 407, "gru": 246}
CONDITIONS = (
    (1, 8, 2),
    (2, 8, 2),
    (3, 8, 2),
    (4, 8, 2),
    (2, 4, 2),
    (2, 16, 2),
    (2, 8, 0),
    (2, 8, 6),
    (1, 32, 6),
)
ROLE = COLORS + KINDS


@dataclass(frozen=True)
class Run:
    kind: str
    seed: int
    graph: int = 0
    frozen: bool = False

    @property
    def name(self):
        return f"{self.kind}_graph{self.graph}_seed{self.seed}" + (
            "_frozen" if self.frozen else ""
        )


def make_fast(config, count, seed):
    """Same fixed-cue distribution as make_batch, different RNG draw order (v2)."""
    if config.cues_min != config.cues_max:
        raise ValueError("fixed cue count required")
    rng = np.random.default_rng(seed)
    n, d = config.cues_max, config.delay
    x = np.zeros((count, config.length, INPUT_DIM), dtype=np.float32)
    rows = np.arange(count)
    colors = rng.random((count, COLORS)).argsort(axis=1)[:, :n]
    kinds = rng.integers(KINDS, size=(count, n))
    for slot in range(n):
        x[rows, slot, colors[:, slot]] = 1
        x[rows, slot, COLORS + kinds[:, slot]] = 1
        x[:, slot, ROLE] = 1
    x[:, n : n + d, ROLE + 2] = 1
    positions = rng.random((count, d)).argsort(axis=1)[:, : config.distractors]
    for slot in range(config.distractors):
        p = n + positions[:, slot]
        x[rows, p, ROLE + 2] = 0
        x[rows, p, ROLE + 1] = 1
        x[rows, p, rng.integers(COLORS, size=count)] = 1
        x[rows, p, COLORS + rng.integers(KINDS, size=count)] = 1
    asked = rng.integers(n, size=count)
    x[rows, -1, colors[rows, asked]] = 1
    x[:, -1, ROLE + 3] = 1
    return {
        "inputs": torch.from_numpy(x),
        "answers": torch.from_numpy(kinds[rows, asked]),
        "query_color": torch.from_numpy(colors[rows, asked]),
        "query_slot": torch.from_numpy(asked),
    }


def model_for(run, device="cpu"):
    torch.manual_seed(run.seed)
    model = MemoryModel(run.kind, HIDDEN[run.kind], run.graph)
    if run.frozen:
        model.edge_values.requires_grad_(False)
    return model.to(device)


def bundle(models):
    params = {
        name: nn.Parameter(
            torch.stack([dict(m.named_parameters())[name].detach() for m in models]),
            requires_grad=p.requires_grad,
        )
        for name, p in models[0].named_parameters()
    }
    buffers = {
        name: torch.stack([dict(m.named_buffers())[name] for m in models])
        for name, _ in models[0].named_buffers()
    }
    return params, buffers


def batched_forward(p, b, kind, x, leak=0.9, trace=False):
    """Leading axis contains independent models, with no cross-model communication."""
    runs, batch, steps, _ = x.shape
    graph = kind in ("fly", "rewired", "leaky")
    hidden = p["head.0.weight"].shape[-1]
    h = x.new_zeros(runs, batch, hidden)
    states = []
    if graph:
        projected = torch.bmm(x.flatten(1, 2), p["input_layer.weight"].transpose(1, 2))
        projected = (
            projected.view(runs, batch, steps, hidden)
            + p["input_layer.bias"][:, None, None]
        )
        matrix = None
        if kind != "leaky":
            matrix = x.new_zeros(runs, hidden, hidden)
            index = torch.arange(runs, device=x.device)[:, None].expand_as(b["pre"])
            matrix = matrix.index_put(
                (index, b["post"], b["pre"]),
                p["edge_values"] * b["edge_scale"],
                accumulate=True,
            )
    else:
        projected = torch.bmm(x.flatten(1, 2), p["core.weight_ih"].transpose(1, 2))
        projected = (
            projected.view(runs, batch, steps, -1) + p["core.bias_ih"][:, None, None]
        )
    for t in range(steps):
        if graph:
            drive = projected[:, :, t]
            if matrix is not None:
                drive = drive + torch.bmm(h, matrix.transpose(1, 2))
            h = leak * h + (1 - leak) * torch.tanh(drive)
        else:
            recurrent = (
                torch.bmm(h, p["core.weight_hh"].transpose(1, 2))
                + p["core.bias_hh"][:, None]
            )
            drive = projected[:, :, t]
            if kind == "rnn":
                h = torch.tanh(drive + recurrent)
            else:
                ir, iz, inn = drive.chunk(3, -1)
                hr, hz, hn = recurrent.chunk(3, -1)
                reset, update = torch.sigmoid(ir + hr), torch.sigmoid(iz + hz)
                new = torch.tanh(inn + reset * hn)
                h = new + update * (h - new)
        if trace:
            states.append(h)
    head = F.relu(
        torch.bmm(h, p["head.0.weight"].transpose(1, 2)) + p["head.0.bias"][:, None]
    )
    logits = (
        torch.bmm(head, p["head.2.weight"].transpose(1, 2)) + p["head.2.bias"][:, None]
    )
    return (logits, torch.stack(states, 2)) if trace else logits


def clip_independent(params, limit=1.0):
    grads = [p.grad for p in params.values() if p.grad is not None]
    norms = torch.stack([g.flatten(1).square().sum(1) for g in grads]).sum(0).sqrt()
    scales = (limit / (norms + 1e-6)).clamp(max=1.0)
    for g in grads:
        g.mul_(scales.view(-1, *([1] * (g.ndim - 1))))


def capture_backward(params, buffers, kind, inputs, answers):
    """Replay the same CUDA forward/backward kernels; AdamW and clipping stay eager."""
    static_x, static_y = inputs.clone(), answers.clone()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            for param in params.values():
                param.grad = None
            logits = batched_forward(params, buffers, kind, static_x)
            (
                F.cross_entropy(logits.flatten(0, 1), static_y.flatten()) * len(inputs)
            ).backward()
            del logits
    torch.cuda.current_stream().wait_stream(stream)
    for param in params.values():
        param.grad = None
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        logits = batched_forward(params, buffers, kind, static_x)
        (
            F.cross_entropy(logits.flatten(0, 1), static_y.flatten()) * len(inputs)
        ).backward()

    def replay(x, y):
        static_x.copy_(x)
        static_y.copy_(y)
        graph.replay()

    return replay


def unpack(model, params, index):
    with torch.no_grad():
        for name, p in model.named_parameters():
            p.copy_(params[name][index])


def metadata(data, config):
    x = data["inputs"]
    colors = x[:, : config.cues_max, :COLORS].argmax(-1)
    kinds = x[:, : config.cues_max, COLORS:ROLE].argmax(-1)
    return colors, kinds, (colors == data["query_color"][:, None]).long().argmax(1)


@torch.no_grad()
def predict(model, x, device, batch=512):
    return torch.cat(
        [model(part.to(device)).argmax(-1).cpu() for part in x.split(batch)]
    )


@torch.no_grad()
def metrics(model, data, config, device, counterfactual=False):
    pred = predict(model, data["inputs"], device)
    _colors, kinds, slots = metadata(data, config)
    correct = pred.eq(data["answers"])
    kind_counts = F.one_hot(kinds, KINDS).sum(1)
    result = {
        "accuracy": float(correct.float().mean()),
        "by_slot": {
            str(i): {
                "accuracy": float(correct[slots == i].float().mean()),
                "count": int((slots == i).sum()),
            }
            for i in range(config.cues_max)
        },
        "last_only_accuracy": float(kinds[:, -1].eq(data["answers"]).float().mean()),
        "majority_kind_accuracy": float(
            kind_counts.argmax(1).eq(data["answers"]).float().mean()
        ),
        "prediction_is_majority_kind": float(
            kind_counts.gather(1, pred[:, None])
            .squeeze(1)
            .eq(kind_counts.max(1).values)
            .float()
            .mean()
        ),
    }
    if counterfactual:
        rows = torch.arange(len(pred))
        edited = data["inputs"].clone()
        changed = (kinds[rows, slots] + 1) % KINDS
        edited[rows, slots, COLORS:ROLE] = 0
        edited[rows, slots, COLORS + changed] = 1
        cf = predict(model, edited, device)
        result["queried_kind_change_accuracy"] = float(cf.eq(changed).float().mean())
        result["queried_kind_change_prediction_rate"] = float(
            cf.ne(pred).float().mean()
        )
        if config.cues_max > 1:
            others = (slots + 1) % config.cues_max
            edited = data["inputs"].clone()
            edited[rows, others, COLORS:ROLE] = 0
            edited[rows, others, COLORS + (kinds[rows, others] + 1) % KINDS] = 1
            cf = predict(model, edited, device)
            result["unqueried_kind_change_accuracy"] = float(
                cf.eq(data["answers"]).float().mean()
            )
            result["unqueried_kind_change_invariance"] = float(
                cf.eq(pred).float().mean()
            )
    return result


def atomic_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temp.replace(path)


def atomic_torch(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    torch.save(data, temp)
    temp.replace(path)


def train_group(
    runs,
    config,
    updates,
    out,
    device,
    *,
    save_at=(),
    batch_size=128,
    validation_size=1024,
    test_size=4096,
    eval_every=500,
    counterfactual=False,
):
    out.mkdir(parents=True, exist_ok=True)
    paths = [out / f"{config.name}_{run.name}.json" for run in runs]
    if all(path.exists() for path in paths):
        for path in paths:
            result = json.loads(path.read_text(encoding="utf-8"))
            if (
                result["updates"] != updates
                or result["generator"] != "vectorized_fixed_v2"
            ):
                raise ValueError(f"incompatible existing result {path}")
        return
    models = [model_for(run, device) for run in runs]
    p, b = bundle(models)
    del models
    kind = runs[0].kind
    if any((r.kind in ("rnn", "gru", "leaky")) and r.kind != kind for r in runs):
        raise ValueError("incompatible bundled model kinds")
    rates = [RATES[r.kind] for r in runs]
    if len(set(rates)) != 1:
        raise ValueError("group learning rates differ")
    optimizer = torch.optim.AdamW(
        [v for v in p.values() if v.requires_grad], lr=rates[0]
    )
    unique = sorted({r.seed for r in runs})
    validation = {
        s: make_batch(config, validation_size, 2_000_000_000 + s) for s in unique
    }
    test = {s: make_batch(config, test_size, 3_000_000_000 + s) for s in unique}
    manifest = {
        "runs": [asdict(r) for r in runs],
        "config": asdict(config),
        "updates": updates,
        "batch_size": batch_size,
        "rates": rates,
        "generator": "vectorized_fixed_v2",
        "validation_size": validation_size,
        "test_size": test_size,
    }
    tag = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()[:16]
    resume = out / f"resume_{tag}.pt"
    curves = [[] for _ in runs]
    first = 0
    if resume.exists():
        saved = torch.load(resume, map_location="cpu", weights_only=True)
        if saved["manifest"] != manifest:
            raise ValueError("resume manifest mismatch")
        with torch.no_grad():
            for name, param in p.items():
                param.copy_(saved["params"][name])
        optimizer.load_state_dict(saved["optimizer"])
        curves, first = saved["curves"], saved["update"]
    start = time.perf_counter()

    def checkpoint(update, final=False):
        vals = []
        for i, run in enumerate(runs):
            m = model_for(run, device)
            unpack(m, p, i)
            score = metrics(m, validation[run.seed], config, device)["accuracy"]
            curves[i].append({"update": update, "validation": score})
            vals.append(score)
            if final or update in save_at:
                result = {
                    **asdict(run),
                    "name": run.name,
                    "config": asdict(config),
                    "task": config.name,
                    "hidden": m.hidden,
                    "parameters": parameter_count(m),
                    "leak": m.leak,
                    "learning_rate": rates[i],
                    "updates": update,
                    "batch_size": batch_size,
                    "episodes_seen": update * batch_size,
                    "generator": "vectorized_fixed_v2",
                    "execution_mode_at_finish": "cuda_graph_replay"
                    if device.type == "cuda"
                    else "eager",
                    "final_validation": score,
                    "curve": curves[i].copy(),
                    "test": metrics(m, test[run.seed], config, device, counterfactual),
                    "to90": updates_to_reach(curves[i], 0.90),
                    "to99": updates_to_reach(curves[i], 0.99),
                    "seconds_this_session": time.perf_counter() - start,
                }
                folder = out if final else out / f"update{update}"
                ckpt = folder / f"{config.name}_{run.name}.pt"
                result["checkpoint"] = str(ckpt)
                atomic_torch(
                    ckpt,
                    {
                        "model_state": {
                            n: v.detach().cpu() for n, v in m.state_dict().items()
                        },
                        "result": result,
                    },
                )
                atomic_json(ckpt.with_suffix(".json"), result)
                print(
                    json.dumps(
                        {
                            "event": "result",
                            "task": config.name,
                            "run": run.name,
                            "update": update,
                            "test": result["test"]["accuracy"],
                            "by_slot": result["test"]["by_slot"],
                        }
                    ),
                    flush=True,
                )
            del m
        atomic_torch(
            resume,
            {
                "manifest": manifest,
                "params": {n: v.detach() for n, v in p.items()},
                "optimizer": optimizer.state_dict(),
                "update": update,
                "curves": curves,
            },
        )
        print(
            json.dumps(
                {
                    "event": "progress",
                    "task": config.name,
                    "group": kind,
                    "runs": len(runs),
                    "update": update,
                    "val": vals,
                    "seconds": round(time.perf_counter() - start, 1),
                }
            ),
            flush=True,
        )

    if not first:
        checkpoint(0)
    replay = None
    for update in range(first + 1, updates + 1):
        batches = {
            s: make_fast(config, batch_size, 1_000_000_000 + s * 1_000_000 + update)
            for s in unique
        }
        x = torch.stack([batches[r.seed]["inputs"] for r in runs]).to(device)
        y = torch.stack([batches[r.seed]["answers"] for r in runs]).to(device)
        if device.type == "cuda":
            if replay is None:
                replay = capture_backward(p, b, kind, x, y)
            replay(x, y)
        else:
            optimizer.zero_grad(set_to_none=True)
            logits = batched_forward(p, b, kind, x)
            loss = F.cross_entropy(logits.flatten(0, 1), y.flatten()) * len(runs)
            loss.backward()
        clip_independent(p)
        optimizer.step()
        if update % eval_every == 0 or update == updates:
            checkpoint(update, update == updates)


def load_model(path, device):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    result = saved["result"]
    run = Run(result["kind"], result["seed"], result["graph"], result["frozen"])
    m = model_for(run, device)
    m.load_state_dict(saved["model_state"])
    return m, result


def baseline(device):
    c = TaskConfig(2, 2, 4, 2)
    graph = [Run("fly", s) for s in SEEDS] + [
        Run("rewired", s, g) for s in SEEDS for g in range(5)
    ]
    train_group(
        graph, c, 16000, OUT / "baseline", device, save_at=(12000,), counterfactual=True
    )
    for k in ("rnn", "gru", "leaky"):
        train_group(
            [Run(k, s) for s in SEEDS],
            c,
            16000,
            OUT / "baseline",
            device,
            save_at=(12000,),
            counterfactual=True,
        )


def frozen(device):
    train_group(
        [Run("fly", s, frozen=True) for s in SEEDS],
        TaskConfig(2, 2, 4, 2),
        16000,
        OUT / "frozen",
        device,
        counterfactual=True,
    )


def grid(device):
    for n, d, k in CONDITIONS:
        c = TaskConfig(n, n, d, k)
        out = OUT / "load"
        if (n, d, k) == (2, 4, 2):
            for s in SEEDS:
                for kind in RATES:
                    r = Run(kind, s, s % 5 if kind == "rewired" else 0)
                    source = (
                        OUT / "baseline" / "update12000" / f"{c.name}_{r.name}.json"
                    )
                    atomic_json(
                        out / source.name,
                        json.loads(source.read_text(encoding="utf-8")),
                    )
            continue
        train_group(
            [Run("fly", s) for s in SEEDS] + [Run("rewired", s, s % 5) for s in SEEDS],
            c,
            12000,
            out,
            device,
        )
        for kind in ("rnn", "gru", "leaky"):
            train_group([Run(kind, s) for s in SEEDS], c, 12000, out, device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("baseline", "frozen", "load"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    {"baseline": baseline, "frozen": frozen, "load": grid}[args.stage](
        torch.device(args.device)
    )


if __name__ == "__main__":
    main()

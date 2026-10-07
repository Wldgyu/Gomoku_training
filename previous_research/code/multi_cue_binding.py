"""Binding and distractor-robustness interventions for the multi-cue memory task.

Three optional, separately switchable additions on top of the earlier MemoryModel dynamics:
  bind: extra input features, the outer product color x kind of the object shown (12 values),
        so a cue's color and kind arrive already conjoined;
  aux:  an auxiliary readout of the final state that must report the kind of EVERY cue color
        shown in the episode (training labels only; evaluation uses the usual queried answer);
  gate: a learned write gate z = sigmoid(W x + b) that scales the state update
        state = (1 - z) * state + z * tanh(input + edges @ state)  (graph-based models only;
        b starts at logit(0.1) so the initial dynamics equal the earlier leak=0.9 update).
All five memories receive the same bind/aux settings; gate exists only for the graph family.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from multi_cue_memory import (
    COLORS,
    GRAPH_DIR,
    INPUT_DIM,
    KINDS,
    MODEL_KINDS,
    TaskConfig,
    make_batch,
    parameter_count,
    updates_to_reach,
)

ROOT = Path(__file__).resolve().parent.parent
CUE_ROLE = COLORS + KINDS  # index of the "cue" role flag
DISTRACTOR_ROLE = CUE_ROLE + 1
GRAPH_KINDS = ("fly", "rewired", "leaky")
VARIANTS = {
    "base": (False, False, False),
    "bind": (True, False, False),
    "aux": (False, False, True),
    "gate": (False, True, False),
    "bind_aux": (True, False, True),
    "gate_aux": (False, True, True),
    "bind_gate_aux": (True, True, True),
}  # name -> (bind, gate, aux)
LEARNING_RATES = {"fly": 0.003, "rewired": 0.003, "leaky": 0.003, "rnn": 0.0003, "gru": 0.01}


def without_gate(variant: str) -> str:
    """Closest variant for models that cannot take a write gate (RNN, GRU)."""
    bind, _, aux = VARIANTS[variant]
    return next(name for name, flags in VARIANTS.items() if flags == (bind, False, aux))


def conjunction(inputs: torch.Tensor) -> torch.Tensor:
    """color x kind outer product of each step, zeros when the step shows no object."""
    color = inputs[..., :COLORS]
    kind = inputs[..., COLORS:COLORS + KINDS]
    return (color.unsqueeze(-1) * kind.unsqueeze(-2)).flatten(-2)


def cue_targets(inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Kind of every cue color (batch, COLORS) and whether that color was shown as a cue."""
    cue = inputs[:, :, CUE_ROLE]
    onehot = torch.einsum("bt,btc,btk->bck", cue, inputs[..., :COLORS],
                          inputs[..., COLORS:COLORS + KINDS])
    return onehot.argmax(-1), onehot.sum(-1) > 0


def episode_descriptors(inputs: torch.Tensor) -> dict[str, torch.Tensor]:
    """Per-episode facts for diagnostics: slot (0 = first cue shown) of the queried cue,
    and whether any distractor reused the queried color."""
    query = inputs[:, -1, :COLORS].argmax(-1)
    cue = inputs[:, :, CUE_ROLE]
    is_query_color = inputs[..., :COLORS].gather(
        2, query.view(-1, 1, 1).expand(-1, inputs.shape[1], 1)).squeeze(-1)
    cue_hit = cue * is_query_color
    order = cue.cumsum(1) - 1  # 0-based cue index at each step
    slot = (cue_hit * order).sum(1).long()
    cues = cue.sum(1).long()
    reused = (inputs[:, :, DISTRACTOR_ROLE] * is_query_color).sum(1) > 0
    return {"slot": slot, "cues": cues, "reused": reused}


class BindMemory(nn.Module):
    def __init__(self, kind: str, hidden: int, variant: str = "base", rewire_seed: int = 0,
                 head: int = 128) -> None:
        super().__init__()
        if kind not in MODEL_KINDS:
            raise ValueError(kind)
        self.kind, self.variant = kind, variant
        self.bind, self.gate, self.aux = VARIANTS[variant]
        if self.gate and kind not in GRAPH_KINDS:
            raise ValueError("write gate is only defined for the graph-based memories")
        input_dim = INPUT_DIM + (COLORS * KINDS if self.bind else 0)
        self.hidden = hidden
        if kind in GRAPH_KINDS:
            suffix = f"_rewired_seed{rewire_seed}" if kind == "rewired" else ""
            with np.load(GRAPH_DIR / f"malecns_cx_1000{suffix}.npz") as graph:
                pre = torch.from_numpy(graph["pre_index"].astype(np.int64))
                post = torch.from_numpy(graph["post_index"].astype(np.int64))
                self.hidden = len(graph["body_ids"])
            degree = torch.bincount(post, minlength=self.hidden).clamp(min=1)
            self.register_buffer("pre", pre)
            self.register_buffer("post", post)
            self.register_buffer("edge_scale", degree[post].float().rsqrt())
            self.edge_values = nn.Parameter(torch.randn(len(pre)) * 0.2,
                                            requires_grad=kind != "leaky")
            if kind == "leaky":
                self.edge_values.data.zero_()
            self.input_layer = nn.Linear(input_dim, self.hidden)
            if self.gate:
                self.gate_layer = nn.Linear(input_dim, self.hidden)
                nn.init.constant_(self.gate_layer.bias, math.log(0.1 / 0.9))
        else:
            cell = nn.RNNCell if kind == "rnn" else nn.GRUCell
            self.core = cell(input_dim, hidden)
            if kind == "rnn":
                with torch.random.fork_rng(devices=[]):
                    nn.init.orthogonal_(self.core.weight_hh)
        self.head = nn.Sequential(nn.Linear(self.hidden, head), nn.ReLU(), nn.Linear(head, KINDS))
        self.aux_head = nn.Linear(self.hidden, COLORS * KINDS) if self.aux else None

    def matrix(self) -> torch.Tensor:
        result = self.edge_values.new_zeros(self.hidden, self.hidden)
        return result.index_put((self.post, self.pre), self.edge_values * self.edge_scale,
                                accumulate=True)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        features = torch.cat((inputs, conjunction(inputs)), dim=-1) if self.bind else inputs
        state = features.new_zeros(len(features), self.hidden)
        graph = self.kind in GRAPH_KINDS
        matrix = self.matrix() if graph else None
        for step in range(features.shape[1]):
            x = features[:, step]
            if graph:
                update = torch.tanh(self.input_layer(x) + F.linear(state, matrix))
                z = torch.sigmoid(self.gate_layer(x)) if self.gate else 0.1
                state = (1 - z) * state + z * update
            else:
                state = self.core(x, state)
        aux = self.aux_head(state).view(-1, COLORS, KINDS) if self.aux else None
        return self.head(state), aux


def count_parameters(kind: str, hidden: int, variant: str) -> int:
    return parameter_count(BindMemory(kind, hidden, variant))


def matched_hidden(kind: str, variant: str, target: int) -> int:
    if kind in GRAPH_KINDS:
        return 1000
    low, high = 1, 512
    while count_parameters(kind, high, variant) < target:
        high *= 2
    while low < high:
        middle = (low + high) // 2
        if count_parameters(kind, middle, variant) < target:
            low = middle + 1
        else:
            high = middle
    return min((low - 1, low), key=lambda n: abs(count_parameters(kind, n, variant) - target))


def loss_for(model: BindMemory, inputs: torch.Tensor, answers: torch.Tensor,
             aux_weight: float = 1.0) -> torch.Tensor:
    logits, aux = model(inputs)
    loss = F.cross_entropy(logits, answers)
    if aux is not None:
        target, present = cue_targets(inputs)
        per_color = F.cross_entropy(aux.flatten(0, 1), target.flatten(), reduction="none")
        loss = loss + aux_weight * (per_color * present.flatten()).sum() / present.sum()
    return loss


@torch.no_grad()
def evaluate(model: BindMemory, data: dict, device: torch.device, detail: bool = False):
    model.eval()
    predictions = []
    for start in range(0, len(data["answers"]), 1024):
        predictions.append(model(data["inputs"][start:start + 1024].to(device))[0]
                           .argmax(-1).cpu())
    correct = torch.cat(predictions).eq(data["answers"])
    accuracy = float(correct.float().mean())
    if not detail:
        return accuracy
    facts = episode_descriptors(data["inputs"])
    result = {"accuracy": accuracy, "by_slot": {}, "distractor_reused_color": {}}
    for slot in range(int(facts["cues"].max())):
        mask = facts["slot"] == slot
        if mask.any():
            result["by_slot"][str(slot)] = float(correct[mask].float().mean())
    for reused in (False, True):
        mask = facts["reused"] == reused
        if mask.any():
            result["distractor_reused_color"][str(reused)] = float(correct[mask].float().mean())
    return result


@dataclass(frozen=True)
class Job:
    stage: str
    kind: str
    variant: str
    seed: int
    cues: int
    delay: int
    distractors: int
    updates: int

    @property
    def name(self) -> str:
        task = f"cues{self.cues}_delay{self.delay}_distract{self.distractors}"
        return f"{task}_{self.kind}_{self.variant}_seed{self.seed}"


def run_job(job: Job, device: torch.device, output: Path, hidden_cache: dict) -> dict:
    config = TaskConfig(job.cues, job.cues, job.delay, job.distractors)
    key = (job.kind, job.variant)
    if key not in hidden_cache:
        target = count_parameters("fly", 1000, job.variant if job.variant in VARIANTS else "base")
        hidden_cache[key] = matched_hidden(job.kind, job.variant, target)
    torch.manual_seed(job.seed)
    rate = LEARNING_RATES[job.kind]
    model = BindMemory(job.kind, hidden_cache[key], job.variant, job.seed % 5).to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=rate)
    # Same seed ranges as the earlier multi-cue runs: training, validation, test never overlap.
    validation = make_batch(config, 1024, 2_000_000_000 + job.seed)
    test = make_batch(config, 4096, 3_000_000_000 + job.seed)
    curve = [{"update": 0, "validation": evaluate(model, validation, device)}]
    started = time.perf_counter()
    for update in range(1, job.updates + 1):
        batch = make_batch(config, 128, 1_000_000_000 + job.seed * 1_000_000 + update)
        loss = loss_for(model, batch["inputs"].to(device), batch["answers"].to(device))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if update % 500 == 0 or update == job.updates:
            curve.append({"update": update, "loss": float(loss.detach()),
                          "validation": evaluate(model, validation, device)})
    result = {**asdict(job), "name": job.name, "hidden": model.hidden,
              "parameters": parameter_count(model), "learning_rate": rate,
              "final_validation": curve[-1]["validation"],
              "test": evaluate(model, test, device, detail=True), "curve": curve,
              "to90": updates_to_reach(curve, 0.9), "seconds": time.perf_counter() - started}
    path = output / f"{job.name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return result


TUNE_CONDITIONS = ((2, 8, 6), (3, 8, 2))
TUNE_SEEDS = (901, 902)
FINAL_CONDITIONS = ((2, 8, 6), (3, 8, 2), (4, 8, 2), (2, 16, 2))
FINAL_SEEDS = (3101, 3102, 3103, 3104, 3105)
TUNE_UPDATES, FINAL_UPDATES = 6000, 8000


def tune_jobs() -> list[Job]:
    jobs = []
    for cues, delay, distractors in TUNE_CONDITIONS:
        for seed in TUNE_SEEDS:
            for kind in ("fly", "leaky", "rnn", "gru"):
                for variant, (_, gate, _) in VARIANTS.items():
                    if gate and kind not in GRAPH_KINDS:
                        continue
                    jobs.append(Job("tune", kind, variant, seed, cues, delay, distractors,
                                    TUNE_UPDATES))
    return jobs


def final_jobs(selected: str) -> list[Job]:
    jobs = []
    for cues, delay, distractors in FINAL_CONDITIONS:
        for seed in FINAL_SEEDS:
            for kind in MODEL_KINDS:
                chosen = selected if kind in GRAPH_KINDS else without_gate(selected)
                for variant in dict.fromkeys(("base", chosen)):
                    jobs.append(Job("final", kind, variant, seed, cues, delay, distractors,
                                    FINAL_UPDATES))
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("tune", "final"))
    parser.add_argument("--worker", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "multi_cue_binding")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.stage == "tune":
        jobs = tune_jobs()
    else:
        selection = json.loads((args.output_dir / "selection.json").read_text(encoding="utf-8"))
        jobs = final_jobs(selection["variant"])
    folder = args.output_dir / args.stage
    hidden_cache: dict = {}
    for job in jobs[args.worker::args.workers]:
        if (folder / f"{job.name}.json").exists():
            continue
        result = run_job(job, device, folder, hidden_cache)
        print(json.dumps({"name": job.name, "val": round(result["final_validation"], 4),
                          "test": round(result["test"]["accuracy"], 4),
                          "to90": result["to90"], "sec": round(result["seconds"])}),
              flush=True)


if __name__ == "__main__":
    main()

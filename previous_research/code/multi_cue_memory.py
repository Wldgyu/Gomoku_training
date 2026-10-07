"""Multi-cue memory with distractors: sequence task, shared-budget memory models, training.

An episode shows 2-4 cues (distinct color, random object kind) in order, then a delay in
which some steps are distractor objects (random color, random kind, marked as distractor)
and the rest are blank. The last step asks for one cue color; the answer is that cue's
object kind. Every model receives the same per-step input and the same readout head, so
only the recurrent memory differs.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent.parent
GRAPH_DIR = ROOT / "data" / "subgraphs"

COLORS = 4  # red, green, blue, yellow
KINDS = 3  # ball, key, box
ROLES = ("cue", "distractor", "blank", "query")
INPUT_DIM = COLORS + KINDS + len(ROLES)
MODEL_KINDS = ("fly", "rewired", "rnn", "gru", "leaky")


@dataclass(frozen=True)
class TaskConfig:
    cues_min: int = 2
    cues_max: int = 2
    delay: int = 4  # steps between the last cue and the query
    distractors: int = 0  # how many of the delay steps are distractors

    def __post_init__(self) -> None:
        if not 1 <= self.cues_min <= self.cues_max <= COLORS:
            raise ValueError("cue count must be within 1..number of colors")
        if not 0 <= self.distractors <= self.delay:
            raise ValueError("distractors must fit inside the delay")

    @property
    def length(self) -> int:
        return self.cues_max + self.delay + 1

    @property
    def name(self) -> str:
        cues = (str(self.cues_min) if self.cues_min == self.cues_max
                else f"{self.cues_min}-{self.cues_max}")
        return f"cues{cues}_delay{self.delay}_distract{self.distractors}"


def make_batch(config: TaskConfig, count: int, seed: int) -> dict[str, torch.Tensor]:
    """Deterministic batch. Fewer cues are left-padded with blanks so the query is last."""
    rng = np.random.default_rng(seed)
    inputs = np.zeros((count, config.length, INPUT_DIM), dtype=np.float32)
    answers = np.zeros(count, dtype=np.int64)
    query_colors = np.zeros(count, dtype=np.int64)
    role_offset = COLORS + KINDS
    blank, query = role_offset + 2, role_offset + 3

    def put(row: int, step: int, role: int, color: int | None, kind: int | None) -> None:
        inputs[row, step, role_offset + role] = 1.0
        if color is not None:
            inputs[row, step, color] = 1.0
        if kind is not None:
            inputs[row, step, COLORS + kind] = 1.0

    for row in range(count):
        cue_count = int(rng.integers(config.cues_min, config.cues_max + 1))
        colors = rng.permutation(COLORS)[:cue_count]
        kinds = rng.integers(0, KINDS, size=cue_count)
        pad = config.cues_max - cue_count
        for step in range(pad):
            inputs[row, step, blank] = 1.0
        for slot in range(cue_count):
            put(row, pad + slot, 0, int(colors[slot]), int(kinds[slot]))
        delay_start = config.cues_max
        positions = set(rng.permutation(config.delay)[:config.distractors].tolist())
        for offset in range(config.delay):
            step = delay_start + offset
            if offset in positions:  # any color, so distractors may reuse a cue color
                put(row, step, 1, int(rng.integers(0, COLORS)), int(rng.integers(0, KINDS)))
            else:
                inputs[row, step, blank] = 1.0
        asked = int(rng.integers(0, cue_count))
        inputs[row, config.length - 1, int(colors[asked])] = 1.0
        inputs[row, config.length - 1, query] = 1.0
        answers[row] = int(kinds[asked])
        query_colors[row] = int(colors[asked])
    return {"inputs": torch.from_numpy(inputs), "answers": torch.from_numpy(answers),
            "query_color": torch.from_numpy(query_colors)}


class MemoryModel(nn.Module):
    """Recurrent memory plus a shared MLP head applied to the final state.

    kind: fly / rewired use the MaleCNS edge mask with the same leaky update as the earlier
    MiniGrid modules; leaky is the same dynamics with all edges fixed at zero.
    """

    def __init__(self, kind: str, hidden: int, rewire_seed: int = 0, head: int = 128,
                 input_dim: int = INPUT_DIM, leak: float = 0.9) -> None:
        super().__init__()
        if kind not in MODEL_KINDS:
            raise ValueError(kind)
        self.kind = kind
        self.hidden = hidden
        self.leak = leak
        if kind in ("fly", "rewired", "leaky"):
            suffix = "_rewired_seed%d" % rewire_seed if kind == "rewired" else ""
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
        else:
            cell = nn.RNNCell if kind == "rnn" else nn.GRUCell
            self.core = cell(input_dim, hidden)
            if kind == "rnn":
                with torch.random.fork_rng(devices=[]):
                    nn.init.orthogonal_(self.core.weight_hh)
        self.head = nn.Sequential(nn.Linear(self.hidden, head), nn.ReLU(), nn.Linear(head, KINDS))

    def matrix(self) -> torch.Tensor:
        result = self.edge_values.new_zeros(self.hidden, self.hidden)
        return result.index_put((self.post, self.pre), self.edge_values * self.edge_scale,
                                accumulate=True)

    def forward(self, inputs: torch.Tensor, lengths: torch.Tensor | None = None,
                return_state: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """inputs: batch × time × features. With lengths, the state at each episode's own
        final step is read out (steps past it are padding)."""
        state = inputs.new_zeros(len(inputs), self.hidden)
        picked = state
        graph_based = self.kind in ("fly", "rewired", "leaky")
        matrix = self.matrix() if graph_based and self.kind != "leaky" else None
        for step in range(inputs.shape[1]):
            x = inputs[:, step]
            if graph_based:
                drive = self.input_layer(x)
                if matrix is not None:
                    drive = drive + F.linear(state, matrix)
                state = self.leak * state + (1 - self.leak) * torch.tanh(drive)
            else:
                state = self.core(x, state)
            if lengths is not None:
                picked = torch.where((lengths - 1 == step).unsqueeze(1), state, picked)
        final_state = state if lengths is None else picked
        logits = self.head(final_state)
        return (logits, final_state) if return_state else logits


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def matched_hidden(kind: str, target: int) -> int:
    """Hidden size whose trainable parameter count is closest to the Fly model's."""
    if kind in ("fly", "rewired", "leaky"):
        return 1000
    low, high = 1, 512
    while parameter_count(MemoryModel(kind, high)) < target:
        high *= 2
    while low < high:
        middle = (low + high) // 2
        if parameter_count(MemoryModel(kind, middle)) < target:
            low = middle + 1
        else:
            high = middle
    return min((low - 1, low), key=lambda n: abs(parameter_count(MemoryModel(kind, n)) - target))


@torch.no_grad()
def accuracy(model: MemoryModel, data: dict[str, torch.Tensor], device: torch.device,
             batch_size: int = 1024) -> float:
    model.eval()
    correct = 0
    for start in range(0, len(data["answers"]), batch_size):
        inputs = data["inputs"][start:start + batch_size].to(device)
        answers = data["answers"][start:start + batch_size].to(device)
        correct += int(model(inputs).argmax(dim=-1).eq(answers).sum())
    return correct / len(data["answers"])


def train_one(kind: str, config: TaskConfig, seed: int, hidden: int, *, updates: int,
              batch_size: int, learning_rate: float, eval_every: int, validation_size: int,
              test_size: int, device: torch.device, rewire_seed: int | None = None,
              leak: float = 0.9) -> dict:
    """One training run: fresh training batches every update, fixed validation/test sets."""
    torch.manual_seed(seed)
    model = MemoryModel(kind, hidden, seed % 5 if rewire_seed is None else rewire_seed,
                        leak=leak).to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=learning_rate)
    # Seed ranges are disjoint: training, validation and test never share episodes.
    validation = make_batch(config, validation_size, 2_000_000_000 + seed)
    test = make_batch(config, test_size, 3_000_000_000 + seed)
    curve = [{"update": 0, "validation": accuracy(model, validation, device)}]
    started = time.perf_counter()
    for update in range(1, updates + 1):
        batch = make_batch(config, batch_size, 1_000_000_000 + seed * 1_000_000 + update)
        loss = F.cross_entropy(model(batch["inputs"].to(device)), batch["answers"].to(device))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if update % eval_every == 0 or update == updates:
            curve.append({"update": update, "loss": float(loss.detach()),
                          "validation": accuracy(model, validation, device)})
    return {"kind": kind, "seed": seed, "config": asdict(config), "task": config.name,
            "hidden": model.hidden, "parameters": parameter_count(model),
            "updates": updates, "batch_size": batch_size, "learning_rate": learning_rate,
            "episodes_seen": updates * batch_size,
            "final_validation": curve[-1]["validation"],
            "test_accuracy": accuracy(model, test, device),
            "chance": 1 / KINDS, "curve": curve,
            "seconds": time.perf_counter() - started}


def updates_to_reach(curve: list[dict], threshold: float) -> int | None:
    """First evaluated update whose validation accuracy reaches the threshold."""
    return next((point["update"] for point in curve if point["validation"] >= threshold), None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=MODEL_KINDS, default=list(MODEL_KINDS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[801, 802, 803])
    parser.add_argument("--cues-min", type=int, default=2)
    parser.add_argument("--cues-max", type=int, default=2)
    parser.add_argument("--delay", type=int, default=4)
    parser.add_argument("--distractors", type=int, default=2)
    parser.add_argument("--updates", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--validation-size", type=int, default=1024)
    parser.add_argument("--test-size", type=int, default=4096)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "multi_cue")
    parser.add_argument("--rates", nargs="*", default=[], metavar="KIND=LR",
                        help="per-model learning rates overriding --learning-rate")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    device = torch.device(args.device)
    rates = {item.split("=")[0]: float(item.split("=")[1]) for item in args.rates}
    config = TaskConfig(args.cues_min, args.cues_max, args.delay, args.distractors)
    torch.manual_seed(0)
    target = parameter_count(MemoryModel("fly", 1000))
    hidden = {kind: matched_hidden(kind, target) for kind in args.models}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        for kind in args.models:
            path = args.output_dir / f"{config.name}_{kind}_seed{seed}.json"
            if args.skip_existing and path.exists():
                continue
            result = train_one(kind, config, seed, hidden[kind], updates=args.updates,
                               batch_size=args.batch_size,
                               learning_rate=rates.get(kind, args.learning_rate),
                               eval_every=args.eval_every, validation_size=args.validation_size,
                               test_size=args.test_size, device=device)
            path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
            print(json.dumps({"task": config.name, "kind": kind, "seed": seed,
                              "hidden": result["hidden"], "parameters": result["parameters"],
                              "test": round(result["test_accuracy"], 4),
                              "to90": updates_to_reach(result["curve"], 0.9)}), flush=True)


if __name__ == "__main__":
    main()

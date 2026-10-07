"""Multi-cue memory in a MiniGrid room and corridor, observed through real movement.

Layout (agent starts at x=1 on the middle row, facing east, and only walks forward):
  * start room (3 rows tall): 2-4 cue objects (distinct color, kind key/ball/box) stand on
    the north row, 3 cells apart, so the agent passes them one after another;
  * corridor (1 cell wide): `distractors` objects of random color and kind stand in side
    alcoves; they may reuse a cue color;
  * at the end a closed door whose COLOR is the query. The answer is the kind of the cue with
    that color; the model reads it out at the agent's final step.

Observation is the 7x7 egocentric partial view only. The walk itself is scripted (it is the
same forward path in every episode, plus random waiting steps), so this tests whether the
memory survives a stream produced by physical movement, not navigation learning.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Ball, Box, Door, Key, Wall
from minigrid.minigrid_env import MiniGridEnv
from torch import nn
from torch.nn import functional as F

from multi_cue_memory import (
    COLORS,
    MODEL_KINDS,
    MemoryModel,
    parameter_count,
    updates_to_reach,
)

ROOT = Path(__file__).resolve().parent.parent
COLOR_NAMES = ("red", "green", "blue", "yellow")
KIND_CLASSES = (Key, Ball, Box)
FEATURES = 64
CELL_CHANNELS = 11 + 6 + 3  # object id, color id, state, one-hot
NOOP = 6
FORWARD = 2


@dataclass(frozen=True)
class GridSpec:
    cues: int
    delay: int  # corridor cells after the cue room; the door sits right after them
    distractors: int
    wait_probability: float = 0.0

    def __post_init__(self) -> None:
        if not 1 <= self.cues <= COLORS:
            raise ValueError("cues")
        if self.delay < 6:
            # shorter corridors would show the query door while the last cue is still in view
            raise ValueError("delay must be at least 6")
        if not 0 <= self.distractors <= self.delay:
            raise ValueError("distractors must fit in the corridor")
        if not 0 <= self.wait_probability < 1:
            raise ValueError("wait_probability must be within [0, 1)")


class CueCorridorEnv(MiniGridEnv):
    def __init__(self, layout: dict) -> None:
        self.layout = layout
        super().__init__(
            mission_space=MissionSpace(mission_func=lambda: "remember the cues"),
            width=layout["width"],
            height=7,
            max_steps=10_000,
            see_through_walls=False,
        )

    def _gen_grid(self, width: int, height: int) -> None:
        layout = self.layout
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)
        for x in range(1, width - 1):
            self.grid.set(x, 1, Wall())
            self.grid.set(x, 5, Wall())
        for x in range(layout["room_end"] + 1, width - 1):
            self.grid.set(x, 2, Wall())
            self.grid.set(x, 4, Wall())
        for x, color, kind in layout["cues"]:
            self.grid.set(x, 2, KIND_CLASSES[kind](COLOR_NAMES[color]))
        for x, row, color, kind in layout["distractors"]:
            self.grid.set(x, row, KIND_CLASSES[kind](COLOR_NAMES[color]))
        self.grid.set(width - 1, 3, Door(COLOR_NAMES[layout["query_color"]]))
        self.agent_pos = np.array((1, 3))
        self.agent_dir = 0
        self.mission = "remember the cues"


def make_layout(spec: GridSpec, rng: np.random.Generator) -> dict:
    colors = rng.permutation(COLORS)[: spec.cues]
    kinds = rng.integers(0, len(KIND_CLASSES), size=spec.cues)
    cue_x = [3 + 3 * i for i in range(spec.cues)]
    room_end = cue_x[-1] + 1
    width = (
        room_end + spec.delay + 2
    )  # corridor cells room_end+1.., then the door column
    xs = rng.permutation(np.arange(room_end + 1, width - 1))[: spec.distractors]
    distractors = [
        (
            int(x),
            int(rng.choice((2, 4))),
            int(rng.integers(0, COLORS)),
            int(rng.integers(0, len(KIND_CLASSES))),
        )
        for x in sorted(xs)
    ]
    asked = int(rng.integers(0, spec.cues))
    return {
        "width": int(width),
        "room_end": int(room_end),
        "cues": [(x, int(c), int(k)) for x, c, k in zip(cue_x, colors, kinds)],
        "distractors": distractors,
        "query_color": int(colors[asked]),
        "answer": int(kinds[asked]),
        "cue_colors": [int(c) for c in colors],
    }


def run_episode(spec: GridSpec, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    layout = make_layout(spec, rng)
    env = CueCorridorEnv(layout)
    try:
        obs, _ = env.reset(seed=seed)
        frames = [obs["image"]]
        moves = layout["width"] - 3  # from x=1 to the cell in front of the door
        for _ in range(moves):
            while rng.random() < spec.wait_probability:
                obs, *_ = env.step(NOOP)
                frames.append(obs["image"])
            obs, *_ = env.step(FORWARD)
            frames.append(obs["image"])
        if int(env.agent_pos[0]) != layout["width"] - 2:
            raise RuntimeError("the agent did not reach the door")
    finally:
        env.close()
    return {"images": np.stack(frames).astype(np.uint8), "layout": layout}


def build_dataset(
    specs: list[GridSpec], count: int, seed: int
) -> dict[str, torch.Tensor]:
    """count episodes, drawing the spec for each episode uniformly from specs."""
    rng = np.random.default_rng(seed)
    episodes = [
        run_episode(specs[int(rng.integers(0, len(specs)))], seed * 1_000_003 + i)
        for i in range(count)
    ]
    longest = max(len(item["images"]) for item in episodes)
    images = np.zeros((count, longest, 7, 7, 3), dtype=np.uint8)
    lengths = np.zeros(count, dtype=np.int64)
    for row, item in enumerate(episodes):
        images[row, : len(item["images"])] = item["images"]
        lengths[row] = len(item["images"])
    cue_kinds = np.zeros((count, COLORS), dtype=np.int64)
    cue_present = np.zeros((count, COLORS), dtype=np.bool_)
    for row, item in enumerate(episodes):
        for _, color, kind in item["layout"]["cues"]:
            cue_kinds[row, color] = kind
            cue_present[row, color] = True
    return {
        "images": torch.from_numpy(images),
        "lengths": torch.from_numpy(lengths),
        "answers": torch.tensor([item["layout"]["answer"] for item in episodes]),
        "delay": torch.tensor([_delay(item["layout"]) for item in episodes]),
        "query_slot": torch.tensor(
            [
                item["layout"]["cue_colors"].index(item["layout"]["query_color"])
                for item in episodes
            ]
        ),
        "cue_kinds": torch.from_numpy(cue_kinds),
        "cue_present": torch.from_numpy(cue_present),
    }


def _delay(layout: dict) -> int:
    return layout["width"] - layout["room_end"] - 2


def one_hot_cells(images: torch.Tensor) -> torch.Tensor:
    """uint8 (..., 7, 7, 3) -> float (..., 20, 7, 7)."""
    images = images.long()
    parts = (
        F.one_hot(images[..., 0].clamp(max=10), 11),
        F.one_hot(images[..., 1].clamp(max=5), 6),
        F.one_hot(images[..., 2].clamp(max=2), 3),
    )
    return torch.cat(parts, dim=-1).float().movedim(-1, -3)


class GridMemory(nn.Module):
    """Identical convolutional encoder for every model, followed by the memory under test."""

    def __init__(self, kind: str, hidden: int, rewire_seed: int = 0) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(CELL_CHANNELS, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 49, FEATURES),
            nn.Tanh(),
        )
        self.memory = MemoryModel(kind, hidden, rewire_seed, input_dim=FEATURES)

    def forward(self, images: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        batch, steps = images.shape[:2]
        features = self.encoder(one_hot_cells(images.flatten(0, 1))).view(
            batch, steps, -1
        )
        return self.memory(features, lengths)


def memory_parameters(model: GridMemory) -> int:
    return parameter_count(model.memory)


@torch.no_grad()
def accuracy(
    model: GridMemory, data: dict, device: torch.device, batch_size: int = 256
) -> float:
    model.eval()
    correct = 0
    for start in range(0, len(data["answers"]), batch_size):
        part = slice(start, start + batch_size)
        logits = model(
            data["images"][part].to(device), data["lengths"][part].to(device)
        )
        correct += int(logits.argmax(-1).cpu().eq(data["answers"][part]).sum())
    return correct / len(data["answers"])


def accuracy_by_delay(
    model: GridMemory, data: dict, device: torch.device
) -> dict[str, float]:
    result = {}
    for delay in sorted(set(data["delay"].tolist())):
        mask = data["delay"] == delay
        subset = {key: value[mask] for key, value in data.items()}
        result[str(delay)] = accuracy(model, subset, device)
    return result


def train_run(
    kind: str,
    seed: int,
    hidden: int,
    train: dict,
    validation: dict,
    tests: dict[str, dict],
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: torch.device,
) -> dict:
    torch.manual_seed(seed)
    model = GridMemory(kind, hidden, seed % 5).to(device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=learning_rate
    )
    generator = torch.Generator().manual_seed(seed + 7)
    curve = [{"epoch": 0, "validation": accuracy(model, validation, device)}]
    best, best_state = -1.0, None
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(train["answers"]), generator=generator)
        for indices in order.split(batch_size):
            loss = F.cross_entropy(
                model(
                    train["images"][indices].to(device),
                    train["lengths"][indices].to(device),
                ),
                train["answers"][indices].to(device),
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        score = accuracy(model, validation, device)
        curve.append(
            {"epoch": epoch, "loss": float(loss.detach()), "validation": score}
        )
        if score > best:
            best = score
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return {
        "kind": kind,
        "seed": seed,
        "hidden": model.memory.hidden,
        "memory_parameters": memory_parameters(model),
        "total_parameters": parameter_count(model),
        "learning_rate": learning_rate,
        "epochs": epochs,
        "episodes_per_epoch": len(train["answers"]),
        "best_validation": best,
        "tests": {
            name: {
                "accuracy": accuracy(model, data, device),
                "by_delay": accuracy_by_delay(model, data, device),
            }
            for name, data in tests.items()
        },
        "curve": curve,
        "seconds": time.perf_counter() - started,
    }


def cached(path: Path, factory) -> dict:
    if path.exists():
        return torch.load(path, weights_only=True)
    data = factory()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, path)
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models", nargs="+", choices=MODEL_KINDS, default=list(MODEL_KINDS)
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[1001, 1002, 1003])
    parser.add_argument("--cues", type=int, default=2)
    parser.add_argument("--train-delays", nargs="+", type=int, default=[6, 10])
    parser.add_argument(
        "--test-delays",
        nargs="+",
        type=int,
        default=[8, 14],
        help="corridor lengths never used for training",
    )
    parser.add_argument("--distractors", type=int, default=2)
    parser.add_argument("--wait-probability", type=float, default=0.0)
    parser.add_argument("--train-episodes", type=int, default=8192)
    parser.add_argument("--validation-episodes", type=int, default=1024)
    parser.add_argument("--test-episodes", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "data" / "grid_multi_cue"
    )
    args = parser.parse_args()
    device = torch.device(args.device)

    def specs(delays):
        return [
            GridSpec(args.cues, d, min(args.distractors, d), args.wait_probability)
            for d in delays
        ]

    tag = f"cues{args.cues}_dis{args.distractors}_wait{args.wait_probability}"
    cache = args.output_dir / "cache"
    train = cached(
        cache / f"{tag}_train_{args.train_delays}_{args.train_episodes}.pt",
        lambda: build_dataset(specs(args.train_delays), args.train_episodes, 11),
    )
    validation = cached(
        cache / f"{tag}_val_{args.train_delays}.pt",
        lambda: build_dataset(specs(args.train_delays), args.validation_episodes, 22),
    )
    tests = {
        "trained_delays": cached(
            cache / f"{tag}_test_{args.train_delays}.pt",
            lambda: build_dataset(specs(args.train_delays), args.test_episodes, 33),
        ),
        "unseen_delays": cached(
            cache / f"{tag}_unseen_{args.test_delays}.pt",
            lambda: build_dataset(specs(args.test_delays), args.test_episodes, 44),
        ),
    }
    torch.manual_seed(0)
    target = parameter_count(MemoryModel("fly", 1000, input_dim=FEATURES))
    hidden = {kind: matched_hidden_for(kind, target) for kind in args.models}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        for kind in args.models:
            result = train_run(
                kind,
                seed,
                hidden[kind],
                train,
                validation,
                tests,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                device=device,
            )
            result["setting"] = {
                "cues": args.cues,
                "train_delays": args.train_delays,
                "test_delays": args.test_delays,
                "distractors": args.distractors,
                "wait_probability": args.wait_probability,
            }
            (args.output_dir / f"{tag}_{kind}_seed{seed}.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print(
                json.dumps(
                    {
                        "tag": tag,
                        "kind": kind,
                        "seed": seed,
                        "val": round(result["best_validation"], 4),
                        "trained": round(
                            result["tests"]["trained_delays"]["accuracy"], 4
                        ),
                        "unseen": round(
                            result["tests"]["unseen_delays"]["accuracy"], 4
                        ),
                        "to90": updates_to_reach(
                            [
                                {"update": c["epoch"], "validation": c["validation"]}
                                for c in result["curve"]
                            ],
                            0.9,
                        ),
                    }
                ),
                flush=True,
            )


def matched_hidden_for(kind: str, target: int) -> int:
    """matched_hidden with the grid feature width as memory input."""
    if kind in ("fly", "rewired", "leaky"):
        return 1000
    low, high = 1, 512

    def count(n):
        return parameter_count(MemoryModel(kind, n, input_dim=FEATURES))

    while count(high) < target:
        high *= 2
    while low < high:
        middle = (low + high) // 2
        if count(middle) < target:
            low = middle + 1
        else:
            high = middle
    return min((low - 1, low), key=lambda n: abs(count(n) - target))


if __name__ == "__main__":
    main()

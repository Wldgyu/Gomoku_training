"""Fresh common CNN encoder, recurrent core, policy/value heads for all models."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

GRAPH_DIR = Path(__file__).resolve().parents[1] / "previous_research" / "data" / "subgraphs"
KINDS = ("fly", "rewired", "rnn", "gru")
FEATURES, HEAD = 64, 128


class Policy(nn.Module):
    def __init__(self, kind: str, seed: int, hidden: int = 1000, size: int = 9,
                 rewire_seed: int = 0):
        super().__init__()
        if kind not in KINDS:
            raise ValueError(kind)
        self.kind, self.hidden, self.size = kind, hidden, size
        self.rewire_seed = rewire_seed
        self.graph_sha256 = None
        # Isolated RNG scopes keep the common encoder identical across architectures.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.encoder = nn.Sequential(nn.Conv2d(4, 16, 3, padding=1), nn.ReLU(),
                                         nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(),
                                         nn.Flatten(), nn.Linear(16 * size * size, FEATURES), nn.Tanh())
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + 100_000)
            if kind in ("fly", "rewired"):
                suffix = f"_rewired_seed{rewire_seed}" if kind == "rewired" else ""
                path = GRAPH_DIR / f"malecns_cx_1000{suffix}.npz"
                self.graph_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
                with np.load(path) as graph:
                    self.hidden = len(graph["body_ids"])
                    pre = torch.tensor(graph["pre_index"].astype(np.int64))
                    post = torch.tensor(graph["post_index"].astype(np.int64))
                degree = torch.bincount(post, minlength=self.hidden).clamp(min=1)
                self.register_buffer("pre", pre)
                self.register_buffer("post", post)
                self.register_buffer("scale", degree[post].float().rsqrt())
                self.edge_values = nn.Parameter(torch.randn(len(pre)) * 0.2)
                self.input_layer = nn.Linear(FEATURES, self.hidden)
            else:
                self.core = (nn.RNNCell if kind == "rnn" else nn.GRUCell)(FEATURES, hidden)
                if kind == "rnn":
                    nn.init.orthogonal_(self.core.weight_hh)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + 200_000)
            self.readout = nn.Sequential(nn.Linear(self.hidden + FEATURES, HEAD), nn.Tanh())
            self.actor = nn.Linear(HEAD, size * size)
            self.critic = nn.Linear(HEAD, 1)
            # Small initial policy logits make exploration close to uniform.
            nn.init.orthogonal_(self.actor.weight, gain=0.01)
            nn.init.zeros_(self.actor.bias)

    def matrix(self) -> torch.Tensor | None:
        if self.kind not in ("fly", "rewired"):
            return None
        w = self.edge_values.new_zeros(self.hidden, self.hidden)
        return w.index_put((self.post, self.pre), self.edge_values * self.scale)

    def initial(self, batch: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch, self.hidden, device=device)

    def from_features(self, x: torch.Tensor, state: torch.Tensor, resets: torch.Tensor,
                      matrix: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        state = state * (~resets).unsqueeze(1)
        if self.kind in ("fly", "rewired"):
            state = 0.9 * state + 0.1 * torch.tanh(self.input_layer(x) + F.linear(state, matrix))
        else:
            state = self.core(x, state)
        y = self.readout(torch.cat((state, x), dim=1))
        return self.actor(y), self.critic(y).squeeze(1), state

    def step(self, obs: torch.Tensor, state: torch.Tensor, resets: torch.Tensor,
             matrix: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.kind in ("fly", "rewired") and matrix is None:
            matrix = self.matrix()
        return self.from_features(self.encoder(obs), state, resets, matrix)

    def sequence(self, obs: torch.Tensor, state: torch.Tensor, resets: torch.Tensor
                 ) -> tuple[torch.Tensor, torch.Tensor]:
        time, batch = obs.shape[:2]
        features = self.encoder(obs.flatten(0, 1)).reshape(time, batch, FEATURES)
        matrix = self.matrix()
        logits, values = [], []
        for t in range(time):
            p, v, state = self.from_features(features[t], state, resets[t], matrix)
            logits.append(p)
            values.append(v)
        return torch.stack(logits), torch.stack(values)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def parameter_formula(kind: str, hidden: int, size: int = 9) -> int:
    encoder = (4 * 16 * 9 + 16) + (16 * 16 * 9 + 16) + (16 * size * size + 1) * FEATURES
    heads = (hidden + FEATURES + 1) * HEAD + (HEAD + 1) * (size * size + 1)
    if kind in ("fly", "rewired"):
        with np.load(GRAPH_DIR / "malecns_cx_1000.npz") as graph:
            recurrent = len(graph["pre_index"]) + (FEATURES + 1) * 1000
    else:
        recurrent = (hidden * hidden + FEATURES * hidden + 2 * hidden) * (3 if kind == "gru" else 1)
    return encoder + heads + recurrent


def matched_hidden(kind: str, size: int = 9) -> int:
    if kind in ("fly", "rewired"):
        return 1000
    target = parameter_formula("fly", 1000, size)
    return min(range(16, 1001), key=lambda h: abs(parameter_formula(kind, h, size) - target))


def masked_distribution(logits: torch.Tensor, legal: torch.Tensor) -> torch.distributions.Categorical:
    if not bool(legal.any(dim=-1).all()):
        raise ValueError("A nonterminal board must have at least one empty intersection")
    return torch.distributions.Categorical(logits=logits.masked_fill(~legal, -1e9))


def load_policy(path: Path, device: torch.device) -> tuple[Policy, dict]:
    saved = torch.load(path, map_location="cpu", weights_only=True)
    config = saved["config"]
    model = Policy(config["kind"], config["seed"], config["hidden"], config["size"],
                   config["rewire_seed"])
    if model.graph_sha256 != saved["graph_sha256"]:
        raise RuntimeError("Graph source changed since training")
    model.load_state_dict(saved["model"])
    return model.to(device).eval(), saved

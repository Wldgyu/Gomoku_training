"""Spatial policy with a wider/deeper common trunk (same recurrent cores and readouts).

The original SpatialEncoder trunk is three convolutions with 16 channels. A
feed-forward probe (defense_probe.py) showed that this trunk, not the recurrent
core, limits defensive accuracy. The trunk here is shared identically by all four
recurrent kinds, so model comparisons stay controlled.
"""
import torch
from torch import nn
from torch.nn import functional as F

from .spatial import SpatialPolicy


class WideEncoder(nn.Module):
    def __init__(self, channels=64, blocks=6):
        super().__init__()
        self.channels = channels
        stem = nn.Sequential(nn.Conv2d(4, channels, 5, padding=2), nn.GroupNorm(8, channels), nn.ReLU())
        self.stem = stem
        self.blocks = nn.ModuleList(nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1), nn.GroupNorm(8, channels), nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1), nn.GroupNorm(8, channels)) for _ in range(blocks))
        self.project = nn.Sequential(nn.Linear(channels, 64), nn.Tanh())

    def trunk(self, obs):
        x = self.stem(obs)
        for block in self.blocks:
            x = F.relu(x + block(x))
        return x

    def forward(self, obs):
        return self.project(self.trunk(obs).mean((-2, -1)))


class WideSpatialPolicy(SpatialPolicy):
    def __init__(self, kind, seed, hidden=1000, size=12, rewire_seed=0, channels=64, blocks=6,
                 threat_aux=False, recurrent_cell=False, graph_neurons=1000):
        super().__init__(kind, seed, hidden, size, rewire_seed, threat_aux, recurrent_cell, graph_neurons)
        self.channels, self.blocks = channels, blocks
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + 400_000)
            self.encoder = WideEncoder(channels, blocks)
            self.actor = nn.Linear(128, 2 * channels)
            self.cell_actor = nn.Conv2d(channels, 1, 1)
            nn.init.zeros_(self.actor.weight)
            nn.init.zeros_(self.actor.bias)
            nn.init.normal_(self.cell_actor.weight, std=0.1)
            nn.init.zeros_(self.cell_actor.bias)

    def sequence(self, obs, state, resets):
        steps, count = obs.shape[:2]
        maps = self.encoder.trunk(obs.flatten(0, 1)).reshape(steps, count, self.channels, self.size, self.size)
        features = self.encoder.project(maps.mean((-2, -1)))
        matrix = self.matrix()
        logits, values = [], []
        for t in range(steps):
            modulation, value, state = self.from_features(features[t], state, resets[t], matrix)
            logits.append(self.policy_logits(maps[t], modulation, state))
            values.append(value)
        return torch.stack(logits), torch.stack(values)


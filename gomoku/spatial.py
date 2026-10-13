"""Common spatial encoder and recurrently conditioned, shared per-cell policy."""
import torch
from torch import nn
from torch.nn import functional as F

from .models import Policy


class SpatialEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = nn.Sequential(nn.Conv2d(4, 16, 5, padding=2), nn.ReLU(),
                                   nn.Conv2d(16, 16, 5, padding=2), nn.ReLU(),
                                   nn.Conv2d(16, 16, 3, padding=1), nn.ReLU())
        self.project = nn.Sequential(nn.Linear(16, 64), nn.Tanh())

    def forward(self, obs):
        return self.project(self.trunk(obs).mean((-2, -1)))


class SpatialPolicy(Policy):
    def __init__(self, kind, seed, hidden=1000, size=12, rewire_seed=0,
                 threat_aux=False):
        super().__init__(kind, seed, hidden, size, rewire_seed)
        self.threat_aux = threat_aux
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + 300_000)
            self.encoder = SpatialEncoder()
            self.actor = nn.Linear(128, 32)
            self.cell_actor = nn.Conv2d(16, 1, 1)
            nn.init.zeros_(self.actor.weight)
            nn.init.zeros_(self.actor.bias)
            nn.init.normal_(self.cell_actor.weight, std=0.01)
            nn.init.zeros_(self.cell_actor.bias)
            if threat_aux:
                # The extra channel identifies the candidate move whose
                # consequences are being predicted. Outputs are opponent
                # immediate-win and double-threat maps.
                self.threat_head = nn.Sequential(
                    nn.Conv2d(17, 16, 5, padding=2), nn.ReLU(),
                    nn.Conv2d(16, 2, 5, padding=2),
                )
                for layer in self.threat_head:
                    if isinstance(layer, nn.Conv2d):
                        nn.init.normal_(layer.weight, std=0.01)
                        nn.init.zeros_(layer.bias)

    def cell_logits(self, maps, modulation):
        scale, bias = modulation.chunk(2, dim=1)
        maps = F.relu(maps * (1 + 0.1 * scale.tanh()[:, :, None, None])
                      + 0.1 * bias.tanh()[:, :, None, None])
        return self.cell_actor(maps).flatten(1)

    def step(self, obs, state, resets, matrix=None):
        if self.kind in ("fly", "rewired") and matrix is None:
            matrix = self.matrix()
        maps = self.encoder.trunk(obs)
        features = self.encoder.project(maps.mean((-2, -1)))
        modulation, value, state = self.from_features(features, state, resets, matrix)
        return self.cell_logits(maps, modulation), value, state

    def sequence(self, obs, state, resets):
        steps, count = obs.shape[:2]
        maps = self.encoder.trunk(obs.flatten(0, 1)).reshape(steps, count, 16, self.size, self.size)
        features = self.encoder.project(maps.mean((-2, -1)))
        matrix = self.matrix()
        logits, values = [], []
        for t in range(steps):
            modulation, value, state = self.from_features(features[t], state, resets[t], matrix)
            logits.append(self.cell_logits(maps[t], modulation))
            values.append(value)
        return torch.stack(logits), torch.stack(values)
    def threat(self, obs, state, resets, actions, matrix=None):
        if not self.threat_aux:
            raise RuntimeError("Threat head is not enabled")
        if matrix is None and self.kind in ("fly", "rewired"):
            matrix = self.matrix()
        maps = self.encoder.trunk(obs)
        features = self.encoder.project(maps.mean((-2, -1)))
        modulation, _, state = self.from_features(features, state, resets, matrix)
        scale, bias = modulation.chunk(2, dim=1)
        conditioned = F.relu(maps * (1 + 0.1 * scale.tanh()[:, :, None, None])
                             + 0.1 * bias.tanh()[:, :, None, None])
        chosen = F.one_hot(actions.long(), self.size * self.size).to(conditioned.dtype)
        chosen = chosen.reshape(-1, 1, self.size, self.size)
        return self.threat_head(torch.cat((conditioned, chosen), dim=1)), state




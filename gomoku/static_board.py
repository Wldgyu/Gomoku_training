"""Current-board processing with internal recurrence and no cross-move memory."""
import torch
from torch import nn

class StaticBoardPolicy(nn.Module):
    def __init__(self, base, iterations=8):
        super().__init__()
        if iterations < 1:
            raise ValueError('iterations must be positive')
        self.base = base
        self.iterations = int(iterations)
        self.kind, self.size, self.hidden = base.kind, base.size, base.hidden
        self.graph_sha256 = base.graph_sha256

    @property
    def encoder(self):
        return self.base.encoder

    def matrix(self):
        return self.base.matrix()

    def initial(self, batch, device):
        return self.base.initial(batch, device)

    def from_features(self, features, state, resets, matrix):
        local = self.initial(len(features), features.device)
        keep = torch.zeros(len(features), dtype=torch.bool, device=features.device)
        for _ in range(self.iterations):
            output, value, local = self.base.from_features(features, local, keep, matrix)
        return output, value, torch.zeros_like(local)

    def step(self, obs, state, resets, matrix=None):
        if matrix is None:
            matrix = self.matrix()
        if hasattr(self.base, 'cell_logits'):
            maps = self.encoder.trunk(obs)
            features = self.encoder.project(maps.mean((-2, -1)))
            modulation, value, carry = self.from_features(features, state, resets, matrix)
            return self.base.cell_logits(maps, modulation), value, carry
        return self.from_features(self.encoder(obs), state, resets, matrix)

    def sequence(self, obs, state, resets):
        matrix = self.matrix()
        logits, values = [], []
        for t in range(len(obs)):
            logit, value, _ = self.step(obs[t], state, resets[t], matrix)
            logits.append(logit)
            values.append(value)
        return torch.stack(logits), torch.stack(values)

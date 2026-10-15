"""History-conditioned tactical supervision with detached full-history burn-in."""
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .models import label_set_nll
from .tactics import TacticalData

CATEGORIES = {1: "win", 2: "block", 3: "create_fork", 4: "prevent_fork",
              5: "create_forcing_chain", 6: "prevent_forcing_chain"}


def prefix_observations(histories: torch.Tensor, size: int):
    """Observations at this player's PREVIOUS turns, excluding the target turn.

    History is alternating black/white, padded with -1. No future stone is visible.
    Output is time,batch,4,size,size and a time,batch validity mask.
    """
    lengths = (histories >= 0).sum(1)
    count = len(histories)
    turns = lengths // 2
    steps = int(turns.max())
    if steps == 0:
        return None, None
    max_plies = int(lengths.max())
    moves = histories[:, :max_plies]
    valid = moves >= 0
    sign = torch.where(torch.arange(max_plies, device=moves.device) % 2 == 0, 1, -1)
    stones = F.one_hot(moves.clamp(min=0).long(), size * size).to(torch.int8)
    stones = stones * valid[:, :, None] * sign[None, :, None]
    boards = stones.cumsum(1, dtype=torch.int8)
    white = lengths % 2 == 1
    before = torch.arange(steps, device=moves.device)[None, :] * 2 + white[:, None]
    previous = (before - 1).clamp(0, max_plies - 1)
    prefix = boards.gather(1, previous[:, :, None].expand(-1, -1, size * size))
    prefix = prefix * (before > 0)[:, :, None]
    colors = torch.where(white, -1, 1)
    last = F.one_hot(moves.gather(1, previous).clamp(min=0).long(), size * size).to(torch.int8)
    last = last * (before > 0)[:, :, None]
    black = (~white)[:, None, None].expand(count, steps, size * size)
    obs = torch.stack((prefix == colors[:, None, None], prefix == -colors[:, None, None],
                       last, black), 2).to(torch.float32)
    active = torch.arange(steps, device=moves.device)[:, None] < turns[None, :]
    return obs.permute(1, 0, 2, 3).reshape(steps, count, 4, size, size), active


@torch.no_grad()
def burn_in(model, histories: torch.Tensor):
    """Recompute preceding state with current weights; gradient starts at target."""
    state = model.initial(len(histories), histories.device)
    obs, active = prefix_observations(histories, model.size)
    if obs is None:
        return state
    steps, count = active.shape
    features = torch.zeros(steps, count, 64, device=histories.device)
    features[active] = model.encoder(obs[active])
    resets = torch.zeros(count, dtype=torch.bool, device=histories.device)
    matrix = model.matrix()
    for t in range(steps):
        _, _, new = model.from_features(features[t], state, resets, matrix)
        state = torch.where(active[t, :, None], new, state)
    return state


class HistoryTacticalData(TacticalData):
    def __init__(self, path: Path, device: torch.device, mode: str = "history"):
        super().__init__(path, device)
        self.mode = mode
        self.history_tensor = torch.from_numpy(self.histories.astype(np.int64)).to(device)
        self.frames_used = 0
        self.balanced = False
        self.base_count = len(self.obs)
        self.dagger_fraction = 0.0
        self.root_groups = []
        with np.load(path) as saved:
            self.dagger_keys = set(saved["keys"].tolist())
            if "root_keys" in saved:
                root_keys = saved["root_keys"].copy()
                if len(root_keys) != len(self.obs):
                    raise ValueError("One root key per training record is required")
                self.root_groups = [torch.as_tensor(np.flatnonzero(root_keys == key), device=device)
                                    for key in np.unique(root_keys)]
        self.category_groups = [torch.as_tensor(np.flatnonzero(self.category == c),device=device)
                                for c in sorted(set(self.category.tolist()))]
        self.samples_by_category = {str(c):0 for c in sorted(set(self.category.tolist()))}

    def append_file(self, path: Path, dagger_fraction: float = 0.5):
        if self.root_groups:
            raise ValueError("Merge root-grouped data with provenance before appending")
        with np.load(path) as saved:
            obs = torch.from_numpy(saved["obs"].astype(np.float32)).to(self.obs.device)
            labels = torch.from_numpy(saved["labels"]).to(self.obs.device)
            category = saved["category"].copy()
            histories = saved["histories"].copy()
            keys = saved["keys"].tolist()
        keep = np.array([key not in self.dagger_keys for key in keys])
        if not keep.any():
            return
        self.dagger_keys.update(key for key, selected in zip(keys, keep) if selected)
        obs = obs[torch.from_numpy(keep).to(obs.device)]
        labels = labels[torch.from_numpy(keep).to(labels.device)]
        category = category[keep]
        histories = histories[keep]
        self.obs = torch.cat((self.obs, obs))
        self.labels = torch.cat((self.labels, labels))
        self.category = np.concatenate((self.category, category))
        self.histories = np.concatenate((self.histories, histories))
        self.history_tensor = torch.cat((
            self.history_tensor,
            torch.from_numpy(histories.astype(np.int64)).to(self.obs.device),
        ))
        self.legal = torch.cat((
            self.legal,
            (obs[:, 0] + obs[:, 1]).flatten(1) == 0,
        ))
        self.dagger_fraction = dagger_fraction
        categories = sorted(set(self.category.tolist()))
        self.category_groups = [
            torch.as_tensor(np.flatnonzero(self.category == c), device=self.obs.device)
            for c in categories
        ]
        for category_id in categories:
            self.samples_by_category.setdefault(str(category_id), 0)

    def sample_ids(self, batch: int):
        if self.root_groups:
            group_ids = torch.randint(len(self.root_groups), (batch,), device=self.obs.device)
            ids = torch.empty(batch, dtype=torch.long, device=self.obs.device)
            for group, candidates in enumerate(self.root_groups):
                mask = group_ids == group
                n = int(mask.sum())
                ids[mask] = candidates[torch.randint(len(candidates), (n,), device=self.obs.device)]
        elif self.dagger_fraction and len(self.obs) > self.base_count:
            extra = min(batch - 1, max(1, round(batch * self.dagger_fraction)))
            base = batch - extra
            ids = torch.cat((
                torch.randint(self.base_count, (base,), device=self.obs.device),
                torch.randint(self.base_count, len(self.obs), (extra,), device=self.obs.device),
            ))
            ids = ids[torch.randperm(batch, device=self.obs.device)]
        elif self.balanced:
            group_ids = torch.randint(len(self.category_groups),(batch,),device=self.obs.device)
            ids = torch.empty(batch,dtype=torch.long,device=self.obs.device)
            for group, candidates in enumerate(self.category_groups):
                mask = group_ids == group
                n = int(mask.sum())
                ids[mask] = candidates[torch.randint(len(candidates),(n,),device=self.obs.device)]
        else:
            ids = torch.randint(len(self.obs), (batch,), device=self.obs.device)
        return ids

    def loss(self, model, batch: int):
        ids = self.sample_ids(batch)
        cats = self.category[ids.cpu().numpy()]
        for c in self.samples_by_category:
            self.samples_by_category[c] += int((cats == int(c)).sum())
        if self.mode == "history":
            history = self.history_tensor[ids]
            self.frames_used += int(((history >= 0).sum(1) // 2).sum())
            state = burn_in(model, history)
        else:
            state = model.initial(batch, self.obs.device)
        logits, _, _ = model.step(self.obs[ids], state,
                                  torch.zeros(batch, dtype=torch.bool, device=self.obs.device))
        return label_set_nll(logits, self.legal[ids], self.labels[ids])

    @torch.no_grad()
    def evaluate(self, model, history=False):
        hits = []
        for start in range(0, len(self.obs), 128):
            obs = self.obs[start:start + 128]
            state = burn_in(model, self.history_tensor[start:start + len(obs)]) if history else model.initial(len(obs), obs.device)
            logits, _, _ = model.step(obs, state, torch.zeros(len(obs), dtype=torch.bool, device=obs.device))
            action = logits.masked_fill(~self.legal[start:start + len(obs)], -1e9).argmax(1)
            hits.extend(self.labels[start:start + len(obs)].gather(1, action[:, None]).squeeze(1).cpu().tolist())
        hits = np.asarray(hits)
        return {name: dict(positions=int((self.category == cat).sum()),
                           correct=int(hits[self.category == cat].sum()),
                           accuracy=float(hits[self.category == cat].mean()))
                for cat, name in CATEGORIES.items() if (self.category == cat).any()} | {"history_replayed": history}




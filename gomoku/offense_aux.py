"""Fixed history-supervision mixtures for experiments E and F."""
from __future__ import annotations

from .history_aux import HistoryTacticalData


class FixedMixtureData:
    def __init__(self, base_path, defense_path, attack_path, device, mode,
                 base_batch=64, defense_batch=32, attack_batch=32,
                 preparation_path=None, preparation_batch=0):
        paths = [base_path, defense_path, attack_path]
        quotas = [base_batch, defense_batch, attack_batch]
        if preparation_path is not None:
            paths.append(preparation_path)
            quotas.append(preparation_batch)
        if any(q < 1 for q in quotas):
            raise ValueError(f"Every fixed-mixture quota must be positive: {quotas}")
        self.datasets = [HistoryTacticalData(path, device, mode) for path in paths]
        self.quotas = quotas
        self.total_batch = sum(self.quotas)
        self.balanced = False
        self.dagger_records = 0
        self._frame_offset = 0
        self._saved_counts = {}

    def loss(self, model, batch):
        if batch != self.total_batch:
            raise ValueError((batch, self.total_batch))
        losses = [dataset.loss(model, quota)
                  for dataset, quota in zip(self.datasets, self.quotas)]
        return sum(loss * quota for loss, quota in zip(losses, self.quotas)) / batch

    @property
    def frames_used(self):
        return self._frame_offset + sum(dataset.frames_used for dataset in self.datasets)

    @frames_used.setter
    def frames_used(self, value):
        self._frame_offset = int(value) - sum(dataset.frames_used for dataset in self.datasets)

    @property
    def samples_by_category(self):
        result = dict(self._saved_counts)
        for dataset in self.datasets:
            for key, value in dataset.samples_by_category.items():
                result[key] = result.get(key, 0) + value
        return result

    @samples_by_category.setter
    def samples_by_category(self, value):
        self._saved_counts = dict(value)

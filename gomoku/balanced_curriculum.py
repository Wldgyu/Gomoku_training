"""Fixed defense/color quotas and preservation loss for a two-stage curriculum."""
import numpy as np
import torch
from torch.nn import functional as F


STAGES = (
    ("basic", {"guard": 64, "block": 32, "win": 32}),
    ("forcing", {"guard": 48, "block": 16, "prevent": 16, "win": 16, "fork": 16, "chain": 16}),
)
CATEGORIES = {"guard": {2}, "block": {2}, "win": {1}, "prevent": {4, 6}, "fork": {3}, "chain": {5}}


def sample_batch(records, quotas, rng):
    """Exact per-group and black/white quotas; reserve a share for new corrections."""
    categories = np.array([int(r["category"]) for r in records])
    black = np.array([bool(r["obs"][3, 0, 0]) for r in records])
    sources = np.array([r["source"] for r in records])
    ids, ledger = [], {}
    for group, quota in quotas.items():
        if quota <= 0 or quota % 2:
            raise ValueError("Every curriculum quota must be positive and even")
        allowed = np.isin(categories, list(CATEGORIES[group]))
        if group == "guard":
            allowed &= sources == "guard"
        ledger[group] = {}
        for color, mask in (("black", black), ("white", ~black)):
            pool = np.flatnonzero(allowed & mask)
            if not len(pool):
                raise ValueError(f"No {color} records for curriculum group {group}")
            fresh = pool[sources[pool] == "fresh"]
            n = quota // 2
            new_count = n // 4 if len(fresh) else 0
            chosen = np.concatenate((rng.choice(pool, n-new_count), rng.choice(fresh, new_count))) if new_count else rng.choice(pool, n)
            ids.extend(chosen.tolist())
            ledger[group][color] = n
    return np.asarray(ids, dtype=np.int64), ledger


def preservation_loss(student_logits, teacher_logits, legal, labels, categories):
    """Keep the teacher distribution only on defense examples it answers correctly."""
    teacher_logits = teacher_logits.detach().masked_fill(~legal, -1e9)
    student_logits = student_logits.masked_fill(~legal, -1e9)
    actions = teacher_logits.argmax(1)
    correct = labels.gather(1, actions[:, None]).squeeze(1)
    defense = (categories == 2) | (categories == 4) | (categories == 6)
    selected = correct & defense
    if not bool(selected.any()):
        return student_logits.sum() * 0, 0
    loss = F.kl_div(F.log_softmax(student_logits[selected], dim=1),
                    F.softmax(teacher_logits[selected], dim=1), reduction="batchmean")
    return loss, int(selected.sum())

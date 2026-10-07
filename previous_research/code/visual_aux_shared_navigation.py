"""Shared learned forward/stop policy, then frozen answer models on its rollouts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from grid_multi_cue import CueCorridorEnv, GridSpec, make_layout, one_hot_cells
from multi_cue_visual_aux import OUT, VisualAuxModel, dataset

NAV = OUT / "shared_navigation"


class StopPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(20, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(),
            nn.Flatten(), nn.Linear(16 * 49, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, images):
        return self.net(one_hot_cells(images)).squeeze(-1)


def frame_examples(data, episodes=2048):
    pos, neg = [], []
    for i in range(min(episodes, len(data["answers"]))):
        length = int(data["lengths"][i])
        pos.append(data["images"][i, length - 1])
        neg.append(data["images"][i, :length - 1])
    return torch.stack(pos), torch.cat(neg)


def train_policy(device):
    path = NAV / "stop_policy.pt"
    if path.exists():
        saved = torch.load(path, map_location="cpu", weights_only=True)
        policy = StopPolicy().to(device)
        policy.load_state_dict(saved["state"])
        policy.eval()
        return policy
    torch.manual_seed(1907)
    policy = StopPolicy().to(device)
    train = dataset("train", 8192, (6, 10), 61)
    pos, neg = frame_examples(train, 4096)
    opt = torch.optim.AdamW(policy.parameters(), lr=0.001)
    gen = torch.Generator().manual_seed(1907)
    for update in range(1, 501):
        pos_ids = torch.randint(len(pos), (128,), generator=gen)
        neg_ids = torch.randint(len(neg), (128,), generator=gen)
        images = torch.cat((pos[pos_ids], neg[neg_ids])).to(device)
        labels = torch.cat((torch.ones(128), torch.zeros(128))).to(device)
        logits = policy(images)
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if update % 100 == 0:
            print(json.dumps({"stop_update": update, "loss": float(loss.detach())}), flush=True)
    policy.eval()
    NAV.mkdir(parents=True, exist_ok=True)
    torch.save({"state": {k: v.detach().cpu() for k, v in policy.state_dict().items()},
                "updates": 500, "train_episodes": 4096}, path)
    return policy


@torch.no_grad()
def policy_stop(policy, image, device):
    inp = torch.as_tensor(image, device=device).unsqueeze(0)
    return bool(policy(inp).sigmoid().item() >= 0.5)


@torch.no_grad()
def collect_rollouts(policy, device, count=512, seed=91_713):
    rng = np.random.default_rng(seed)
    frames, lengths, answers, reached = [], [], [], []
    for episode in range(count):
        delay = (8, 14)[episode % 2]
        layout = make_layout(GridSpec(2, delay, 2), rng)
        env = CueCorridorEnv(layout)
        try:
            obs, _ = env.reset(seed=seed * 1_000_003 + episode)
            path = [obs["image"]]
            stopped = False
            for _ in range(layout["width"] + 5):
                if policy_stop(policy, obs["image"], device):
                    stopped = True
                    break
                obs, *_ = env.step(2)
                path.append(obs["image"])
            success = stopped and int(env.agent_pos[0]) == layout["width"] - 2
            frames.append(np.stack(path).astype(np.uint8))
            lengths.append(len(path))
            answers.append(layout["answer"])
            reached.append(success)
        finally:
            env.close()
    longest = max(lengths)
    images = np.zeros((count, longest, 7, 7, 3), dtype=np.uint8)
    for i, path in enumerate(frames):
        images[i, :len(path)] = path
    result = {"images": torch.from_numpy(images),
              "lengths": torch.tensor(lengths),
              "answers": torch.tensor(answers),
              "reached": torch.tensor(reached, dtype=torch.bool),
              "count": count, "seed": seed}
    NAV.mkdir(parents=True, exist_ok=True)
    torch.save(result, NAV / "unseen_rollouts.pt")
    return result


@torch.no_grad()
def answer_on_rollouts(model, rollouts, device):
    reached = rollouts["reached"]
    total = int(reached.sum())
    correct = 0
    ids = reached.nonzero(as_tuple=False).flatten()
    for part in ids.split(128):
        logits = model(rollouts["images"][part].to(device),
                       rollouts["lengths"][part].to(device))[0]
        correct += int((logits.argmax(-1).cpu() == rollouts["answers"][part]).sum())
    return {"movement_success": total / rollouts["count"],
            "answer_given_movement_success": correct / total if total else None,
            "end_to_end_answer_success": correct / rollouts["count"],
            "episodes": rollouts["count"]}


def main():
    torch.set_num_threads(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = train_policy(device)
    rollouts = collect_rollouts(policy, device)
    summary = {"policy": "shared learned CNN forward/stop",
               "training_episodes": 4096, "updates": 500,
               "rollout_episodes": rollouts["count"],
               "movement_success": float(rollouts["reached"].float().mean()),
               "models": {}}
    for path in sorted((OUT / "runs").glob("*.pt")):
        saved = torch.load(path, map_location="cpu", weights_only=True)
        model = VisualAuxModel(saved["kind"], saved["hidden"], saved["seed"],
                               saved["aux"]).to(device)
        model.load_state_dict(saved["state"])
        model.eval()
        result = answer_on_rollouts(model, rollouts, device)
        summary["models"][path.stem] = result
        print(path.stem, result, flush=True)
    (NAV / "results.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"movement_success": summary["movement_success"]}), flush=True)


if __name__ == "__main__":
    main()

"""첫 입력의 이진 신호를 나중에 맞히는 과제로 MaleCNS 그래프의 학습을 확인한다.

합성 진단 과제이며 MiniGrid 또는 오목 학습 성능을 뜻하지 않는다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from project_config import PILOT_NEURON_COUNTS


class ConnectomeMemoryNet(nn.Module):
    """실제 연결 위치에만 가중치를 두고, 시간에 따라 상태를 유지하는 모델."""
    def __init__(self, graph_path: Path, leak: float = 0.9):
        super().__init__()
        graph = np.load(graph_path)
        self.neurons = len(graph["body_ids"])
        pre = torch.from_numpy(graph["pre_index"].astype(np.int64))
        post = torch.from_numpy(graph["post_index"].astype(np.int64))
        in_degree = torch.bincount(post, minlength=self.neurons).clamp(min=1)
        self.register_buffer("pre", pre)
        self.register_buffer("post", post)
        self.register_buffer("edge_scale", in_degree[post].float().rsqrt())
        self.edge_values = nn.Parameter(torch.randn(len(pre)) * 0.2)
        self.input_layer = nn.Linear(4, self.neurons)
        self.output_layer = nn.Linear(self.neurons, 2)
        self.leak = leak

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """각 시점의 입력과 이전 뉴런 상태를 합쳐 마지막 시점의 답을 예측한다."""
        batch, steps, _ = x.shape
        state = x.new_zeros(batch, self.neurons)
        weights = x.new_zeros(self.neurons, self.neurons)
        weights = weights.index_put(
            (self.post, self.pre), self.edge_values * self.edge_scale, accumulate=True
        )
        # leak 비율의 이전 상태를 남겨 과거 신호가 즉시 사라지지 않게 한다.
        for step in range(steps):
            drive = self.input_layer(x[:, step])
            state = self.leak * state + (1 - self.leak) * torch.tanh(
                drive + F.linear(state, weights)
            )
        return self.output_layer(state)


def make_batch(
    batch: int, steps: int, device: torch.device, generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    """첫 시점에만 정답 신호를 제시하고 나머지 시점에는 방해 잡음을 넣는다."""
    labels = torch.randint(0, 2, (batch,), device=device, generator=generator)
    x = torch.zeros(batch, steps, 4, device=device)
    x[:, 0, 0] = labels.float() * 2 - 1
    x[:, 0, 3] = 1  # 신호가 등장한 시점을 알려주되 정답값은 담지 않는다.
    x[:, :, 1:3] = 0.2 * torch.randn(
        batch, steps, 2, device=device, generator=generator
    )
    return x, labels


def file_hash(path: Path) -> str:
    """체크포인트와 사용한 그래프가 같은 파일인지 확인할 해시를 만든다."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    """독립 난수로 학습·평가하고 결과와 모델 체크포인트를 저장한다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--graph",
        type=Path,
        default=Path(f"data/subgraphs/malecns_cx_{PILOT_NEURON_COUNTS[0]}.npz"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/sanity"))
    parser.add_argument("--steps", type=int, default=250, help="optimizer updates")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--delay", type=int, default=16, help="sequence length")
    parser.add_argument("--eval-batch", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if min(args.steps, args.batch, args.delay, args.eval_batch) < 1:
        parser.error("all sizes and steps must be positive")

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    # 평가 난수를 학습 난수와 분리해 같은 학습 예시를 다시 평가하지 않는다.
    train_rng = torch.Generator(device=device).manual_seed(args.seed + 1)
    eval_rng = torch.Generator(device=device).manual_seed(args.seed + 10_000)
    model = ConnectomeMemoryNet(args.graph).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002)
    eval_x, eval_y = make_batch(args.eval_batch, args.delay, device, eval_rng)

    def evaluate() -> float:
        """가중치를 바꾸지 않은 채 보관한 평가 예시의 정확도를 계산한다."""
        model.eval()
        with torch.no_grad():
            accuracy = (model(eval_x).argmax(dim=1) == eval_y).float().mean().item()
        model.train()
        return accuracy

    initial_accuracy = evaluate()
    history = []
    # update는 최적화 횟수다. 게임 환경 상호작용 스텝과 혼동하면 안 된다.
    for update in range(1, args.steps + 1):
        x, y = make_batch(args.batch, args.delay, device, train_rng)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        if model.edge_values.grad is None or not torch.isfinite(model.edge_values.grad).all():
            raise RuntimeError("graph edge gradients are missing or invalid")
        optimizer.step()
        if update == 1 or update % 25 == 0 or update == args.steps:
            accuracy = evaluate()
            row = {
                "update": update,
                "training_loss": float(loss.item()),
                "heldout_accuracy": accuracy,
            }
            history.append(row)
            print(json.dumps(row), flush=True)

    final_accuracy = evaluate()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.graph.stem}_delay{args.delay}_seed{args.seed}"
    checkpoint = args.output_dir / f"{stem}.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "graph_sha256": file_hash(args.graph),
            "neurons": model.neurons,
            "training_updates": args.steps,
            "delay": args.delay,
            "seed": args.seed,
            "task": "synthetic delayed binary cue",
        },
        checkpoint,
    )
    result = {
        "task": "synthetic delayed binary cue",
        "graph": str(args.graph),
        "graph_sha256": file_hash(args.graph),
        "checkpoint": str(checkpoint),
        "device": str(device),
        "neurons": model.neurons,
        "edges": model.edge_values.numel(),
        "trainable_parameters": sum(p.numel() for p in model.parameters()),
        "delay": args.delay,
        "training_updates": args.steps,
        "batch": args.batch,
        "heldout_examples": args.eval_batch,
        "chance_accuracy": 0.5,
        "initial_heldout_accuracy": initial_accuracy,
        "final_heldout_accuracy": final_accuracy,
        "history": history,
    }
    output = args.output_dir / f"{stem}.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Saved metrics: {output}", flush=True)
    print(f"Saved checkpoint: {checkpoint}", flush=True)


if __name__ == "__main__":
    main()

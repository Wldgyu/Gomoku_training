"""추출한 Fly 그래프의 학습 한 스텝 시간과 CUDA 메모리를 측정한다.

무작위 입력으로 순전파·역전파를 실행하는 자원 진단이며 게임 학습 결과가 아니다.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from project_config import PILOT_NEURON_COUNTS


class FlyProbe(nn.Module):
    """실제 연결 위치에만 학습 파라미터를 두는 순환망 진단 모델."""
    def __init__(self, graph_path: Path, mode: str, input_size: int = 32):
        super().__init__()
        graph = np.load(graph_path)
        node_count = len(graph["body_ids"])
        pre = torch.from_numpy(graph["pre_index"].astype(np.int64))
        post = torch.from_numpy(graph["post_index"].astype(np.int64))
        in_degree = torch.bincount(post, minlength=node_count).clamp(min=1)
        scale = in_degree[post].float().rsqrt()
        self.register_buffer("pre", pre)
        self.register_buffer("post", post)
        self.register_buffer("scale", scale)
        self.edge_values = nn.Parameter(torch.randn(len(pre)) * 0.1)
        self.input_layer = nn.Linear(input_size, node_count)
        self.output_layer = nn.Linear(node_count, 8)
        self.node_count = node_count
        self.mode = mode

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """같은 연결을 밀집 계산과 COO 희소 계산으로 각각 수행한다."""
        batch_size, sequence_length, _ = observations.shape
        state = observations.new_zeros(batch_size, self.node_count)
        values = self.edge_values * self.scale
        # 밀집 모드도 빈 연결에는 학습 파라미터를 만들지 않는다.
        if self.mode == "dense":
            matrix = observations.new_zeros(self.node_count, self.node_count)
            matrix = matrix.index_put((self.post, self.pre), values, accumulate=True)
        elif self.mode == "sparse":
            indices = torch.stack((self.post, self.pre))
            matrix = torch.sparse_coo_tensor(
                indices, values, (self.node_count, self.node_count), check_invariants=False
            ).coalesce()
        else:
            raise ValueError(self.mode)
        for step in range(sequence_length):
            if self.mode == "dense":
                recurrent = F.linear(state, matrix)
            else:
                recurrent = torch.sparse.mm(matrix, state.T).T
            state = torch.tanh(self.input_layer(observations[:, step]) + recurrent)
        return self.output_layer(state)


def benchmark(graph_path: Path, mode: str, args: argparse.Namespace) -> dict:
    """역전파·가중치 갱신까지 포함한 시간을 재고 PyTorch 메모리 피크를 기록한다."""
    device = torch.device(args.device)
    model = FlyProbe(graph_path, mode).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    observations = torch.randn(args.batch, args.steps, 32, device=device)
    labels = torch.randint(0, 8, (args.batch,), device=device)
    times = []
    # CUDA 호출은 비동기이므로 시간을 재기 전후에 동기화한다.
    if device.type == "cuda":
        torch.cuda.synchronize()
        free_before, total = torch.cuda.mem_get_info()
        torch.cuda.reset_peak_memory_stats()
    for repeat in range(args.warmup + args.repeats):
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        logits = model(observations)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        if model.edge_values.grad is None or not torch.isfinite(model.edge_values.grad).all():
            raise RuntimeError("Missing or non-finite edge gradients")
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        if repeat >= args.warmup:
            times.append((time.perf_counter() - started) * 1000)
    result = {
        "graph": graph_path.name,
        "mode": mode,
        "device": str(device),
        "neurons": model.node_count,
        "edges": model.edge_values.numel(),
        "trainable_parameters": sum(p.numel() for p in model.parameters()),
        "batch": args.batch,
        "unroll_steps": args.steps,
        "warmup": args.warmup,
        "timed_repeats": args.repeats,
        "step_ms_median": float(np.median(times)),
        "step_ms_min": float(min(times)),
        "step_ms_max": float(max(times)),
    }
    # 이 값은 PyTorch가 추적한 할당량이며 전체 프로세스 VRAM과 다르다.
    if device.type == "cuda":
        result.update(
            {
                "gpu_total_gib": total / 2**30,
                "gpu_free_before_gib": free_before / 2**30,
                "torch_peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                "torch_peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
            }
        )
    del model, optimizer, observations, labels
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-dir", type=Path, default=Path("data/subgraphs"))
    parser.add_argument("--sizes", type=int, nargs="+", default=PILOT_NEURON_COUNTS)
    parser.add_argument("--modes", choices=["dense", "sparse"], nargs="+", default=["dense", "sparse"])
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=Path("data/phase0_probe.json"))
    args = parser.parse_args()
    if min(args.batch, args.steps, args.repeats) < 1:
        parser.error("batch, steps, and repeats must be positive")
    torch.manual_seed(42)
    results = []
    for size in args.sizes:
        graph_path = args.graph_dir / f"malecns_cx_{size}.npz"
        for mode in args.modes:
            try:
                result = benchmark(graph_path, mode, args)
            except (RuntimeError, NotImplementedError) as exc:
                result = {"graph": graph_path.name, "mode": mode, "error": str(exc)}
            results.append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

"""Measure board-size resource costs with synthetic recurrent PPO-sized batches.

This is a forward/backward/Adam diagnostic, not a Gomoku learning experiment.
It uses the same CNN, recurrent core, occupancy mask, and policy/value heads.
CUDA context, display and other processes are excluded from PyTorch counters.
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from .models import KINDS, Policy, count_parameters, masked_distribution, matched_hidden
from .train import atomic_json


def benchmark(kind: str, size: int, steps: int, args: argparse.Namespace) -> dict:
    gc.collect()
    torch.cuda.empty_cache()
    torch.manual_seed(8101)
    device = torch.device("cuda")
    model = Policy(kind, 8101, matched_hidden(kind, size), size, rewire_seed=1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0)
    # Realistic tensor shapes without claiming a valid game trajectory or skill gain.
    cells = torch.randint(0, 3, (steps, args.batch, size, size), device=device)
    obs = torch.zeros(steps, args.batch, 4, size, size, device=device)
    obs[:, :, 0] = cells == 1
    obs[:, :, 1] = cells == 2
    obs[:, :, 3] = torch.arange(args.batch, device=device)[None, :, None, None] % 2
    legal = (cells == 0).flatten(2)
    legal[:, :, 0] = True
    initial = model.initial(args.batch, device)
    resets = torch.zeros(steps, args.batch, dtype=torch.bool, device=device)
    resets[0] = True
    with torch.no_grad():
        logits, _ = model.sequence(obs, initial, resets)
        dist = masked_distribution(logits, legal)
        actions = dist.sample()
        old_log = dist.log_prob(actions).detach()
    del logits, dist
    adv = torch.randn(steps, args.batch, device=device)
    target = torch.randn(steps, args.batch, device=device)
    times = []
    # One warmup and several measured PPO updates, each with the pilot's two epochs.
    for repeat in range(args.repeats + 1):
        if repeat == 1:
            torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(2):
            logits, value = model.sequence(obs, initial, resets)
            dist = masked_distribution(logits, legal)
            ratio = (dist.log_prob(actions) - old_log).exp()
            loss = -torch.minimum(ratio * adv, ratio.clamp(0.8, 1.2) * adv).mean()
            loss = loss + 0.5 * F.mse_loss(value, target) - 0.01 * dist.entropy().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
        torch.cuda.synchronize()
        if repeat:
            times.append((time.perf_counter() - started) * 1000)
    return dict(kind=kind, board_size=size, batch=args.batch, recurrent_steps=steps,
                hidden=model.hidden, parameters=count_parameters(model),
                peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
                two_epoch_update_ms_median=statistics.median(times),
                graph_sha256=model.graph_sha256)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[9, 12, 15])
    parser.add_argument("--steps", nargs="+", type=int, default=[16, 64])
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "resource_benchmark.json")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA is required for VRAM measurements")
    if min(args.steps + [args.batch, args.repeats]) < 1 or min(args.sizes) < 5:
        parser.error("Invalid board size or batch/step/repeat budget")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    free, total = torch.cuda.mem_get_info()
    rows = []
    for size in args.sizes:
        for steps in args.steps:
            for kind in KINDS:
                row = benchmark(kind, size, steps, args)
                rows.append(row)
                print(json.dumps(row), flush=True)
    atomic_json(args.output, dict(date="2026-10-06", device=torch.cuda.get_device_name(),
                                 torch_version=str(torch.__version__),
                                 total_vram_gib=total / 2**30, driver_free_before_gib=free / 2**30,
                                 scope="synthetic training shapes; not complete game rollout or skill training",
                                 settings="one model at a time; FP32, TF32 off; AdamW, two PPO epochs",
                                 rows=rows))
    lines = ["# 오목 판 크기별 GPU 자원 진단", "", "2026-10-06. RTX 5060 Ti 16GB, 배치32, FP32, 모델을 하나씩 실행.",
             "같은 CNN·순환망·정책/가치 출력으로 합성 배치의 순전파·역전파·Adam 갱신을 측정했다.",
             "실제 12×12·15×15 오목 게임을 학습한 결과가 아니며 환경 수집과 전체 GPU 사용량을 포함하지 않는다.",
             "PyTorch 할당·예약 피크는 CUDA 컨텍스트·디스플레이·다른 프로세스의 VRAM을 제외한다.", "",
             "| 판 | 역전파 길이 | 모델별 최대 할당 범위 | 최대 예약 | 2epoch 계산 시간 범위 |",
             "|---|---:|---:|---:|---:|"]
    for size in args.sizes:
        for steps in args.steps:
            group = [r for r in rows if r["board_size"] == size and r["recurrent_steps"] == steps]
            alloc = [r["peak_allocated_gib"] for r in group]
            ms = [r["two_epoch_update_ms_median"] for r in group]
            lines.append(f"| {size}×{size} | {steps} | {min(alloc):.3f}–{max(alloc):.3f} GiB | "
                         f"{max(r['peak_reserved_gib'] for r in group):.3f} GiB | {min(ms):.1f}–{max(ms):.1f} ms |")
    lines += ["", "현재 규모의 12×12·15×15는 GPU 메모리 관점에서 충분히 여유가 있다. 큰 배치·긴 역전파·큰 모델·탐색은 별도 측정한다.",
              "현재 학습/평가 CLI는 9×9에 고정되어 있어 큰 판의 게임 학습에는 환경·착수 수·시작판·평가 루프를 함께 일반화해야 한다.",
              "판 크기에 묶인 CNN 압축층과 정책 출력층이 달라 9×9 체크포인트를 그대로 큰 판에 불러올 수 없다.",
              "기존 9×9 실제 게임 학습의 최대 할당은 Fly 0.056, 재배선 0.055, RNN 0.035, GRU 0.037 GiB였다."]
    args.output.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

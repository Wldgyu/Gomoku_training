"""MaleCNS 방향성 그래프의 뉴런별 입력·출력 연결 수를 보존하며 재배선한다.

두 간선 (a→b), (c→d)의 목적지를 맞바꿔 (a→d), (c→b)를 만든다.
자기 연결이나 중복 간선이 생기는 교환은 건너뛴다. 이 대조군은 연결
상대의 효과를 보려는 것으로, 실제 시냅스 수는 현재 게임 모델에 쓰지 않는다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np

from project_config import PILOT_NEURON_COUNTS


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rewire(path: Path, seed: int, swaps_per_edge: int) -> Path:
    """방향별 차수를 그대로 둔 채 충분한 간선 교환을 수행하고 검증한다."""
    graph = np.load(path)
    body_ids = graph["body_ids"].copy()
    pre = graph["pre_index"].astype(np.int32).copy()
    post = graph["post_index"].astype(np.int32).copy()
    weights = graph["synapse_count"].astype(np.int32).copy()
    n = len(body_ids)
    m = len(pre)
    original_pairs = {(int(a), int(b)) for a, b in zip(pre, post, strict=True)}
    if len(original_pairs) != m:
        raise ValueError("원본 그래프에 중복 간선이 있습니다")
    pairs = set(original_pairs)
    rng = random.Random(seed)
    target_swaps = swaps_per_edge * m
    accepted = 0
    attempted = 0
    attempt_limit = target_swaps * 20
    while accepted < target_swaps and attempted < attempt_limit:
        attempted += 1
        i = rng.randrange(m)
        j = rng.randrange(m)
        if i == j:
            continue
        a, b = int(pre[i]), int(post[i])
        c, d = int(pre[j]), int(post[j])
        new_a = (a, d)
        new_c = (c, b)
        if a == c or b == d or a == d or c == b:
            continue
        if new_a in pairs or new_c in pairs:
            continue
        pairs.remove((a, b))
        pairs.remove((c, d))
        pairs.add(new_a)
        pairs.add(new_c)
        post[i], post[j] = d, b
        accepted += 1

    if accepted < target_swaps:
        raise RuntimeError(f"교환 부족: {accepted}/{target_swaps} (시도 {attempted})")
    if len(pairs) != m or any(a == b for a, b in pairs):
        raise AssertionError("재배선 결과에 중복 또는 자기 연결이 생겼습니다")
    # 목적지만 교환했으므로 출력 차수는 구조상 같지만 두 방향을 모두 확인한다.
    assert np.array_equal(np.bincount(pre, minlength=n), np.bincount(graph["pre_index"], minlength=n))
    assert np.array_equal(np.bincount(post, minlength=n), np.bincount(graph["post_index"], minlength=n))
    overlap = len(pairs & original_pairs)
    output = path.with_name(f"{path.stem}_rewired_seed{seed}.npz")
    np.savez_compressed(
        output,
        body_ids=body_ids,
        pre_index=pre,
        post_index=post,
        synapse_count=weights,
    )
    metadata = {
        "source_graph": path.name,
        "source_graph_sha256": sha256(path),
        "rewire_seed": seed,
        "neurons": n,
        "edges": m,
        "target_accepted_swaps": target_swaps,
        "accepted_swaps": accepted,
        "attempted_swaps": attempted,
        "directed_in_degrees_preserved": True,
        "directed_out_degrees_preserved": True,
        "self_edges": 0,
        "duplicate_edges": 0,
        "edge_overlap_with_real": overlap,
        "edge_overlap_fraction": overlap / m,
        "weight_values_retained": True,
        "weight_note": "synapse_count is unused by the current game model",
        "file": output.name,
    }
    output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False), flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-dir", type=Path, default=Path("data/subgraphs"))
    parser.add_argument("--sizes", type=int, nargs="+", default=PILOT_NEURON_COUNTS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--swaps-per-edge", type=int, default=10)
    args = parser.parse_args()
    if args.swaps_per_edge < 1:
        parser.error("swaps-per-edge must be positive")
    for size in args.sizes:
        rewire(args.graph_dir / f"malecns_cx_{size}.npz", args.seed, args.swaps_per_edge)


if __name__ == "__main__":
    main()

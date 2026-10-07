"""원본 연결 전체를 RAM에 올리지 않고 1,000개·2,000개 MaleCNS 부분 그래프를 만든다.

뉴런 선정에는 해부학적 주석과 연결 강도만 사용한다. 게임 성적은 보지 않으며,
저장하는 간선은 원본의 뉴런 간 신호 방향을 그대로 따른다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.feather as feather
import pyarrow.ipc as ipc

from project_config import PILOT_NEURON_COUNTS

ANNOTATIONS = "body-annotations-male-cns-v1.0-minconf-0.5.feather"
CONNECTIVITY = "connectome-weights-male-cns-v1.0-minconf-0.5.feather"
SOURCE_URL = "https://male-cns.janelia.org/download/"


def sha256(path: Path) -> str:
    """큰 원본 파일도 작은 블록으로 읽어 재현성 확인용 해시를 계산한다."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def largest_scc_size(node_count: int, pre: np.ndarray, post: np.ndarray) -> int:
    """방향성 그래프에서 서로 왕복 경로가 있는 최대 뉴런 집합의 크기를 구한다."""
    outgoing = [[] for _ in range(node_count)]
    incoming = [[] for _ in range(node_count)]
    for a, b in zip(pre.tolist(), post.tolist(), strict=True):
        outgoing[a].append(b)
        incoming[b].append(a)

    # 첫 순회에서는 출발 뉴런마다 탐색을 끝낸 순서를 기록한다.
    seen = set()
    finish = []
    for root in range(node_count):
        if root in seen:
            continue
        seen.add(root)
        stack = [(root, 0)]
        while stack:
            node, next_index = stack[-1]
            if next_index == len(outgoing[node]):
                finish.append(node)
                stack.pop()
                continue
            neighbor = outgoing[node][next_index]
            stack[-1] = (node, next_index + 1)
            if neighbor not in seen:
                seen.add(neighbor)
                stack.append((neighbor, 0))

    # 간선 방향을 거꾸로 따라가며 강연결요소의 크기를 센다.
    seen.clear()
    largest = 0
    for root in reversed(finish):
        if root in seen:
            continue
        size = 0
        stack = [root]
        seen.add(root)
        while stack:
            node = stack.pop()
            size += 1
            for neighbor in incoming[node]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        largest = max(largest, size)
    return largest


def main() -> None:
    """주석으로 후보를 정하고, 연결을 배치별로 읽어 두 크기의 그래프를 저장한다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/subgraphs"))
    parser.add_argument("--sizes", type=int, nargs="+", default=PILOT_NEURON_COUNTS)
    parser.add_argument("--min-synapses", type=int, default=3)
    args = parser.parse_args()

    sizes = sorted(set(args.sizes))
    if not sizes or sizes[0] < 1 or args.min_synapses < 1:
        parser.error("sizes and min-synapses must be positive")

    annotation_path = args.data_dir / ANNOTATIONS
    connectivity_path = args.data_dir / CONNECTIVITY
    # CX로 분류된 중앙뇌 내부 뉴런만 초기 후보로 사용한다.
    annotations = feather.read_table(
        annotation_path, columns=["bodyId", "superclass", "class"]
    )
    eligible = pc.and_(
        pc.equal(annotations["superclass"], "cb_intrinsic"),
        pc.equal(annotations["class"], "CX"),
    )
    body_ids = np.unique(
        annotations.filter(eligible)["bodyId"].to_numpy().astype(np.int64)
    )
    if len(body_ids) < sizes[-1]:
        raise ValueError(f"only {len(body_ids)} eligible neurons")
    print(f"Eligible annotated CX neurons: {len(body_ids):,}", flush=True)

    # 원본 연결 테이블은 매우 크므로 한 번에 읽지 않고 Arrow 배치별로 거른다.
    value_set = pa.array(body_ids)
    pre_chunks: list[np.ndarray] = []
    post_chunks: list[np.ndarray] = []
    weight_chunks: list[np.ndarray] = []
    with pa.memory_map(str(connectivity_path), "r") as source:
        reader = ipc.open_file(source)
        if reader.schema.names[:3] != ["body_pre", "body_post", "weight"]:
            raise ValueError(f"unexpected edge schema: {reader.schema.names}")
        for batch_number in range(reader.num_record_batches):
            batch = reader.get_batch(batch_number)
            mask = pc.and_(
                pc.is_in(batch.column("body_pre"), value_set=value_set),
                pc.is_in(batch.column("body_post"), value_set=value_set),
            )
            chosen = batch.filter(mask)
            if chosen.num_rows:
                pre = chosen.column("body_pre").to_numpy().astype(np.int64)
                post = chosen.column("body_post").to_numpy().astype(np.int64)
                weight = chosen.column("weight").to_numpy().astype(np.int64)
                # 약한 연결과 자기 자신으로 향하는 연결은 제외한다.
                keep = (weight >= args.min_synapses) & (pre != post)
                if keep.any():
                    pre_chunks.append(pre[keep])
                    post_chunks.append(post[keep])
                    weight_chunks.append(weight[keep])
            if (batch_number + 1) % 250 == 0:
                print(f"Read {batch_number + 1}/{reader.num_record_batches} batches", flush=True)

    pre_ids = np.concatenate(pre_chunks)
    post_ids = np.concatenate(post_chunks)
    weights = np.concatenate(weight_chunks)
    pre_pos = np.searchsorted(body_ids, pre_ids)
    post_pos = np.searchsorted(body_ids, post_ids)
    # 후보 집단 안에서 주고받는 시냅스 수의 합을 선정 점수로 삼는다.
    scores = np.bincount(pre_pos, weights=weights, minlength=len(body_ids))
    scores += np.bincount(post_pos, weights=weights, minlength=len(body_ids))
    # 점수가 같으면 bodyId가 작은 뉴런을 먼저 골라 결과가 항상 같게 한다.
    ranking = np.lexsort((body_ids, -scores))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    provenance = {
        "dataset": "MaleCNS v1.0",
        "source_url": SOURCE_URL,
        "annotation_sha256": sha256(annotation_path),
        "connectivity_sha256": sha256(connectivity_path),
        "candidate_superclass": "cb_intrinsic",
        "candidate_class": "CX",
        "candidate_neurons": len(body_ids),
        "minimum_synapses": args.min_synapses,
        "selection": "descending internal weighted degree, bodyId ascending tie break",
        "self_edges_removed": True,
    }
    for size in sizes:
        selected = body_ids[ranking[:size]]
        keep = np.isin(pre_ids, selected) & np.isin(post_ids, selected)
        # 순위가 곧 그래프 인덱스이므로 1,000개 집합은 2,000개 집합에 포함된다.
        sorted_order = np.argsort(selected)
        sorted_ids = selected[sorted_order]
        pre_index = sorted_order[np.searchsorted(sorted_ids, pre_ids[keep])]
        post_index = sorted_order[np.searchsorted(sorted_ids, post_ids[keep])]
        edge_weights = weights[keep]
        output = args.output_dir / f"malecns_cx_{size}.npz"
        np.savez_compressed(
            output,
            body_ids=selected,
            pre_index=pre_index.astype(np.int32),
            post_index=post_index.astype(np.int32),
            synapse_count=edge_weights.astype(np.int32),
        )
        connected = np.unique(np.concatenate((pre_index, post_index)))
        metadata = {
            **provenance,
            "neurons": size,
            "edges": int(len(edge_weights)),
            "directed_density": float(len(edge_weights) / (size * (size - 1))),
            "isolated_neurons": int(size - len(connected)),
            "largest_strongly_connected_component": largest_scc_size(
                size, pre_index, post_index
            ),
            "total_synapses": int(edge_weights.sum()),
            "file": output.name,
        }
        output.with_suffix(".json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(metadata, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

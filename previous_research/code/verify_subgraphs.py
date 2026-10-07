"""추출 그래프를 MaleCNS 원본 표와 독립적으로 다시 대조한다."""

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


def digest(path: Path) -> str:
    """원본 파일이 추출 당시 파일과 같은지 확인할 SHA-256을 계산한다."""
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def open_graph(path: Path) -> tuple[np.ndarray, dict[tuple[int, int], int], dict]:
    """저장 파일의 인덱스·중복·가중치 범위를 점검하고 bodyId 간선으로 복원한다."""
    graph = np.load(path)
    meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    ids = graph["body_ids"]
    pre = graph["pre_index"]
    post = graph["post_index"]
    weights = graph["synapse_count"]
    n = len(ids)
    assert n == meta["neurons"] == int(path.stem.rsplit("_", 1)[-1])
    assert len(np.unique(ids)) == n, "duplicate neurons"
    assert len(pre) == len(post) == len(weights) == meta["edges"]
    assert np.all((0 <= pre) & (pre < n))
    assert np.all((0 <= post) & (post < n))
    assert np.all(pre != post), "self edges present"
    assert np.all(weights >= meta["minimum_synapses"])
    edges = {}
    for a, b, w in zip(ids[pre], ids[post], weights, strict=True):
        pair = (int(a), int(b))
        assert pair not in edges, "duplicate edges"
        edges[pair] = int(w)
    assert sum(edges.values()) == meta["total_synapses"]
    return ids, edges, meta


def main() -> None:
    """두 그래프의 출처, 선정 규칙, 모든 간선을 원본과 비교한다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--graph-dir", type=Path, default=Path("data/subgraphs"))
    parser.add_argument("--sizes", type=int, nargs=2, default=PILOT_NEURON_COUNTS)
    args = parser.parse_args()
    small, large = sorted(args.sizes)
    if small < 1 or small == large:
        parser.error("two distinct positive sizes are required")

    graphs = {}
    for size in (small, large):
        graphs[size] = open_graph(args.graph_dir / f"malecns_cx_{size}.npz")
    assert np.array_equal(graphs[small][0], graphs[large][0][:small]), "graphs not nested"

    annotation_path = args.data_dir / ANNOTATIONS
    connectivity_path = args.data_dir / CONNECTIVITY
    annotation_hash = digest(annotation_path)
    connectivity_hash = digest(connectivity_path)
    for _, _, meta in graphs.values():
        assert meta["dataset"] == "MaleCNS v1.0"
        assert meta["annotation_sha256"] == annotation_hash, "annotation hash mismatch"
        assert meta["connectivity_sha256"] == connectivity_hash, "edge source hash mismatch"
        assert meta["candidate_class"] == "CX"
        assert meta["candidate_superclass"] == "cb_intrinsic"
        assert meta["minimum_synapses"] == 3
    print("PASS: official source file hashes match graph metadata", flush=True)

    annotations = feather.read_table(
        annotation_path, columns=["bodyId", "superclass", "class"]
    )
    eligible = pc.and_(
        pc.equal(annotations["superclass"], "cb_intrinsic"),
        pc.equal(annotations["class"], "CX"),
    )
    candidate_ids = np.unique(
        annotations.filter(eligible)["bodyId"].to_numpy().astype(np.int64)
    )
    assert len(candidate_ids) == graphs[large][2]["candidate_neurons"]
    assert np.isin(graphs[large][0], candidate_ids).all()
    print("PASS: all selected neuron IDs have the expected MaleCNS annotations", flush=True)

    # 약 1.5억 원본 연결 행을 한꺼번에 RAM에 올리지 않고 선정 점수를 재계산한다.
    # 원본을 다시 읽기 때문에 생성 코드와 별개의 검증이 된다.
    value_set = pa.array(candidate_ids)
    scores = np.zeros(len(candidate_ids), dtype=np.int64)
    with pa.memory_map(str(connectivity_path), "r") as source:
        reader = ipc.open_file(source)
        for index in range(reader.num_record_batches):
            batch = reader.get_batch(index)
            mask = pc.and_(
                pc.is_in(batch.column("body_pre"), value_set=value_set),
                pc.is_in(batch.column("body_post"), value_set=value_set),
            )
            chosen = batch.filter(mask)
            if not chosen.num_rows:
                continue
            pre = chosen.column("body_pre").to_numpy()
            post = chosen.column("body_post").to_numpy()
            weight = chosen.column("weight").to_numpy()
            keep = (weight >= 3) & (pre != post)
            if keep.any():
                scores += np.bincount(
                    np.searchsorted(candidate_ids, pre[keep]),
                    weights=weight[keep],
                    minlength=len(candidate_ids),
                ).astype(np.int64)
                scores += np.bincount(
                    np.searchsorted(candidate_ids, post[keep]),
                    weights=weight[keep],
                    minlength=len(candidate_ids),
                ).astype(np.int64)
    order = np.lexsort((candidate_ids, -scores))
    assert np.array_equal(graphs[large][0], candidate_ids[order[:large]])
    print(f"PASS: {small}/{large} neurons follow the recorded deterministic selection rule", flush=True)

    # 두 번째 순회에서는 선정된 뉴런 사이의 실제 간선을 모아 저장 결과와 대조한다.
    selected_ids = graphs[large][0]
    value_set = pa.array(selected_ids)
    expected = {}
    with pa.memory_map(str(connectivity_path), "r") as source:
        reader = ipc.open_file(source)
        for index in range(reader.num_record_batches):
            batch = reader.get_batch(index)
            mask = pc.and_(
                pc.is_in(batch.column("body_pre"), value_set=value_set),
                pc.is_in(batch.column("body_post"), value_set=value_set),
            )
            chosen = batch.filter(mask)
            if not chosen.num_rows:
                continue
            pre = chosen.column("body_pre").to_numpy()
            post = chosen.column("body_post").to_numpy()
            weight = chosen.column("weight").to_numpy()
            for a, b, w in zip(pre, post, weight, strict=True):
                if a != b and w >= 3:
                    pair = (int(a), int(b))
                    assert pair not in expected, "duplicate source edges"
                    expected[pair] = int(w)
    assert graphs[large][1] == expected, f"{large}-node edge list differs from source"
    smaller_ids = set(graphs[small][0].tolist())
    expected_small = {
        pair: weight
        for pair, weight in expected.items()
        if pair[0] in smaller_ids and pair[1] in smaller_ids
    }
    assert graphs[small][1] == expected_small, f"{small}-node edge list differs from source"
    print("PASS: every saved edge direction and synapse count matches the source", flush=True)
    print("PASS: graph extraction verified; this does not test game learning", flush=True)


if __name__ == "__main__":
    main()

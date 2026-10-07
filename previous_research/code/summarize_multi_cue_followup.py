"""Summarize completed three-stage multi-cue experiments without selecting by test scores."""

from __future__ import annotations

import hashlib
import itertools
import json
import shutil
import statistics
import sys
from datetime import datetime

import torch

from multi_cue_followup import CONDITIONS, OUT, ROOT, SEEDS, atomic_json

KINDS = ("fly", "rewired", "rnn", "gru", "leaky")
NAMES = {
    "fly": "실제 Fly",
    "rewired": "재배선 Fly",
    "rnn": "RNN",
    "gru": "GRU",
    "leaky": "누설 기억",
}


def records(folder):
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(folder.glob("*.json"))
    ]


def stats(values):
    return {
        "mean": statistics.mean(values),
        "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "values": values,
    }


def fmt(values):
    s = stats(values)
    return f"{100 * s['mean']:.2f} ± {100 * s['sample_std']:.2f}%"


def main():
    baseline = records(OUT / "baseline")
    frozen = records(OUT / "frozen")
    diagnosis = records(OUT / "diagnosis")
    load = records(OUT / "load")
    visual = records(OUT / "visual" / "main")
    tuning = json.loads((OUT / "visual" / "tuning.json").read_text())
    if (
        len(baseline),
        len(frozen),
        len(diagnosis),
        len(load),
        len(visual),
        len(tuning),
    ) != (27, 3, 15, 135, 15, 30):
        raise ValueError("incomplete stage counts")
    base_means = {}
    paired = []
    lines = [
        "다중 단서 기억 후속 연구: 1→2→3 결과 보고서",
        "보고서 생성: "
        + datetime.now().astimezone().isoformat()
        + ". 프로토콜 작성 2026-10-04.",
        "",
        "1. 기준 조건 재현 (단서2·지연4·방해2, 16,000 업데이트)",
        "검증1024, 시험4096, 초기화1101~1103. 평균±표본 표준편차, 단위 %.",
        "재배선은 초기화마다 그래프0~4 평균을 먼저 구한 뒤 3초기화로 요약한다.",
    ]
    for kind in KINDS:
        group = [r for r in baseline if r["kind"] == kind]
        values = [
            statistics.mean(r["test"]["accuracy"] for r in group if r["seed"] == s)
            for s in SEEDS
        ]
        base_means[kind] = stats(values)
        first = [
            statistics.mean(
                r["test"]["by_slot"]["0"]["accuracy"] for r in group if r["seed"] == s
            )
            for s in SEEDS
        ]
        last = [
            statistics.mean(
                r["test"]["by_slot"]["1"]["accuracy"] for r in group if r["seed"] == s
            )
            for s in SEEDS
        ]
        lines.append(
            f"{NAMES[kind]}: 전체 {fmt(values)}, 첫 단서 {fmt(first)}, 마지막 단서 {fmt(last)}"
        )
        lines.append(
            "  초기화별 전체: "
            + ", ".join(f"{s}={v * 100:.2f}%" for s, v in zip(SEEDS, values))
        )
    for seed in SEEDS:
        actual = next(r for r in baseline if r["kind"] == "fly" and r["seed"] == seed)
        rewired = [r for r in baseline if r["kind"] == "rewired" and r["seed"] == seed]
        diff = actual["test"]["accuracy"] - statistics.mean(
            r["test"]["accuracy"] for r in rewired
        )
        paired.append(diff)
    lines.extend(
        [
            "실제 Fly - 5재배선 평균의 초기화별 차이(pp): "
            + ", ".join(f"{d * 100:+.2f}" for d in paired),
            f"차이 평균 {statistics.mean(paired) * 100:+.2f}pp. 초기화3개이므로 탐색적 결과다.",
            "",
            "시드/그래프별 전체·첫 단서·마지막 단서 및 검증 90%/99% 최초 도달:",
        ]
    )
    for r in baseline:
        t = r["test"]
        lines.append(
            f"{r['name']}: {t['accuracy'] * 100:.2f}/{t['by_slot']['0']['accuracy'] * 100:.2f}/"
            f"{t['by_slot']['1']['accuracy'] * 100:.2f}%, to90={r['to90'] if r['to90'] is not None else '>16000'}, "
            f"to99={r['to99'] if r['to99'] is not None else '>16000'}"
        )
    lines.extend(
        ["", "단서 종류만 바꾸는 대응 평가 (시험 선택이나 학습에 사용하지 않음):"]
    )
    for kind in KINDS:
        group = [r for r in baseline if r["kind"] == kind]
        lines.append(
            f"{NAMES[kind]}: 질문 대상 변경 정답률 "
            f"{statistics.mean(r['test']['queried_kind_change_accuracy'] for r in group) * 100:.2f}%, "
            f"비질문 대상 변경 후 정답률 "
            f"{statistics.mean(r['test']['unqueried_kind_change_accuracy'] for r in group) * 100:.2f}%, "
            f"비질문 대상 변경 예측 불변율 "
            f"{statistics.mean(r['test']['unqueried_kind_change_invariance'] for r in group) * 100:.2f}%"
        )
    lines.extend(
        [
            "",
            "2. 원인 진단",
            "간선 학습/초기 간선 고정/간선0 고정을 같은 초기화·배치·예산으로 비교했다.",
        ]
    )
    for label, group in (
        ("간선 학습", [r for r in baseline if r["kind"] == "fly"]),
        ("초기 간선 고정", frozen),
        ("간선0 고정", [r for r in baseline if r["kind"] == "leaky"]),
    ):
        lines.append(f"{label}: {fmt([r['test']['accuracy'] for r in group])}")
    probe_means = {}
    for kind, is_frozen in (
        ("fly", False),
        ("fly", True),
        ("leaky", False),
        ("rnn", False),
        ("gru", False),
    ):
        label = "초기 간선 고정" if is_frozen else NAMES[kind]
        group = [
            r
            for r in diagnosis
            if r["run"]["kind"] == kind and r["run"]["frozen"] == is_frozen
        ]
        lines.append(f"{label}: 상태 판독 전체 제시 색 정확도 (선형/MLP)")
        lines.append(
            "  탐색적 질문 색 교체: 두 질문 모두 정답 "
            f"{statistics.mean(r['query_switch']['both_queries_correct'] for r in group) * 100:.2f}%, "
            "물체 종류가 다른 경우 예측 변경 "
            f"{statistics.mean(r['query_switch']['prediction_changes_when_kinds_differ'] for r in group) * 100:.2f}%"
        )
        for row in sorted(group, key=lambda r: r["run"]["seed"]):
            switch = row["query_switch"]
            lines.append(
                f"    seed{row['run']['seed']}: 두 질문 모두 정답 "
                f"{switch['both_queries_correct'] * 100:.2f}%, 다른 종류의 질문으로 변경 시 예측 변경 "
                f"{switch['prediction_changes_when_kinds_differ'] * 100:.2f}%"
            )
        for phase in ("cue_end", "delay_end", "query_end"):
            values = []
            for probe in ("linear", "mlp"):
                ps = [
                    next(
                        p
                        for p in r["probes"]
                        if p["phase"] == phase and p["probe"] == probe
                    )
                    for r in group
                ]
                vals = [p["accuracy_all_present_colors"] for p in ps]
                probe_means[f"{label}/{phase}/{probe}"] = stats(vals)
                values.append(
                    f"{probe}={fmt(vals)} (첫/마지막 "
                    f"{statistics.mean(p['by_cue_slot']['0'] for p in ps) * 100:.2f}/"
                    f"{statistics.mean(p['by_cue_slot']['1'] for p in ps) * 100:.2f}%)"
                )
            lines.append(f"  {phase}: " + "; ".join(values))
    lines.append("학습된 Fly에 추론 시점 간선 개입:")
    for r in diagnosis:
        if r["interventions"]:
            i = r["interventions"]
            lines.append(
                f"  seed{r['run']['seed']}: 간선0 {i['zero_edges_at_inference']['accuracy'] * 100:.2f}%, "
                f"초기 간선 {i['initial_edges_at_inference']['accuracy'] * 100:.2f}%, "
                f"학습 간선 RMS 변화={i['learned_edge_delta_rms']:.6f}"
            )
    heads = json.loads((OUT / "head_inspection.json").read_text())
    lines.append("탐색적 출력층 점검 (모델을 수정하거나 다시 학습하지 않음):")
    for row in heads:
        lines.append(
            f"  {row['name']}: 시험4096개에서 한 번도 활성화되지 않은 ReLU "
            f"{row['relu_units_never_active']}/128, 예측 종류별 개수={row['predicted_class_counts']}"
        )
    lines.append(
        "GRU seed1102의 출력층 ReLU 128개가 시험 전체에서 모두 비활성이고 같은 종류만 답했다."
    )
    lines.append(
        "이 시드의 별도 MLP 상태 판독은 약99.9%라, 현재 출력층의 붕괴와 기억 정보의 소실을 구분한다."
    )
    lines.extend(
        [
            "판독기는 원래 모델을 고정한 채 별도 데이터로 학습했다. 특징은 학습 데이터 평균/표준편차로",
            "정규화했다(표준편차 하한1e-4). 판독 성공은 원래 정책이 정보를 이용한다는 뜻이 아니며,",
            "판독 실패는 모든 비선형 정보가 사라졌다는 뜻도 아니다. 간선 제거는 자기 상태 누설은 유지한다.",
            "",
            "3a. 9조건 기억 부하 비교 (각 12,000 업데이트)",
            "조건: 단서 수/지연/방해. 재배선은 초기화별 그래프1/2/3 하나씩이며 그래프 변동과 분리되지 않는다.",
        ]
    )
    load_means = {}
    majority_theory = {}
    for n in (1, 2, 3, 4):
        possibilities = list(itertools.product(range(3), repeat=n))
        majority_theory[n] = statistics.mean(
            max(kinds.count(k) for k in range(3)) / n for kinds in possibilities
        )
    lines.append(
        "질문 색을 무시하고 가장 많은 물체 종류만 답하는 이론 기준선: "
        + ", ".join(f"단서{n}={v * 100:.2f}%" for n, v in majority_theory.items())
    )
    for n, d, k in CONDITIONS:
        task = f"cues{n}_delay{d}_distract{k}"
        load_means[task] = {}
        entries = []
        for kind in KINDS:
            group = [r for r in load if r["task"] == task and r["kind"] == kind]
            if len(group) != 3:
                raise ValueError("missing load runs")
            vals = [r["test"]["accuracy"] for r in group]
            load_means[task][kind] = stats(vals)
            entries.append(f"{NAMES[kind]} {fmt(vals)}")
        lines.append(f"{n}/{d}/{k}: " + " | ".join(entries))
    lines.extend(
        [
            "기준2/4/2는 1단계의12,000회 체크포인트를 재사용했다.",
            "단서 순서별 정확도·학습 곡선·가중치는 각 실행 JSON/PT에 보존했다.",
            "",
            "3b. 방·복도 MiniGrid 관측 비교",
            "단서2·방해2. 학습 복도6/10, 미사용 복도8/14. 고정 전진 경로, 별도 CNN+기억 학습.",
            "2048개·128업데이트 조정 후, 새 초기화3개·학습8192개·1280업데이트로 최종 비교했다.",
            "검증으로 체크포인트를 선택하고 그룹별2048개 시험. 대기 조건은 미사용 길이에 확률0.5 대기.",
            "선택 학습률: "
            + json.dumps(
                json.loads((OUT / "visual" / "selected_rates.json").read_text())
            ),
        ]
    )
    visual_means = {}
    for kind in KINDS:
        group = [r for r in visual if r["kind"] == kind]
        visual_means[kind] = {}
        entries = []
        for name in ("trained_delays", "unseen_delays", "unseen_delays_wait"):
            vals = [r["tests"][name]["accuracy"] for r in group]
            visual_means[kind][name] = stats(vals)
            entries.append(f"{name} {fmt(vals)}")
        lines.append(f"{NAMES[kind]}: " + " | ".join(entries))
    lines.extend(
        [
            "",
            "해석 범위 및 다음 연구",
            "같은 3초기화에 여러 모델을 짝지었으나 3회만으로 안정적인 모집단 우위를 확정하지 않는다.",
            "시드는 초기 파라미터와 학습 배치의 난수를 함께 정한다. 변동을 초기 파라미터만의 효과로 단정하지 않는다.",
            "생성 시드를 분리했지만 유한한 과제에서 동일 내용의 에피소드가 학습과 시험에 다시 나올 수 있다.",
            "9조건은 각 조건에서 새로 학습한 비교이며 단서 수가 다른 조건으로 가중치를 직접 전이한 시험이 아니다.",
            "파라미터 수는 RNN/GRU를 Fly에 맞췄고 누설 기억은 간선 제거 대조로 더 적다.",
            "새 학습 배치 생성기의 RNG 순서는 기존과 다르므로 이전 실행의 수치적 재현이 아닌 분포 재현이다.",
            "여러 모델 동시 계산은 단일 모델 출력·기울기·한 AdamW 업데이트와 수치 오차 내 동등함을 검증했다.",
            "긴 예산과 부하 비교의 예산 차이(16000/12000)를 구분하고 임의의 최고 시험 시점을 선택하지 않았다.",
            "방·복도는 고정 경로 관측 과제이며 자율 탐색/제어 성능이나 시퀀스 가중치 직접 전이를 증명하지 않는다.",
            "시각 과제는 적은 고정 예제를 반복 학습하므로 기호 시퀀스와 성능 수치를 직접 비교하지 않는다.",
            "시각 관측은 RGB 사진이 아닌 MiniGrid 물체/색/상태 채널이다. 초기 화면에 두 단서가 함께 보일 수 있다.",
            "실제 Fly의 평균 우위는 기준 조건의 한 시드에서만 생겼다. 부하 조건과 방·복도에서도 재배선보다 일관되게 높지 않았다.",
            "실패한 기준 Fly 두 시드는 질문 색만 바꿔도 예측이 바뀌지 않았다. 약66.7%만으로 마지막 단서만 기억한다고 판단하지 않는다.",
            "방·복도의 실제 Fly와 누설 기억은 약66~67%다. 질문을 무시하는 이론 기준선에 가깝지만 시각 과제의 전략을 직접 진단한 결과는 아니다.",
            "최종 시험으로 모델 구조/학습률을 바꾸지 않았다. 개선 연구는 별도 프로토콜·새 시험 데이터가 필요하다.",
            "오목과 5,000개 확장은 계속 보류한다.",
            "",
            "다음 진행 권장 순서 (이번 완료 범위 이후의 제안)",
            "1) 출력층 안정화: 저장된 기억 모듈을 고정하고 질문 조건을 받는 출력층만 다시 학습해 고원/붕괴 원인을 분리한다.",
            "   ReLU 대조와 비활성 구간에도 기울기가 흐르는 출력층을 모든 모델에 같은 방식으로 비교한다.",
            "2) 방·복도 인식/기억 분리: 실제 부분관측 화면에 보이는 단서와 질문의 색·종류 판독을 먼저 평가한다.",
            "   동일 화면에서 추출한 정답 특징 대조로 인식 실패와 기억·질문 선택 실패를 구분한다. 보이지 않는 지도 정보는 입력하지 않는다.",
            "3) 질문 선택이 안정된 뒤 지연16·방해6 조건을 실제 Fly/재배선5개/간선0 대조와 새 시드5개 이상으로 반복한다.",
            "   추가 기억 게이트를 도입한다면 모든 모델에 공통으로 적용하고 별도 조정 데이터·예산·시험을 사전에 고정한다.",
            "",
            "검증·재현",
            "기존14개+동시 계산(CPU/CUDA)/CUDA재실행/데이터/체크포인트/질문 교체6개 테스트 통과. 소스와 프로토콜은 source/에 보존.",
            "기준27, 고정 간선3, 상태 판독90, 부하135(기준15개 재사용), 시각 조정30·최종15회 완료.",
        ]
    )
    lines[3:3] = [
        "요약",
        "기준 조건 시험 평균: "
        + ", ".join(
            f"{NAMES[k]} {v['mean'] * 100:.2f}%" for k, v in base_means.items()
        ),
        f"실제 Fly - 초기화별 재배선5개 평균: {statistics.mean(paired) * 100:+.2f}pp (초기화3개).",
        f"실제 Fly의 질문 후 상태 MLP 판독 평균: {probe_means['실제 Fly/query_end/mlp']['mean'] * 100:.2f}%.",
        "학습 도중 최고 검증 점수와 마지막 모델의 성능은 구분한다. 고원과 학습 불안정도 결과에 포함했다.",
        "",
    ]
    report = ROOT / "reports" / "memory" / "multi_cue_followup_results.txt"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    sources = (
        "multi_cue_memory.py",
        "multi_cue_followup.py",
        "diagnose_multi_cue.py",
        "inspect_multi_cue_heads.py",
        "multi_cue_visual_followup.py",
        "grid_multi_cue.py",
        "run_multi_cue_followup.py",
        "summarize_multi_cue_followup.py",
        "test_multi_cue_followup.py",
        "multi_cue_followup_protocol.txt",
        "multi_cue_visual_protocol.txt",
    )
    source = OUT / "source"
    source.mkdir(exist_ok=True)
    hashes = {}
    for name in sources:
        path = (ROOT.parent if name.endswith("_protocol.txt") else ROOT / "code") / name
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        shutil.copy2(path, source / name)
    graph_hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (ROOT / "data" / "subgraphs").glob("malecns_cx_1000*.npz")
    }
    atomic_json(
        OUT / "analysis.json",
        {
            "baseline": base_means,
            "paired_fly_minus_rewired": paired,
            "probe_means": probe_means,
            "load": load_means,
            "visual": visual_means,
            "code_sha256": hashes,
            "graph_sha256": graph_hashes,
            "python": sys.version,
            "torch": torch.__version__,
            "device": torch.cuda.get_device_name(),
        },
    )
    print(
        json.dumps(
            {"report": str(report), "baseline": base_means, "visual": visual_means}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

"""Select the intervention from tuning runs and summarize the final runs.

  select  -> data/multi_cue_binding/selection.json + tuning table (tuning seeds only)
  report  -> multi_cue_binding_results.txt (final seeds, written after the final stage)
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from multi_cue_binding import VARIANTS, without_gate

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "multi_cue_binding"
KIND_LABEL = {"fly": "실제 Fly", "rewired": "재배선 Fly", "rnn": "RNN", "gru": "GRU",
              "leaky": "누설 기억"}


def load(stage: str) -> list[dict]:
    return [json.loads(path.read_text(encoding="utf-8"))
            for path in sorted((DATA / stage).glob("*.json"))]


def mean(values):
    return statistics.fmean(values)


def sd(values):
    return statistics.stdev(values) if len(values) > 1 else 0.0


def condition(row) -> str:
    return f"단서{row['cues']}·지연{row['delay']}·방해{row['distractors']}"


def select() -> None:
    rows = load("tune")
    graph = defaultdict(list)
    table = defaultdict(list)
    for row in rows:
        table[(row["kind"], row["variant"], condition(row))].append(row["final_validation"])
        if row["kind"] in ("fly", "leaky"):
            graph[row["variant"]].append(row["final_validation"])
    scores = {variant: mean(values) for variant, values in graph.items()}
    best = max(scores.values())
    near = [v for v, s in scores.items() if s >= best - 0.01]
    chosen = min(near, key=lambda v: (sum(VARIANTS[v]), -scores[v]))
    lines = ["조정 결과 (시드 901·902, 검증 정확도 평균, 6,000 업데이트)", ""]
    lines.append("그래프 계열(실제 Fly+누설) 변형별 평균 검증 정확도:")
    for variant, score in sorted(scores.items(), key=lambda item: -item[1]):
        lines.append(f"  {variant}: {score * 100:.2f}% (n={len(graph[variant])})")
    lines.append(f"선택: {chosen} (최고 {best * 100:.2f}%, 1pp 이내 후보 {near})")
    lines.append("")
    lines.append("모델·변형·조건별 평균 검증 정확도(두 시드 값):")
    for (kind, variant, cond), values in sorted(table.items()):
        lines.append(f"  {KIND_LABEL[kind]} / {variant} / {cond}: {mean(values) * 100:.2f}% "
                     f"({', '.join(f'{v * 100:.1f}' for v in values)})")
    text = "\n".join(lines) + "\n"
    (DATA / "selection.json").write_text(json.dumps(
        {"variant": chosen, "scores": scores, "rnn_gru_variant": without_gate(chosen)},
        indent=1) + "\n", encoding="utf-8")
    (DATA / "tuning_table.txt").write_text(text, encoding="utf-8")
    print(text)


def fmt(values) -> str:
    return f"{mean(values) * 100:.2f} ± {sd(values) * 100:.2f}%"


def report() -> None:
    rows = load("final")
    selection = json.loads((DATA / "selection.json").read_text(encoding="utf-8"))
    chosen = selection["variant"]
    index = {(condition(r), r["kind"], r["variant"], r["seed"]): r for r in rows}
    conditions = sorted({condition(r) for r in rows},
                        key=lambda c: [int(x) for x in "".join(
                            ch if ch.isdigit() else " " for ch in c).split()])
    seeds = sorted({r["seed"] for r in rows})
    kinds = ("fly", "rewired", "rnn", "gru", "leaky")

    def variant_of(kind: str, which: str) -> str:
        return "base" if which == "base" else (chosen if kind in ("fly", "rewired", "leaky")
                                                else without_gate(chosen))

    def tests(cond, kind, which):
        return [index[(cond, kind, variant_of(kind, which), s)] for s in seeds
                if (cond, kind, variant_of(kind, which), s) in index]

    out = ["다중 단서 후속 연구 III: 색·종류 결합과 방해 입력 기억 유지 개선 결과",
           f"작성: {datetime.now().astimezone().isoformat()}",
           "프로토콜: multi_cue_binding_protocol.txt. 오목 및 5,000개 확장은 보류.", "",
           f"선택된 개입(조정 시드 기준): {chosen}  (RNN·GRU는 게이트 제외 변형 "
           f"{selection['rnn_gru_variant']})", "",
           "1. 조정 요약", (DATA / "tuning_table.txt").read_text(encoding="utf-8").split(
               "모델·변형·조건별")[0].rstrip(), "",
           "2. 최종 결과 (새 시드, 시험 정확도 평균±표본표준편차, 5시드, 8,000 업데이트)"]
    for cond in conditions:
        out.append(f"\n[{cond}]")
        for kind in kinds:
            base, new = tests(cond, kind, "base"), tests(cond, kind, "new")
            if not base:
                continue
            b = [r["test"]["accuracy"] for r in base]
            n = [r["test"]["accuracy"] for r in new]
            diff = [y - x for x, y in zip(b, n)]
            reached = lambda rs: sum(r["to90"] is not None for r in rs)
            out.append(f"  {KIND_LABEL[kind]}: base {fmt(b)} → 개입 {fmt(n)} "
                       f"(짝차 {mean(diff) * 100:+.2f}pp; 90% 도달 시드 {reached(base)}→{reached(new)}/5)")
            out.append("      시드별 base: " + ", ".join(f"{v * 100:.1f}" for v in b)
                       + " / 개입: " + ", ".join(f"{v * 100:.3f}" for v in n))
            to90 = [r["to90"] for r in new]
            out.append("      개입 90% 도달 업데이트: "
                       + ", ".join(">8000" if v is None else str(v) for v in to90))
    out.append("\n3. 짝 비교 (같은 시드) — 실제 Fly 대비, base 및 개입 시험 정확도 차이(pp)")
    for cond in conditions:
        for other in ("rewired", "leaky", "gru", "rnn"):
            for which in ("base", "new"):
                a, b = tests(cond, "fly", which), tests(cond, other, which)
                if len(a) == len(b) and a:
                    d = [(x["test"]["accuracy"] - y["test"]["accuracy"]) * 100
                         for x, y in zip(a, b)]
                    out.append(f"  [{cond}] 실제 Fly - {KIND_LABEL[other]} ({'base' if which == 'base' else '개입'}): "
                               f"{mean(d):+.2f} ± {sd(d):.2f}pp, 시드별 "
                               + ", ".join(f"{x:+.1f}" for x in d))
    out.append("\n4. 질문된 단서의 제시 순서별 정확도(0=가장 먼저 본 단서, 평균, 5시드)")
    for cond in conditions:
        for kind in kinds:
            for which in ("base", "new"):
                rs = tests(cond, kind, which)
                if not rs:
                    continue
                slots = sorted({s for r in rs for s in r["test"]["by_slot"]})
                values = {s: mean([r["test"]["by_slot"][s] for r in rs if s in r["test"]["by_slot"]])
                          for s in slots}
                out.append(f"  [{cond}] {KIND_LABEL[kind]} {'base' if which == 'base' else '개입'}: "
                           + ", ".join(f"{s}번째 {v * 100:.1f}%" for s, v in values.items()))
    out.append("\n5. 방해 물체가 질문 색을 재사용했는지에 따른 정확도(평균, 5시드)")
    for cond in conditions:
        for kind in kinds:
            for which in ("base", "new"):
                rs = tests(cond, kind, which)
                if not rs or "True" not in rs[0]["test"]["distractor_reused_color"]:
                    continue
                values = {flag: mean([r["test"]["distractor_reused_color"][flag] for r in rs
                                      if flag in r["test"]["distractor_reused_color"]])
                          for flag in ("False", "True")}
                out.append(f"  [{cond}] {KIND_LABEL[kind]} {'base' if which == 'base' else '개입'}: "
                           f"재사용 안 함 {values['False'] * 100:.1f}% / 재사용 {values['True'] * 100:.1f}%")
    out.extend([
        "\n6. 수정된 해석 (2026-10-04 원자료·코드 대조)",
        "  성과: aux를 사용한 새 학습은 평가한 다중 단서 조건에서 답 정확도를 크게 개선했다. "
        "지연8의 세 조건은 5모델·5시드 모두 시험 100%였다. 지연16에서는 실제 Fly 평균 "
        "99.99%, 재배선 Fly 98.47%, 나머지 세 모델 100%였다. 모든 조건·시드가 100%인 것은 아니다.",
        "  지연16 예외: 실제 Fly 시드3104는 99.951% (4096개 중 2개 오답), 재배선 Fly "
        "시드3104는 약92.3%였다. 제시 순서 표의 100.0%는 반올림을 포함한다.",
        "  제시 순서: base의 고원을 '마지막 단서만 기억'으로 일반화할 수 없다. 단서2·지연8·방해6의 "
        "실제 Fly는 첫 단서66.5%, 마지막65.7%였고, 단서4의 실제 Fly도 위치별66.8~67.6%였다. "
        "누설 기억 일부 조건에서는 뒤 단서의 정확도가 높았지만, 전체 모델에 공통인 원인으로 확정되지 않았다. "
        "aux의 위치별 높은 답 정확도는 각 단서를 질문에 맞게 활용한 결과이며, 모든 정보의 완전 보존이나 "
        "입력에 의한 덮어쓰기 원인을 직접 측정한 것은 아니다.",
        "  색 재사용: '모든 차이가 1pp 미만'은 아니다. 단서2·지연8·방해6 base에서 실제 Fly는 "
        "64.7%/66.4%, RNN은64.5%/66.9%, 누설 기억은65.1%/66.7%였다. 재사용에 따른 일관된 "
        "성능 저하는 관찰되지 않았지만, 이 조건별 비교로 방해 입력의 간섭을 배제하거나 덮어쓰기를 "
        "주원인으로 확정할 수 없다.",
        "  GRU: 지연16 base 시드3103·3105의 시험은34.0%·33.1%였다. aux는 이 체크포인트를 "
        "재개한 것이 아니라 같은 시드로 처음부터 별도 학습했다. 따라서 붕괴 모델의 복구 실험이 아니라 "
        "별도 학습에서 실패를 피한 결과다. '90% 도달 업데이트'는 500회 간격 검증에서 처음90% 이상인 "
        "시점이다. 시드3103의 500회 검증은99.902%, 1000회는100%였으며, 최종 시험은8000회 뒤에 수행했다.",
        "  감독과 구조: aux는 질문받지 않은 단서의 종류 정답도 학습에 제공한다. 종류는 ball/key/box의 "
        "3개이며 이진 선택 과제가 아니다. 조정에서는 bind·gate 단독도100%였으므로 aux만이 유일한 "
        "해법이라고 할 수 없다. 간선 없는 누설 기억도 추가 감독으로 높은 성능을 얻었지만, 평가 범위 "
        "밖의 모든 순환 동역학이나 실제 Fly 연결의 우수성을 입증한 것은 아니다.",
        "  다음 방향: aux 학습 후 더 긴 지연·새 방해 조건·방/복도 관측으로의 일반화와 기억 보존·" 
        "색-종류 결합·질문 활용의 실패를 분리하는 검증이 필요하다.",
    ])
    out.append("\n한계: 시드 5개는 탐색적이다. 개입별 학습률을 다시 찾지 않았다. aux는 질문받지 않은 "
               "단서의 정답을 쓰는 추가 감독이며 gate·bind는 구조 추가다. 재배선은 시드별 그래프 하나이므로 "
               "시드와 그래프가 섞인다. RNN·GRU는 개입 포함 파라미터 수에 맞춰 은닉 크기도 다시 정하므로 "
               "같은 초기 시드가 동일한 크기·초기 가중치를 뜻하지 않는다. 이번 시험으로 개입을 다시 고르지 않았다.")
    path = ROOT / "reports" / "memory" / "multi_cue_binding_results.txt"
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("select", "report"))
    {"select": select, "report": report}[parser.parse_args().mode]()

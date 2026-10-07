"""Report all prespecified recovery stages, using seed as the comparison unit."""

from __future__ import annotations

import hashlib
import json
import shutil
import statistics
from datetime import datetime, timedelta, timezone

import torch

from multi_cue_followup import atomic_json
from multi_cue_memory import ROOT
from multi_cue_recovery import KINDS, OUT, VARIANTS

NAMES = {
    "fly": "실제 Fly",
    "rewired": "재배선 Fly",
    "rnn": "RNN",
    "gru": "GRU",
    "leaky": "누설 기억",
}
TESTS = ("trained_delays", "unseen_delays", "unseen_delays_wait")


def records(folder):
    return [
        json.loads(p.read_text(encoding="utf-8")) for p in sorted(folder.glob("*.json"))
    ]


def stats(values):
    return {
        "mean": statistics.mean(values),
        "sample_std": statistics.stdev(values) if len(values) > 1 else 0,
        "values": values,
    }


def fmt(values):
    s = stats(values)
    return f"{s['mean'] * 100:.2f} ± {s['sample_std'] * 100:.2f}%"


def per_seed(rows, kind, getter):
    group = [r for r in rows if r["kind"] == kind]
    seeds = sorted({r["seed"] for r in group})
    return [
        statistics.mean(getter(r) for r in group if r["seed"] == seed) for seed in seeds
    ]


def main():
    readout = records(OUT / "readout/results")
    validation = json.loads(
        (OUT / "readout/validation.json").read_text(encoding="utf-8")
    )
    choice = json.loads((OUT / "selected_head.json").read_text(encoding="utf-8"))
    visual = records(OUT / "visual/diagnosis")
    tuning = json.loads(
        (OUT / "visual/summary_tuning.json").read_text(encoding="utf-8")
    )
    summary = records(OUT / "visual/summary_main")
    raw = records(OUT / "repeat/raw")
    repaired = records(OUT / "repeat/repaired")
    counts = {
        "readout_models": len(readout),
        "readout_heads": len(validation),
        "visual_diagnosis": len(visual),
        "summary_tuning": len(tuning),
        "summary_main": len(summary),
        "repeat_raw": len(raw),
        "repeat_repaired": len(repaired),
    }
    expected = {
        "readout_models": 15,
        "readout_heads": 60,
        "visual_diagnosis": 15,
        "summary_tuning": 30,
        "summary_main": 15,
        "repeat_raw": 90,
        "repeat_repaired": 90,
    }
    if counts != expected:
        raise ValueError(f"incomplete stages: {counts}")
    if any(r.get("perception_metric_version") != 2 for r in visual):
        raise ValueError("refresh forced query-color metrics before summarizing")
    selected = choice["variant"]
    analysis = {
        "counts": counts,
        "selected_head": choice,
        "readout": {},
        "visual": {},
        "repeat": {},
    }
    lines = [
        "다중 단서 후속 연구 II: 출력층 → 관측 분리 → 어려운 조건 반복 결과",
        "작성: " + datetime.now(timezone(timedelta(hours=9))).isoformat(),
        "프로토콜: multi_cue_recovery_protocol.txt. 오목 및 5,000개 확장은 보류.",
        "",
        "1. 기억 고정·출력층만 재학습",
        "기준 단서2·지연4·방해2의 기존 기억15개. 새로운 판독 데이터4096/1024/4096.",
        "이 단계의 재배선은 시드별 그래프1/2/3 하나씩이다. 3단계의5그래프 평균과 구분한다.",
        "각 출력층500업데이트,B256. 상태 정규화와 출력층 재초기화를 함께 적용했다.",
        "검증 평균 손실로 선택한 공통 출력층: "
        + selected
        + ". 시험으로 선택하지 않았다.",
        "후보의 평균 검증 정확도/손실:",
    ]
    for variant in VARIANTS:
        score = choice["scores"][variant]
        lines.append(
            f"  {variant}: {score['mean_validation_accuracy'] * 100:.2f}% / {score['mean_validation_loss']:.6f}"
        )
    lines.append("새 시험에서 원래 출력 → 선택한 출력층(평균±표본표준편차):")
    for kind in KINDS:
        old = per_seed(readout, kind, lambda r: r["original"]["accuracy"])
        new = per_seed(readout, kind, lambda r: r["heads"][selected]["accuracy"])
        analysis["readout"][kind] = {"original": stats(old), "repaired": stats(new)}
        lines.append(f"  {NAMES[kind]}: {fmt(old)} → {fmt(new)}")
    lines.append("15개 전체 원자료(원래/보수 후/서로 다른 질문의 예측 변경율):")
    for row in readout:
        chosen = row["heads"][selected]
        lines.append(
            f"  {row['run']}: {row['original']['accuracy'] * 100:.2f}/{chosen['accuracy'] * 100:.2f}% / "
            f"{row['original']['prediction_changes_when_kinds_differ'] * 100:.2f}→{chosen['prediction_changes_when_kinds_differ'] * 100:.2f}%"
        )
    lines.extend(
        [
            "판독기 재학습은 추가 감독 예산을 쓴 진단이다. 원래 정책이 스스로 복구한 결과가 아니다.",
            "후보 간 검증 손실 차이는 작다. 선택된 활성함수의 일반적인 우위를 뜻하지 않는다.",
            "정규화·재초기화·새 출력층 학습을 함께 바꿨으므로 복구 효과를 활성함수 하나에만 귀속하지 않는다.",
            "실패한 Fly에서 단순 출력층 교체가 충분했는지는 시드별 결과로 판단한다.",
            "500회 학습과128개 은닉의 출력층이 실패해도 모든 기억 정보가 사라졌다고 단정할 수 없다.",
            "",
            "2. 방·복도 관측과 기억/답 선택의 분리",
            "기존 CNN+기억15개 고정. 기존 학습/검증만 재사용하고 새로운 생성시드86/87/88에서 시험했다.",
            "2a. CNN 특징 선형 판독: 실제 화면에 보이는 색별 종류12비트+질문 문 색4비트.",
            "양성 F1, 질문 문이 보일 때 질문 색 정확도(평균±표본표준편차):",
        ]
    )
    for kind in KINDS:
        analysis["visual"][kind] = {}
        for test in TESTS:
            group = [r for r in visual if r["kind"] == kind]
            f1 = [r["perception"][test]["positive_f1"] for r in group]
            query = [
                r["perception"][test]["visible_query_color_accuracy"] for r in group
            ]
            old = [r["tests"][test]["original"]["accuracy"] for r in group]
            new = [r["tests"][test]["repaired"]["accuracy"] for r in group]
            control = [
                r["tests"][test]["accuracy"] for r in summary if r["kind"] == kind
            ]
            analysis["visual"][kind][test] = {
                "perception_f1": stats(f1),
                "query_color": stats(query),
                "original": stats(old),
                "repaired": stats(new),
                "summary_control": stats(control),
            }
            lines.append(
                f"  {NAMES[kind]} / {test}: F1 {fmt(f1)}, 질문 색 {fmt(query)}"
            )
    lines.extend(
        [
            "2b. 최종 상태의 출력층 재학습: 원래 → 보수 후. 2c. CNN 없는 관측 속성 요약 학습 대조:",
            "각 요약 대조는 새 시드1201~1203,검증 선택1280업데이트. 원래ReLU 출력층을 사용했다.",
            "요약64차원은 눈에 보이는 색별 종류 개수12,문 색4,벽 위치48이며 지도/정답/역할 정보를 주지 않는다.",
            "물체 위치를 풀링해 원화면의 모든 정보를 보존하지 않는다. 총 파라미터에는 CNN이 없으므로 완전히 같은 구조 비교가 아니다.",
            "요약은 색×종류 결합을 입력에 명시적으로 인코딩한다. 개선은 관측 인식뿐 아니라 결합 표현과 학습 난도의 변화를 포함한다.",
            "단위: 원래 CNN+기억 → 출력층 보수 / 관측 요약+기억(별도 새 학습):",
        ]
    )
    for kind in KINDS:
        for test in TESTS:
            s = analysis["visual"][kind][test]
            lines.append(
                f"  {NAMES[kind]} / {test}: {fmt(s['original']['values'])} → {fmt(s['repaired']['values'])} / {fmt(s['summary_control']['values'])}"
            )
    lines.extend(
        [
            "CNN 판독 성공은 관측 속성이 특징에 존재한다는 뜻이다. 원래 기억이 해당 특징을 이용한다는 증거는 아니다.",
            "선형 판독 실패는 더 복잡한 판독기로도 정보가 없다는 뜻이 아니다. 관측 요약 실패도 공간 정보 풀링의 영향을 포함한다.",
            "따라서 요약 대조의 개선을 CNN 인식 오류만의 인과 증거로 단정하지 않는다.",
            "훈련/검증 데이터는 이전 연구에 사용했던 것이므로 연구 반복에 따른 검증 적응 가능성이 있다. 새 시험은 선택에 쓰지 않았다.",
            "고정 경로이며 RGB 사진 인식이나 자율 주행 결과가 아니다.",
            "",
            "3. 두 어려운 조건: 새 시드5개·재배선 그래프5개 독립 반복",
            "조건마다 본체45회 ×12,000업데이트,B128. 실제5·재배선25·RNN5·GRU5·누설5.",
            "원래 본체의 마지막 시험과 출력층 추가500회 진단을 구분한다.",
            "재배선은 각 초기화 안에서5그래프를 먼저 평균하고 그 뒤5초기화를 평균한다.",
        ]
    )
    for task in ("cues2_delay16_distract2", "cues2_delay8_distract6"):
        raw_group = [r for r in raw if r["task"] == task]
        repaired_group = [r for r in repaired if r["task"] == task]
        if len(raw_group) != 45 or len(repaired_group) != 45:
            raise ValueError("wrong difficult-condition count")
        analysis["repeat"][task] = {}
        lines.append(task + ": 원래12,000회 시험 / 새 판독 시험의 원래 → 보수 후")
        for kind in KINDS:
            final = per_seed(raw_group, kind, lambda r: r["test"]["accuracy"])
            old = per_seed(repaired_group, kind, lambda r: r["original"]["accuracy"])
            new = per_seed(repaired_group, kind, lambda r: r["repaired"]["accuracy"])
            analysis["repeat"][task][kind] = {
                "raw_final": stats(final),
                "original": stats(old),
                "repaired": stats(new),
            }
            lines.append(f"  {NAMES[kind]}: {fmt(final)} / {fmt(old)} → {fmt(new)}")
        for key in ("raw_final", "original", "repaired"):
            actual = analysis["repeat"][task]["fly"][key]["values"]
            rewired = analysis["repeat"][task]["rewired"][key]["values"]
            differences = [a - r for a, r in zip(actual, rewired)]
            analysis["repeat"][task][f"paired_fly_minus_rewired_{key}"] = stats(
                differences
            )
            lines.append(
                f"  실제-재배선({key}): {statistics.mean(differences) * 100:+.2f} ± {statistics.stdev(differences) * 100:.2f}pp, 시드별 "
                + ", ".join(f"{v * 100:+.2f}" for v in differences)
            )
    lines.extend(
        [
            "",
            "결론 및 한계",
            "기준 GRU의 붕괴 시드는 출력층만 다시 학습해100%로 복구됐다. 실패한 Fly 두 시드는 약66~67%로 남았다.",
            "관측 요약 대조의 실제 Fly는 학습 길이97.35%,미사용 길이96.65%,대기 추가82.73%였다.",
            "긴 지연에서 실제 Fly의 재배선 대비 평균 이득은+10.81pp지만 두 시드가 대부분의 이득을 만들었다.",
            "방해6에서 실제/재배선/RNN/누설 기억은약67%,GRU는새5시드 모두100%였다.",
            "출력층 보수만으로 어려운 조건의 Fly 고원은 해결되지 않았다. 모든 GRU 실패가 단순 출력층 붕괴인 것도 아니다.",
            "실제 연결 배치의 이점은 재배선 대비 짝 차이와 시드별 방향을 함께 보고 판단한다.",
            "시드는 초기 파라미터와 훈련 데이터 난수를 함께 바꾼다. 5회도 탐색적 연구이며 일반적인 모집단 우위를 확정하지 않는다.",
            "출력층 보수는 추가500×256개의 감독 예산을 썼다. 본체 학습 예산과 합치거나 숨기지 않는다.",
            "이 조건별 학습은 다른 단서 수/조건으로의 가중치 직접 전이가 아니다.",
            "단순 출력층으로 복구되지 않으면 색-종류 결합의 유지·질문 조건 선택·방해 억제를 공통 보조 손실/게이트로 연구할 필요가 있다.",
            "방·복도의 다음 후보는 모든 CNN에 동일한 보이는 색×종류·질문 색 보조 감독을 주는 실험이다. 새 조정·시험으로 효과를 확인해야 한다.",
            "개선 연구의 구조·검증·시험·예산은 별도 사전 기록으로 고정한다. 이번 시험으로 이번 후보를 다시 선택하지 않는다.",
            "새 코드 검증4개(가변 길이 상태 출력 일치,노란색/미관측 구분,관측만 사용,양성 지표) 통과.",
            "상세 단서 순서별 지표와 판독 시험/본체 곡선/가중치는 data/multi_cue_recovery의 JSON/PT에 보존했다.",
        ]
    )
    report = ROOT / "reports" / "memory" / "multi_cue_recovery_results.txt"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    source = OUT / "source"
    source.mkdir(exist_ok=True)
    hashes = {}
    for name in (
        "multi_cue_recovery_protocol.txt",
        "multi_cue_recovery.py",
        "multi_cue_observation_recovery.py",
        "run_multi_cue_recovery.py",
        "summarize_multi_cue_recovery.py",
        "test_multi_cue_recovery.py",
        "probe_recovery_batch.py",
        "multi_cue_followup.py",
        "multi_cue_memory.py",
        "grid_multi_cue.py",
        "multi_cue_visual_followup.py",
    ):
        path = (ROOT.parent if name.endswith("_protocol.txt") else ROOT / "code") / name
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        shutil.copy2(path, source / name)
    analysis.update(
        {
            "code_sha256": hashes,
            "torch": torch.__version__,
            "device": torch.cuda.get_device_name(),
            "report": str(report),
        }
    )
    atomic_json(OUT / "analysis.json", analysis)
    print(json.dumps({"report": str(report), "counts": counts}), flush=True)


if __name__ == "__main__":
    main()

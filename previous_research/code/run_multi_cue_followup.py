"""Run authorized research stages in order, retaining separate logs and protocols."""

import argparse
import json
import subprocess
import sys

from multi_cue_followup import OUT, ROOT


def execute(label, script, *args):
    print(json.dumps({"event": "stage_start", "stage": label}), flush=True)
    with (OUT / f"{label}.log").open("a", encoding="utf-8") as log:
        subprocess.run(
            [sys.executable, str(ROOT / "code" / script), *args],
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    print(json.dumps({"event": "stage_complete", "stage": label}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--from-stage",
        choices=("baseline", "frozen", "diagnose", "load", "visual"),
        default="baseline",
    )
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    stages = [
        ("baseline", "multi_cue_followup.py", ("baseline",)),
        ("frozen", "multi_cue_followup.py", ("frozen",)),
        ("diagnose", "diagnose_multi_cue.py", ()),
        ("load", "multi_cue_followup.py", ("load",)),
        ("visual", "multi_cue_visual_followup.py", ("tune",)),
    ]
    first = next(i for i, s in enumerate(stages) if s[0] == args.from_stage)
    for label, script, params in stages[first:]:
        if label == "visual":
            protocol = ROOT.parent / "multi_cue_visual_protocol.txt"
            if not protocol.exists():
                protocol.write_text(
                    "방·복도 다중 단서 비교 프로토콜 (3b, 최종 결과 생성 전)\n"
                    "2026-10-04. 경로는 전진으로 고정, 자율 주행 학습 아님.\n"
                    "단서2, 방해2. 학습/검증 복도6·10, 미사용 복도8·14.\n"
                    "CNN 구조는 모든 모델 동일, 기억 모듈의 파라미터 수는 Fly에 맞춤.\n"
                    "조정 초기화901·902: 학습2048개, 검증1024개, 8epoch(128업데이트).\n"
                    "학습률0.0003/0.003/0.01. 최고 검증 정확도 평균 우선, 동점은 손실,\n"
                    "다시 동점은 작은 학습률. Fly/재배선은 함께 선택한 공통값 사용.\n"
                    "최종 초기화1101~1103: 학습8192개, 검증1024개, 20epoch(1280업데이트).\n"
                    "배치128, AdamW, 기울기 제한1.0, FP32/TF32끔. 동일 데이터와 학습 순서.\n"
                    "검증 정확도 우선·동점 손실로 체크포인트 선택. 시험은 훈련 길이,\n"
                    "미사용 길이, 미사용 길이+확률0.5 대기의 세 그룹 각2048개.\n"
                    "시험 생성 시드는 학습·검증과 분리. 전체/복도 길이/단서 순서별 정확도.\n"
                    "기억 모듈을 새로 학습하므로 시퀀스 가중치의 직접 전이 시험 아님.\n"
                    "8192개를 반복 학습하며 시퀀스의 새 예제 생성 예산과 직접 비교하지 않음.\n",
                    encoding="utf-8",
                )
        execute(label, script, *params)
    rates = json.loads((OUT / "visual" / "selected_rates.json").read_text())
    protocol = ROOT.parent / "multi_cue_visual_protocol.txt"
    with protocol.open("a", encoding="utf-8") as handle:
        handle.write(
            "\n조정 검증으로 고정한 학습률 (최종 시험 전): " + json.dumps(rates) + "\n"
        )
    execute("visual_final", "multi_cue_visual_followup.py", "final")
    execute("summary", "summarize_multi_cue_followup.py")


if __name__ == "__main__":
    main()

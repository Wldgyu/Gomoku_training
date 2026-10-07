"""Summarize the three-step MiniGrid follow-up using seed-level graph averages."""

from __future__ import annotations

import json
import statistics
import sys

import numpy as np
import torch

from minigrid_memory_pilot import ROOT, sha256

OUT = ROOT / "data/minigrid_relation_followup"


def read(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def main():
    stage1, stage2, training = read("stage1.json"), read("stage2.json"), read("stage3_training.json")
    final, controls, leaky = read("stage3_autonomous.json"), read("controls.json"), read("leaky_baseline.json")
    if len(final["runs"]) != 40 or len(leaky["runs"]) != 5 or len(controls["runs"]) != 13:
        raise RuntimeError("A required follow-up run is incomplete")
    seed_values, means, paired = {}, {}, {}
    for kind in ("fly", "rewired", "rnn", "gru"):
        seed_values[kind], means[kind] = {}, {}
        for size in ("9", "15", "19"):
            values = [statistics.mean(row["sizes"][size]["all_four_success"] for row in final["runs"]
                                      if row["kind"] == kind and row["seed"] == seed)
                      for seed in range(711, 716)]
            seed_values[kind][size] = values
            means[kind][size] = statistics.mean(values)
    for size in ("9", "15", "19"):
        differences = [a - b for a, b in zip(seed_values["fly"][size], seed_values["rewired"][size])]
        paired[size] = {"per_seed_difference": differences, "mean_difference": statistics.mean(differences),
                        "sample_standard_deviation": statistics.stdev(differences)}
    leaky_means = {size: statistics.mean(row["sizes"][size]["all_four_success"] for row in leaky["runs"])
                   for size in ("9", "15", "19")}
    if not all(value == 1.0 for kind in ("fly", "rewired") for value in means[kind].values()):
        raise RuntimeError("Update the report's topology interpretation to match non-saturated results")
    if not all(value == 1.0 for value in leaky_means.values()):
        raise RuntimeError("Update the report's leaky-baseline interpretation")
    if not all(item["branch_reach_rate"] == 1.0 and item["branch_override_rate"] == 1.0
               and item["early_override_rate"] == 0 for row in final["runs"] for item in row["sizes"].values()):
        raise RuntimeError("Update the report's navigation and false-positive metrics")
    source_files = ("minigrid_relation_followup.py", "minigrid_learned_relation.py",
                    "evaluate_minigrid_learned_relation.py", "minigrid_compositional_policy.py",
                    "minigrid_relation_data.py", "minigrid_memory_diagnostic.py", "minigrid_memory_pilot.py",
                    "summarize_minigrid_relation_followup.py", "minigrid_relation_followup_protocol.txt")
    source_dir = OUT / "source"
    source_dir.mkdir(exist_ok=True)
    hashes = {}
    for name in source_files:
        path = (ROOT.parent if name.endswith("_protocol.txt") else ROOT) / name
        (source_dir / name).write_bytes(path.read_bytes())
        hashes[name] = sha256(path)
    analysis = {"means": means, "per_seed_graph_averages": seed_values, "paired_topology": paired,
                "graph_audit": training["graph_audit"],
                "leaky_means": leaky_means, "final_base_per_size": final["base_per_size"],
                "source_hashes": hashes, "actor_sha256": sha256(ROOT / "data/minigrid_relation_curriculum/relation_variable_control_rnn_1000_seed401.pt"),
                "detector_sha256": sha256(OUT / "detector.pt"), "python": sys.version,
                "torch": torch.__version__, "numpy": np.__version__,
                "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}
    (OUT / "analysis.json").write_text(json.dumps(analysis, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    labels = {"fly": "실제 Fly", "rewired": "재배선 Fly", "rnn": "RNN", "gru": "GRU"}
    table = "\n".join(f"  {labels[kind]:<10} " + "  ".join(f"{means[kind][size]*100:6.2f}%" for size in ("9", "15", "19"))
                      for kind in labels)
    seed_table = "\n".join(f"  {labels[kind]:<10} " + "/".join(f"{x*100:.2f}" for x in seed_values[kind]["19"])
                           for kind in labels)
    leaky_table = "  간선 없는 누설 기억 " + "  ".join(f"{leaky_means[size]*100:.2f}%" for size in ("9", "15", "19"))
    text = f"""MiniGrid 실제 주행·설정 조정·다중 재배선 비교 결과
작성일: 2026-10-03
실행 조건: minigrid_relation_followup_protocol.txt

핵심 결과

요청한 세 단계를 순서대로 완료했다. 실제 주행 경로를 수집해 시각
판독기를 개선하고, 훈련 길이 검증에서 모델별 학습률을 선택한 뒤,
새 초기화 5시드와 재배선 그래프 5개를 비교했다. 최종 자율 주행에서
실제 Fly와 재배선 Fly는 S9/S15/S19의 네 조합을 모두 성공했다.
실제 연결 배치가 재배선보다 유리한 차이는 이번 조건에서 없었다.
보조 대조에서는 처음부터 간선 없이 학습한 누설 기억도 세 길이를
모두 해결했다. 오목과 5,000개 확장은 실행하지 않았다.

1. 실제 주행 경로 및 분기 판독 개선

기존 RNN 401 이동 정책과 판독기 501, 명시적인 단서 보관·일치
비교를 쓰는 결합 정책의 실제 환경 주행을 기록했다. 환경 내부
상태는 오프라인 학습·측정 라벨만 제공하며 주행 행동을 정하지 않는다.
분기 라벨은 실제 갈림길 중앙에서 동쪽을 보는 경우에만 1이다.
S7/S11/S13 및 S17Random에서 설정마다 기본 에피소드 128개×4조합,
검증은 새 기본 에피소드 64개×4조합이다. 학습 경로 2,048개,
검증 경로 1,024개 모두 분기점 전 단서를 관찰해 기억 학습에 사용했다.
S7 수집 정책의 전체 성공률은 훈련 80.3%, 검증 81.6%였지만,
마지막 선택 직전의 경로는 모두 확보했다. 나머지 설정의 수집 성공률은
100%였다. 수집된 전체 방문 화면으로 분기 판독기를 학습했다.

S17Random의 갈림길은 x=5/6/8/9/10/11/12/14/15만 사용했다.
S9의 x=7, S15의 x=13, S19의 x=17은 훈련·검증 모두 제외했다.
훈련·검증 환경 시드의 교집합은 0이다. 최종 평가 환경 시드도 별도다.
S9/S15는 가변 훈련 길이 사이의 보간이고, S19는 최대 훈련
갈림길 x=15보다 두 칸 긴 x=17의 외삽이다.

시각 판독기는 기존 가중치에서 학습률 0.0003, batch512, 8 epoch
추가 학습했다. 선택은 훈련 길이의 새 검증 화면으로만 했다.
검증 화면 {stage1['after']['frames']:,}개에서 분기 오검출은
{stage1['before']['branch_false_positive_frames']}개 → {stage1['after']['branch_false_positive_frames']}개,
단서 종류 정확도는 {stage1['before']['cue_accuracy']*100:.2f}% → {stage1['after']['cue_accuracy']*100:.2f}%,
분기 precision/recall과 분기점 위쪽 물체 판독은 최종 모두 100%였다.
선택된 detector epoch는 {stage1['selected_epoch']}다. 모든 이후 구조에 같은 새 판독기를 고정했다.

원자료: data/minigrid_relation_followup/stage1.json,
        actor_paths.pt, memory_data.pt, detector.pt.

2. 모델별 검증 학습률 선택

후속 최종 학습과 겹치지 않는 초기화 시드 701/702에서
0.0003/0.001/0.003을 각 12 epoch씩 탐색했다. 모든 구조에
동일한 경로·순서·batch128·AdamW·선택 CE+단서 CE를 사용했다.
비교기는 참 단서·정답을 제공해 200회 사전 지도학습하고 고정했다.
체크포인트 선택은 검증 네 조합 성공률, 선택 정확도, 단서 정확도,
낮은 검증 손실 순이다. 만점 동점에서는 더 낮은 손실을 선택했다.

선택된 학습률: Fly와 재배선 Fly {stage2['selected_rates']['fly']},
RNN {stage2['selected_rates']['rnn']}, GRU {stage2['selected_rates']['gru']}.
Fly·재배선 Fly는 두 구조의 평균 검증 점수로 공통 학습률을 선택해
연결 배치 비교의 학습 조건을 동일하게 유지했다. RNN의 두 조정
시드 네 조합 검증 성공률은 0.0003에서 100/100%, 0.001에서
95.31/99.61%, 0.003에서 85.94/32.42%였다. RNN의 공통 0.001
설정은 이번 경로 학습에서도 최선이 아니었다. Fly·재배선 Fly·GRU는
탐색한 모든 학습률에서 두 조정 시드 모두 검증 100%였고, 손실로
학습률을 선택했다. 탐색 예산은 네 구조마다 동일하다.

원자료: data/minigrid_relation_followup/stage2.json 및 tuning/.

3. 새 5시드와 5개 재배선 그래프의 최종 자율 비교

설정을 고정한 뒤 초기화 시드 711~715에서 실제 Fly/RNN/GRU
각 5개를 학습했고, 차수 보존 재배선 seed0~4를 각 초기화에서
반복해 재배선 25개를 학습했다. 총 40개 실행이다. 모든 실행은
훈련 길이의 새 검증 경로에서 네 조합 성공률 100%였다.

기존에 만든 재배선 그래프의 뉴런 ID, 양방향 차수, 간선 수,
중복·자기 연결 부재를 검사했고, 각 초기화 시드의 실제/재배선
학습 가능 초기 파라미터가 같음을 확인했다. 그래프별 해시는
stage3_training.json의 graph_audit에 저장했다.

모든 구조에 같은 RNN 401 이동 정책과 새 시각 판독기를 붙여
자율 행동을 평가했다. 길이마다 새 기본 환경 에피소드 128개×4조합,
실행당 512개다. 최종 환경 시드는 44,000,000+크기×100,000부터다.
아래는 기본 에피소드의 네 조합을 모두 해결한 비율의 5시드 평균이다.
재배선은 각 초기화 시드에서 그래프 5개의 평균을 먼저 계산했다.

                   S9        S15       S19
{table}

S19 시드별 네 조합 성공률(711/712/713/714/715, %):
{seed_table}

실제 Fly - 재배선 5개 평균의 짝 차이는 S9/S15/S19에서 각 시드
모두 0 퍼센트포인트였다. 이 과제와 표본에서 실제 연결 배치의
최종 성능 이점을 찾지 못했다. 두 Fly 조건의 점수가 모두 포화돼
작은 차이가 다른 난이도에서도 없다는 뜻은 아니다. 재배선 25개를
독립적인 실제 Fly 짝 25개로 취급하지 않았다.

모든 40개 실행·세 길이의 갈림길 도달·분기 개입은 100%였고
조기 분기 개입은 0%였다. 남은 RNN·GRU의 S19 실패는 실제
갈림길에서의 단서 판독 오류와 일치했다. 원래 탐색 연구의 S19
평균(실제/재배선/RNN/GRU=81.1/78.8/50.3/66.9%)보다 이번
학습 체계의 값이 높았다. 경로·가변 길이·판독기·학습률·체크포인트
선택·환경 시드가 함께 바뀌어, 상승량 전체를 한 변경의 효과로
해석할 수는 없다.

원자료: stage3_training.json, stage3_autonomous.json, analysis.json,
        main/의 시드·그래프별 체크포인트.

4. 기억 및 순환 간선 대조

새 시드 711의 네 구조를 같은 S19 평가 에피소드에서 단서 입력
차단 또는 분기 직전 기억 초기화로 평가했다. 모든 조건에서
개별 에피소드 성공률 50%, 네 조합 성공률 0%였다. 단서 기억을
실제로 사용한다는 근거다. 원자료: controls.json.

최종 비교의 포화 결과를 해석하기 위해 추가한 탐색적 대조에서,
학습한 실제 Fly 5시드 모두 평가 중 순환 간선 가중치를 0으로
만들어도 S19 네 조합 성공률 100%였다. 이때 0.9×이전 상태를
유지하는 누설 메모리는 남는다. 이어 같은 입력·초기값·학습률·
예산·검증 기준으로 처음부터 간선을 0으로 고정한 5개 대조를
새로 학습했다. 결과는 아래와 같다.

                   S9        S15       S19
{leaky_table}

간선 없는 대조는 실제 Fly의 입력 투영과 누설 상태, cue readout,
비교기를 그대로 사용하고 그래프 간선만 학습하지 않는다. 기억
학습 파라미터는 5,002개(비교기의 고정된 20개는 제외)다. 이
대조가 만점을 냈으므로 현재 이진 단서 과제의 최종 성공률은
학습 가능한 뉴런 간 간선 없이도 얻을 수 있다. 실제 연결의
학습 속도 효과나 다른 과제의 효과까지 부정하는 결과는 아니다.
원자료: leaky_baseline.json, leaky/.

해석 범위와 다음 연구

학습형 기억·비교 정책의 새 길이 자율 주행을 개선했고, 여러
재배선 그래프에서도 결과가 재현됐다. S19에서 Fly 계열의
성능은 안정적이지만, 원본 연결 배치의 이점은 확인되지 않았다.
누설 상태만으로도 해결돼 현재 과제에서 연결망 상호작용의
필요성이 충분히 드러나지 않는다. 다음 연구는 간선 없는 누설
메모리를 기준 모델로 포함하고, 중간 방해 단서나 여러 단서를
구분해 보관해야 하는 과제를 설계하는 것이 타당하다.

이번 체계는 시각 판독과 참 단서·정답을 사용한 지도학습,
학습된 분기 신호에서의 행동 개입, 고정된 이동 정책을 포함한다.
보상만으로 학습한 원래 단일 정책의 결과는 아니다. Fly와
RNN/GRU는 파라미터 총량이 비슷하지만 상태 크기와 누설 동역학은
다르다. 학습률 선택 시드는 두 개, 본 반복 시드는 다섯 개다.
S9/S15/S19 길이는 이전 연구에서도 관찰했고 이번에는 독립
환경 시드로 평가했다. 100%는 사용한 Memory 환경 분포의
평가 표본 결과이며 모든 부분관측 과제에 대한 보장은 아니다.

실행 코드: minigrid_relation_followup.py,
           summarize_minigrid_relation_followup.py.
최종 코드 사본·해시·런타임 정보: data/minigrid_relation_followup/source/,
                                 analysis.json.
검증: 단위 검사 10/10, Ruff, py_compile 통과. 훈련·검증 시드의
교집합 0과 미사용 갈림길 위치의 누출 없음도 저장된 경로에서 확인했다.
"""
    (ROOT / "minigrid_relation_followup_results.txt").write_text(text, encoding="utf-8")
    print(json.dumps({"means": means, "leaky_means": leaky_means, "paired": paired}), flush=True)


if __name__ == "__main__":
    main()

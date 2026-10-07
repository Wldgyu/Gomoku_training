"""Summarize frozen aux transfer, order holdout, diagnostics and visual rollouts."""

from __future__ import annotations

import json
import statistics as stats
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRANSFER = ROOT / "data" / "multi_cue_aux_transfer"
VISUAL = ROOT / "data" / "multi_cue_visual_aux" / "runs"
OUT = ROOT / "reports" / "memory" / "multi_cue_aux_generalization_results.txt"
SEEDS = (3101, 3102, 3103, 3104, 3105)
VISUAL_SEEDS = (1101, 1102, 1103)
ORDER_SEEDS = (3201, 3202, 3203)
MODELS = ("fly", "rewired", "leaky", "rnn", "gru")
NAMES = {"fly": "실제 Fly", "rewired": "재배선 Fly", "leaky": "누설 기억",
         "rnn": "RNN", "gru": "GRU"}
CONDITIONS = ("trained", "reversed_cue_order", "delay24", "delay32",
              "distractors6", "distractors10", "combined24_6", "three_cues")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def pct(x):
    return f"{100 * x:.2f}%"


def ms(values):
    return f"{100 * stats.mean(values):.2f} ± {100 * stats.stdev(values):.2f}%"


def main():
    gen = {(kind, seed): read(TRANSFER / "generalization" / f"{kind}_seed{seed}.json")
           for kind in MODELS for seed in SEEDS}
    vis = {(kind, seed, aux): read(VISUAL / f"{kind}_{'aux' if aux else 'base'}_seed{seed}.json")
           for kind in MODELS for seed in VISUAL_SEEDS for aux in (False, True)}
    holdout = {(kind, seed): read(TRANSFER / "order_holdout" / f"{kind}_seed{seed}.json")
               for kind in MODELS for seed in ORDER_SEEDS}
    shared_nav = read(ROOT / "data" / "multi_cue_visual_aux" / "shared_navigation" /
                      "results.json")
    diagnostic = read(TRANSFER / "diagnostic" / "rewired_seed3104.json")["diagnostics"]
    replay_diffs = [abs(gen[k]["published_test"] - gen[k]["replayed_test"]) for k in gen]
    lines = ["다중 단서 aux 일반화·실패 분리·방/복도 적용 결과",
             f"작성: {datetime.now(timezone(timedelta(hours=9))).isoformat(timespec='seconds')}",
             "기존 multi_cue_binding_results.txt 이후 후속 실험. 오목·5,000개 확장은 보류.", "",
             "1. 고정 가중치 일반화",
             "원 연구는 가중치가 아닌 JSON 성적만 저장했다. 단서2·지연16·방해2 aux 모델을",
             "시드3101~3105별 원래 8,000 업데이트·배치·옵티마이저로 재현해 가중치를",
             "저장했다. 원 시험(각4096개)의 발표값과 재현값 최대 차이는 "
             f"{100 * max(replay_diffs):.5f}pp. 이후 모든 평가에서 가중치를 고정했다.",
             "각 조건은 새 4096개/시드, 모델끼리 같은 에피소드이며 학습·원 시험 시드와 분리된다.",
             "3종류 정답의 무작위 기준은 33.33%. 숫자는 5시드 평균 ± 표본표준편차.",
             "2단서 학습에서 3단서 시험은 새 부하이고,",
             "순서 뒤집기는 기존 학습 분포의 순열이므로 엄밀한 미학습 순서 시험은 아니다.", ""]
    labels = {"trained": "학습 조건(새 표본)", "reversed_cue_order": "단서 순서 뒤집기",
              "delay24": "지연24/방해2", "delay32": "지연32/방해2",
              "distractors6": "지연16/방해6", "distractors10": "지연16/방해10",
              "combined24_6": "지연24/방해6", "three_cues": "3단서/지연16/방해2"}
    for condition in CONDITIONS:
        lines.append(f"[{labels[condition]}]")
        for kind in MODELS:
            values = [gen[(kind, seed)]["conditions"][condition]["accuracy"] for seed in SEEDS]
            per_seed = ", ".join(f"{seed}:{pct(value)}" for seed, value in zip(SEEDS, values))
            lines.append(f"  {NAMES[kind]} {ms(values)} ({per_seed})")
        lines.append("")
    lines.extend(["엄격한 단서 순서 보류 대조 (새 학습, 시드3201~3203):",
                  "4색의 순서 있는 쌍12개 중 각 색 쌍의 한 방향6개만 학습했다. 모든 색은",
                  "훈련에서 첫째·둘째 위치에 모두 나타난다. aux 5모델을 같은 5,000 업데이트로",
                  "새로 학습한 후 고정해 훈련 방향·보류 방향 각4096개를 시험했다.",
                  "'짝 순서 반전'은 보류 시험의 동일한 에피소드에서 두 단서만 바꾼 대조다.",
                  "수치는 3시드 평균 ± 표본표준편차다."])
    for kind in MODELS:
        seen = [holdout[(kind, seed)]["results"]["train_order_new_episodes"]
                for seed in ORDER_SEEDS]
        unseen = [holdout[(kind, seed)]["results"]["heldout_order"]
                  for seed in ORDER_SEEDS]
        paired = [holdout[(kind, seed)]["results"]["same_episodes_reversed_to_seen"]
                  for seed in ORDER_SEEDS]
        first = [holdout[(kind, seed)]["heldout_by_slot"]["0"]["accuracy"]
                 for seed in ORDER_SEEDS]
        second = [holdout[(kind, seed)]["heldout_by_slot"]["1"]["accuracy"]
                  for seed in ORDER_SEEDS]
        per_seed = ", ".join(f"{seed}:{pct(value)}" for seed, value in zip(ORDER_SEEDS, unseen))
        lines.append(f"  {NAMES[kind]}: 훈련 방향 {ms(seen)} / 보류 방향 {ms(unseen)} "
                     f"({per_seed}) / 짝 순서 반전 {ms(paired)}; "
                     f"보류 첫/둘 단서 질문 {ms(first)}/{ms(second)}")
    lines.extend(["이 대조는 다른 학습 분포·시드라 원 5시드 일반화와 합쳐 평균내지 않는다.", "",
                  "2. 재배선 Fly 시드3104의 지연16 실패 분리",
                  "독립 판독 훈련3072개·시험4096개에서 가중치가 고정된 기억 상태를 읽었다."])
    ac = diagnostic["after_cues"]
    bq = diagnostic["before_query"]
    swap = diagnostic["query_swap"]
    final = diagnostic["final_readout"]
    lines.extend([
        f"  단서 직후 선형 판독 {pct(ac['linear_probe'])}; 지연 후 새 선형 판독 "
        f"{pct(bq['linear_probe'])}; 단서 직후 판독기를 지연 후 그대로 적용 "
        f"{pct(ac['same_probe_after_delay'])}.",
        f"  학습된 aux 판독기로 지연 후 색별 종류 판독 {pct(bq['trained_aux_head'])}.",
        f"  같은 새 시험에서 본래 답 출력 {pct(final['trained_answer_head'])}; 이미 학습된",
        f"  aux 색별 출력을 질문 색으로 명시 선택 {pct(final['aux_with_explicit_query_selection'])}.",
        f"  두 질문 색을 각각 제시해 모두 정답인 에피소드 {pct(swap['both_correct'])};",
        f"  두 종류가 다를 때 답이 바뀐 비율 {pct(swap['changes_when_kind_differs'])}.",
        "원 시험 92.33%는 별도 4096개 표본이다. 이 시드의 지연16 오답은 색·종류",
        "정보의 소실보다 질문에 따른 본래 답 출력 선택에 주로 놓인다. 선형 판독기와",
        "명시 선택은 진단용 추가 readout이며 원 모델의 실제 답 성능으로 세지 않는다.", "",
        "3. 방·복도 화면 관측과 자율 전진/정지",
        "MiniGrid 7×7 물체·색·상태 코드(실사 RGB 아님)를 CNN으로 읽고 고정 전진",
        "경로의 마지막 종류 답을 학습했다. base/aux는 같은 초기화 시드·데이터·학습률·",
        "20epoch·8192 훈련/1024 검증 에피소드로 짝지었다. aux는 모든 단서 색의 종류를",
        "추가 감독한다. 시험은 학습 길이6/10, 미사용 길이8/14, 미사용 길이+대기",
        "각2048개다. 아래는 3시드 평균 ± 표본표준편차와 평균 짝차(pp).", ""])
    for test_name, label in (("trained_lengths", "학습 복도 길이 마지막 답"),
                             ("unseen_lengths", "미사용 복도 길이 마지막 답"),
                             ("unseen_wait", "미사용 길이+대기 마지막 답")):
        lines.append(f"[{label}]")
        for kind in MODELS:
            base = [vis[(kind, seed, False)]["tests"][test_name]["answer_accuracy"]
                    for seed in VISUAL_SEEDS]
            aux = [vis[(kind, seed, True)]["tests"][test_name]["answer_accuracy"]
                   for seed in VISUAL_SEEDS]
            cue = [vis[(kind, seed, True)]["tests"][test_name]["aux_cue_accuracy"]
                   for seed in VISUAL_SEEDS]
            routed = [vis[(kind, seed, True)]["tests"][test_name]
                      ["aux_visible_door_route_accuracy"] for seed in VISUAL_SEEDS]
            diff = 100 * stats.mean(a - b for a, b in zip(aux, base))
            lines.append(f"  {NAMES[kind]}: base {ms(base)} → aux {ms(aux)} "
                         f"(짝차 {diff:+.2f}pp); aux 색별 종류 {ms(cue)}; "
                         f"보이는 문 색 명시 선택 {ms(routed)}")
        lines.append("")
    lines.extend(["'보이는 문 색 명시 선택'은 마지막 화면의 물체·색 코드를 외부에서 읽어",
                  "학습된 aux 색별 출력을 선택한 진단이며 원 모델의 실제 답 성능이 아니다.",
                  "공통 자율 이동 정책: 별도 CNN 전진/정지기를 교사 경로4096개·500업데이트로",
                  "학습했다. 미사용 복도8/14의 새512개에서 화면만 보고 전진·정지했으며",
                  f"이동 성공률은 {pct(shared_nav['movement_success'])}. 모든 답 모델은 같은 실제",
                  "자율 궤적과 같은 질문을 받았다. 아래는 종단 답 성공률(3시드 평균±표본표준편차)."])
    for kind in MODELS:
        base = [shared_nav["models"][f"{kind}_base_seed{seed}"]["end_to_end_answer_success"]
                for seed in VISUAL_SEEDS]
        aux = [shared_nav["models"][f"{kind}_aux_seed{seed}"]["end_to_end_answer_success"]
               for seed in VISUAL_SEEDS]
        lines.append(f"  {NAMES[kind]}: base {ms(base)} → aux {ms(aux)} "
                     f"(짝차 {100 * stats.mean(a - b for a, b in zip(aux, base)):+.2f}pp)")
    lines.extend(["", "모델별 정지 헤드 대조: 각 답 모델의 고정된 CNN 특징에서 별도 전진/정지 헤드만",
                  "교사 경로의 2048개로 학습했다. 평가 시에는 경로·정답을 주지 않고,",
                  "화면을 보고 전진 또는 정지한다. 미사용 복도8/14 새256개/시드.",
                  "도달·정지 성공률과 도달 실패를 오답으로 센 종단 답 성공률을 분리했다."])
    for kind in MODELS:
        bm = [vis[(kind, seed, False)]["tests"]["autonomous_unseen"]["movement_success"]
              for seed in VISUAL_SEEDS]
        am = [vis[(kind, seed, True)]["tests"]["autonomous_unseen"]["movement_success"]
              for seed in VISUAL_SEEDS]
        be = [vis[(kind, seed, False)]["tests"]["autonomous_unseen"]["end_to_end_answer_success"]
              for seed in VISUAL_SEEDS]
        ae = [vis[(kind, seed, True)]["tests"]["autonomous_unseen"]["end_to_end_answer_success"]
              for seed in VISUAL_SEEDS]
        lines.append(f"  {NAMES[kind]}: 이동 base {ms(bm)} → aux {ms(am)}; "
                     f"종단 답 base {ms(be)} → aux {ms(ae)}")
    lines.extend(["", "해석과 한계:",
                  "  숫자 입력에서 aux의 훈련 조건 고성능은 새로운 지연·방해·단서 부하에",
                  "  일률적으로 유지되지 않았다. 결과는 모델과 시드에 따라 크게 다르다.",
                  "  방·복도에서는 aux만으로 마지막 답의 일관된 개선이 확인되지 않았다.",
                  "  자율 이동은 직선 복도의 전진/정지 정책이며 회전·탐색·장애물 회피를",
                  "  검증하지 않는다. 모델별 정지 헤드는 고정된 시각 특징만 읽으므로",
                  "  그 이동 실패는 더 강한 항법 정책의 한계를 뜻하지 않는다.",
                  "  시각 학습은 유한 8192개를",
                  "  반복했고, 숫자 과제는 매 업데이트 새 에피소드를 썼으므로 직접 비교 불가.",
                  "  재배선 그래프는 시드마다 하나여서 시드·그래프 효과를 분리하지 못한다.",
                  "  일반화 5시드·엄격 순서 보류 3시드·시각 3시드는 탐색적 근거다.",
                  "", "원자료: data/multi_cue_aux_transfer/ 및 data/multi_cue_visual_aux/",
                  "재실행: multi_cue_aux_transfer.py, diagnose_aux_transfer.py,",
                  "        multi_cue_order_holdout.py, refresh_order_holdout.py,",
                  "        multi_cue_visual_aux.py, visual_aux_shared_navigation.py,",
                  "        refresh_visual_aux_results.py,",
                  "        summarize_multi_cue_aux_transfer.py"])
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()

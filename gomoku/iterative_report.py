"""Build the final report for iterative DAgger/threat experiments."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from .train import atomic_json

NAMES = {
    "baseline": "A 기존 방식",
    "step1": "B 중간 판 커리큘럼",
    "c_dagger25": "C 반복 수집 25%",
    "d_threat_spatial": "D 공간 위협 예측",
}

def aggregate_full(data, arm, split):
    rows = [row for row in data["full_games"] if row["arm"] == arm and row["split"] == split]
    games = sum(row["games"] for row in rows)
    block = sum(row["tactical_decisions"]["block"]["positions"] for row in rows)
    omissions = sum(row["defense_omissions"] for row in rows)
    multiple = sum(row["games_reaching_multiple_threat"] for row in rows)
    return {
        "games": games, "wins": sum(row["wins"] for row in rows),
        "defense_opportunities": block, "defense_omissions": omissions,
        "defense_omission_rate": omissions / block,
        "multiple_threat_games": multiple, "multiple_threat_rate": multiple / games,
        "safe_forks": sum(row["actual_safe_forks_created"] for row in rows),
    }

def aggregate_mid(data, arm, split):
    rows = [row for row in data["midgames"]
            if row["arm"] == arm and row["split"] == split and row["family"] == "all"]
    return {str(depth): {
        "games": sum(row["games"] for row in rows if row["horizon"] == depth),
        "within_horizon_wins": sum(row["within_horizon_wins"] for row in rows
                                   if row["horizon"] == depth),
    } for depth in (1, 2, 3)}

def pct(value):
    return f"{100 * value:.1f}%"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--prior", type=Path, required=True)
    args = parser.parse_args()
    prior_eval = json.loads((args.prior / "evaluation.json").read_text(encoding="utf-8"))
    current_eval = json.loads((args.root / "evaluation.json").read_text(encoding="utf-8"))
    audit = json.loads((args.root / "audit.json").read_text(encoding="utf-8"))
    clean = json.loads((args.root / "clean_common_test.json").read_text(encoding="utf-8"))
    threat_validation = json.loads((args.root / "threat_spatial_validation.json").read_text(encoding="utf-8"))
    threat_test = json.loads((args.root / "threat_spatial_test.json").read_text(encoding="utf-8"))
    source = {"baseline": prior_eval, "step1": prior_eval,
              "c_dagger25": current_eval, "d_threat_spatial": current_eval}
    summary = {arm: {"test_full": aggregate_full(source[arm], arm, "test"),
                     "test_mid": aggregate_mid(source[arm], arm, "test")} for arm in NAMES}
    c50 = {"validation_full": aggregate_full(current_eval, "c_dagger", "validation"),
           "validation_mid": aggregate_mid(current_eval, "c_dagger", "validation")}
    flawed_d = {"validation_full": aggregate_full(current_eval, "d_threat", "validation"),
                "validation_mid": aggregate_mid(current_eval, "d_threat", "validation")}
    machine = {
        "experiment": "12x12 freestyle Gomoku; Fly and GRU; single training seed 4810",
        "arms": summary,
        "validation_only_diagnostics": {"dagger_50_percent": c50,
                                        "nonspatial_threat_head": flawed_d},
        "threat_validation": threat_validation, "threat_test": threat_test,
        "audit": audit, "clean_common_test": clean,
        "decision": {
            "expand_four_models_three_seeds": False,
            "reason": "No arm won a fresh full game, no arm created an actual safe fork, and D increased multiple-threat reach.",
            "best_defensive_candidate": "c_dagger25_gru",
            "next_bottleneck": "On-policy aggregation contained almost only defense categories; visited-state offensive policy improvement is missing.",
        },
    }
    atomic_json(args.root / "summary.json", machine)

    lines = [
        "# 반복 기보 수집과 상대 위협 예측 결과", "",
        "## 결론", "",
        "요청한 1단계 반복 기보 수집과 2단계 상대 위협 예측을 12×12 Fly·GRU에서 완료했다. "
        "반복 수집 25% 혼합(C)은 실제 경기 방어를 크게 개선했지만 공격 연결을 약화시켰다. "
        "공간 위협 예측(D)은 3수 공격을 일부 회복했지만 막을 수 없는 복수 위협 도달을 늘렸다. "
        "모든 최종 모델은 강한 전술 상대와의 새 빈 판 512경기에서 0승이었고 실제 안전한 복수 위협도 만들지 못했다. "
        "따라서 네 모델×3시드 확대 조건은 충족하지 못했다.", "",
        "## 최종 테스트", "",
        "| 방식 | 강한 상대 승리 | 방어 누락 | 복수 위협 도달 | 1수 완주 | 2수 완주 | 3수 완주 | 실제 안전 포크 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in NAMES:
        full, mid = summary[arm]["test_full"], summary[arm]["test_mid"]
        lines.append(
            f"| {NAMES[arm]} | {full['wins']}/{full['games']} | "
            f"{full['defense_omissions']}/{full['defense_opportunities']} ({pct(full['defense_omission_rate'])}) | "
            f"{full['multiple_threat_games']}/{full['games']} ({pct(full['multiple_threat_rate'])}) | "
            f"{mid['1']['within_horizon_wins']}/{mid['1']['games']} ({pct(mid['1']['within_horizon_wins']/mid['1']['games'])}) | "
            f"{mid['2']['within_horizon_wins']}/{mid['2']['games']} ({pct(mid['2']['within_horizon_wins']/mid['2']['games'])}) | "
            f"{mid['3']['within_horizon_wins']}/{mid['3']['games']} ({pct(mid['3']['within_horizon_wins']/mid['3']['games'])}) | "
            f"{full['safe_forks']} |")
    lines += [
        "",
        "C는 B보다 방어 누락률을 39.2%에서 21.7%로 낮추고 복수 위협 도달률도 66.6%에서 58.8%로 낮췄다. "
        "그러나 3수 공격 완주는 16.0%에서 3.9%로 떨어졌다. D는 3수 공격을 10.9%까지 회복했지만 "
        "복수 위협 도달률은 73.0%로 악화됐다. 방어 문제를 더 잘 푸는 것만으로는 공격 주도권이나 전체 승리로 이어지지 않았다.", "",
        "## 1단계: 실제 방문 판 반복 수집", "",
        "각 모델은 워밍업 직후와 PPO 100·200업데이트 뒤에 강한 상대와 256경기씩 두었다. "
        "패배 경기에서 제한된 교사가 안전한 대안을 증명할 수 있고 모델이 그 대안을 벗어난 최초 지점만 저장했다. "
        "평가용 중간 판의 모든 D4 대칭 접두 상태를 제외했다.", "",
        "| 모델 | 재분석 경기 | 채택 문제 | 즉시 방어 | 포크 예방 | 강제 연속 공격 예방 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for kind in ("fly", "gru"):
        run = audit["runs"][f"c_dagger25_{kind}"]
        cats = run["dagger"]["categories"]
        lines.append(f"| {kind.upper()} | {run['dagger_games']} | {run['dagger']['records']} | "
                     f"{cats.get('2', 0)} | {cats.get('4', 0)} | {cats.get('6', 0)} |")
    lines += [
        "",
        "첫 탐색에서는 새 문제를 보조 배치의 50%로 사용했다. 검증에서 공격 커리큘럼이 붕괴해 "
        "1·2·3수 완주가 각각 34.4%, 8.6%, 0.8%까지 떨어졌으므로 테스트를 열지 않고 25% 혼합으로 낮췄다. "
        "최종 C는 이 검증 선택을 반영한 설정이다. 수집 문제는 거의 모두 방어 범주였으며 이것이 공격 약화의 직접적인 자료 분포 신호다.", "",
        "## 2단계: 후보 착수 조건부 위협 예측", "",
        "자기 후보 착수와 현재 공간 특징을 입력해 그 수 뒤 상대의 즉시 승리 칸과 복수 위협 생성 칸을 예측했다. "
        "후보 수의 영향이 5목 선 전체에 전달되도록 두 개의 5×5 합성곱을 사용했다. 이 출력은 학습에만 사용했고 "
        "실전 착수는 기존 정책의 점유 칸 마스크와 argmax로 결정했다.", "",
        "D는 한 최적화 단계마다 기존 착수 예제 64개와 위협 예제 64개를 사용했다. "
        "C의 착수 예제 128개와 합계 예산은 같지만 직접 착수 감독의 절반을 예측 감독으로 교체한 비교다. "
        "위협 헤드 때문에 파라미터는 Fly 300,370→307,988, GRU 300,386→308,004로 늘었다.", "",
        "| 모델 | 즉시 위협 양성/음성 평균 확률 | 즉시 top-k 재현율 | 복수 위협 양성/음성 평균 확률 | 복수 top-k 재현율 |",
        "|---|---:|---:|---:|---:|",
    ]
    for kind in ("fly", "gru"):
        row = threat_test[kind]
        immediate, multiple = row["immediate_win"], row["multiple_threat"]
        lines.append(f"| {kind.upper()} | {immediate['positive_mean_probability']:.3f}/"
                     f"{immediate['negative_mean_probability']:.3f} | {pct(immediate['top_k_recall'])} | "
                     f"{multiple['positive_mean_probability']:.3f}/{multiple['negative_mean_probability']:.3f} | "
                     f"{pct(multiple['top_k_recall'])} |")
    lines += [
        "",
        "위협 지도는 양성 칸을 음성 칸보다 높은 평균 확률로 구분했지만 위치 순위 정확도는 낮았다. "
        "특히 즉시 위협 top-k 재현율이 Fly 4.7%, GRU 3.1%에 그쳐 보조 출력이 충분히 정밀하다고 볼 수 없다.", "",
        "처음 만든 1×1 위협 헤드는 후보 착수의 영향이 주변 칸으로 전파되지 않아 모두 0에 가까운 지도를 냈다. "
        "이 결과는 검증 전용 실패로 보존했다. 공간 헤드와 희소 양성 균형 표집으로 수정하고 전체 학습을 처음부터 다시 실행한 결과만 최종 테스트에 사용했다.", "",
        "## 예산과 데이터 감사", "",
        "- 모든 A/B/C/D 모델은 PPO 300업데이트, 32환경×16롤아웃으로 153,600번의 학습 결정을 사용했다.",
        "- C는 착수 보조 예제 153,600개, D는 착수 76,800개와 위협 76,800개로 총 153,600개를 사용했다.",
        "- C/D는 모델별로 재분석 경기 768개를 추가 생성했다. 교사 탐색 비용은 A/B에 없으므로 총 계산 비용은 같지 않다.",
        "- 반복 수집 라벨을 현재 교사로 전부 재계산했고 체크포인트 300/300, 고유 키, 합법 기보를 확인했다.",
        "- 중간 판 평가 접두 상태 13,057개와 반복 수집 자료의 D4 중복은 0개였다.",
        "- 위협 학습·검증·테스트의 현재 판 D4 중복은 모두 0개였다.", "",
        "실제 빈 판 테스트에서는 모델 경로가 학습 방문 판으로 되돌아오는 경우가 있었다. "
        "이는 반복 방문 학습의 의도와 맞지만 엄격한 일반화 수치에는 영향을 줄 수 있다. "
        f"어느 모델에서든 중복이 발견된 공통 오프닝 {clean['excluded_openings']}개를 모두 제거한 보조 분석은 모델당 "
        f"{clean['retained_games_per_model']}경기를 남겼다.", "",
        "| 방식 | 청정 승리 | 청정 방어 누락률 | 청정 복수 위협 도달률 |",
        "|---|---:|---:|---:|",
    ]
    for arm in NAMES:
        row = clean["summary"][arm]
        lines.append(f"| {NAMES[arm]} | {row['wins']}/{row['games']} | "
                     f"{pct(row['defense_omission_rate'])} | {pct(row['multiple_threat_game_rate'])} |")
    lines += [
        "",
        "이 부분집합은 테스트를 본 뒤 만든 보조 분석이므로 원 테스트를 대체하지 않는다. "
        "그럼에도 C의 방어 개선, D의 복수 위협 악화, 전체 0승이라는 결론은 유지됐다.", "",
        "## 판단과 다음 방향", "",
        "현재 결과로 Fly 연결 구조나 GRU의 우월성을 주장할 수 없다. D-GRU가 중간 판 3수 완주와 방어 누락 수에서 가장 좋았지만 "
        "전체 경기의 복수 위협 도달은 가장 높았고 승리는 없었다. 4모델×3시드 확대는 보류한다.", "",
        "다음 병목은 모델이 실제로 방문하는 공격 상태에 대한 정책 개선이다. 이번 반복 자료는 거의 전부 방어였으므로 "
        "다음 실험은 같은 방문 기보에서 놓친 공격 기회, 방어하면서 공격하는 수, 성공한 공격 연결을 교사가 함께 표시하고 "
        "방어·공격 할당량을 고정해야 한다. 채택 문턱은 새 빈 판에서 실제 안전 포크가 0을 벗어나고 강한 상대 승리가 발생하며 "
        "복수 위협 도달률이 C보다 낮아지는 것이다. 이 문턱을 넘기 전에는 재배열 Fly·RNN과 다중 시드 확대보다 데이터 목표를 고치는 편이 타당하다.", "",
        "## 산출물", "",
        "- summary.json: 모든 수치와 판단의 기계 판독본",
        "- audit.json: 예산·체크포인트·라벨·분리·중복 감사",
        "- clean_common_test.json: 공통 오프닝 청정 보조 분석",
        "- evaluation.json: 원 검증·테스트와 전체 기보 연결 정보",
        "- c_dagger25/, d_threat_spatial/: 최종 C/D 체크포인트와 수집 자료", "",
        "이 결과는 Fly·GRU 단일 학습 시드 4810의 탐색 실험이다. 효과 크기의 시드 일반화는 아직 검증하지 않았다.",
    ]
    (args.root / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

if __name__ == "__main__":
    main()

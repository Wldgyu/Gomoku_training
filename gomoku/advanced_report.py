"""Build the sequential experiment report from saved measurements."""
from pathlib import Path
import argparse
import hashlib
import json
import statistics

from .make_report import NAMES, mean_sd, validate_replays, viewer


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def pct(value):
    return "—" if value is None else f"{100 * value:.2f}%"


def table(data, stage="final", kinds=("fly", "gru")):
    lines = ["| 모델 | 시드 | 즉시 승리 | 필수 방어 | 복수 위협 생성 | 복수 위협 예방 |",
             "|---|---:|---:|---:|---:|---:|"]
    for row in data["tactical_test"]:
        if row["stage"] == stage and row["kind"] in kinds:
            values = [pct(row["history"].get(key, {}).get("accuracy"))
                      for key in ("win", "block", "create_fork", "prevent_fork")]
            lines.append(f"| {NAMES[row['kind']]} | {row['seed']} | " + " | ".join(values) + " |")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=Path(__file__).parent / "runs/advanced_20261007")
    args = p.parse_args()
    root = args.input
    sections = ["# 12×12 일반 오목 — 기보·공간 인식·전술 확대 실험\n\n2026-10-07. 저장된 수치에서 자동 생성한 보고서입니다."]
    for title, folder in (("1. 상태0 보조 학습", "step1_cold"),
                          ("1. 기보 포함 보조 학습", "step1_history"),
                          ("2. 공통 공간 인코더와 칸별 출력", "step2_spatial"),
                          ("3. 공격·방어 문제 확대", "step3_expanded")):
        data = read(root / folder / "evaluation.json")
        sections.append(f"## {title}\n\n{table(data)}\n\n[원 측정값]({folder}/evaluation.json)")
    basic = read(root / "step3_expanded/evaluation_basic_validation.json")
    sections.append("### 3단계의 기존 검증 세트 성적\n\n" + table(basic)
                    + "\n\n2단계와 비교할 때 이 표의 동일한 판을 사용합니다. 확대 전술 표는 다른 검증 판입니다. "
                    "짧은600/300 예산에서는 기존 승리·방어 검증 정확도가2단계보다 낮아졌습니다. "
                    "범주별 기본 예제 수10,000→5,000과 생성 상대의 규칙 비율30%→50%도 함께 바뀌었습니다. "
                    "전술 범주 확대만으로 모든 기본 문제 성적이 높아진다고 결론낼 수 없습니다.")
    sections.append("## 대표 단계 비교의 조건\n\n"
        "모두 독립 초기화한 Fly·GRU, 검증 시드4500, 보조600×128과 PPO300×32×16, "
        "PPO 2에포크마다 보조128로 동일한 정답 수와 결정 수를 사용했습니다. "
        "기보 방식은 추가 관측을 순서대로 처리하므로 시간·연산량까지 같지는 않습니다. "
        "과거 관측은 정답 턴 이전의 자기 차례만 포함하며 역전파 없이 현재 순환 상태를 복원합니다. "
        "단일 시드의 탐색적 검증 결과이며 최종 시험을 단계 선택에 사용하지 않았습니다. "
        "공간 개선은 CNN과 출력 방식의 결합 변경이고 전체 파라미터 수가 기존 구조와 다릅니다. "
        "모든 대표 단계에서 완전 전술 상대 승률은 0%였습니다.")
    costs = ["| 단계 | 모델 | 파라미터 | 학습 시간(초) | 추가 기보 관측 | 할당/예약 GiB |",
             "|---|---|---:|---:|---:|---:|"]
    for folder in ("step1_cold", "step1_history", "step2_spatial", "step3_expanded"):
        for kind in ("fly", "gru"):
            r = read(root / folder / f"{kind}_seed4500.json")
            costs.append(f"| {folder} | {NAMES[kind]} | {r['parameters']:,} | "
                f"{r['elapsed_seconds']:.2f} | {r['auxiliary_history_frames']:,} | "
                f"{r['peak_allocated_gib']:.3f}/{r['peak_reserved_gib']:.3f} |")
    sections.append("### 비교 예산과 연산량\n\n" + "\n".join(costs)
        + "\n\n대표 실행마다 보조 정답153,600개와 PPO 결정153,600개. "
        "기보 관측은 정답 예제를 추가하는 것이 아니라 같은 예제의 앞선 관측을 복원하는 연산입니다. "
        "시간은 평가를 제외한 학습 측정이며 당시 병행 작업의 영향도 포함됩니다.")
    probe_games = ["| 단계 | 모델 | 약한 상대 승률 | 완전 전술 상대 승률 |", "|---|---|---:|---:|"]
    for folder in ("step1_cold", "step1_history", "step2_spatial", "step3_expanded"):
        measured = read(root / folder / "evaluation.json")
        for kind in ("fly", "gru"):
            values = [pct(next(r["win_rate"] for r in measured["fixed_opponents"]
                              if r["kind"] == kind and r["stage"] == "final" and r["opponent"] == o))
                      for o in ("weak", "tactical")]
            probe_games.append(f"| {folder} | {NAMES[kind]} | " + " | ".join(values) + " |")
    sections.append("### 대표 단계의 경기 결과\n\n" + "\n".join(probe_games)
        + "\n\n모델·조건마다128판(흑백64). 기보 상태의 전술 정확도가 올라가도 "
        "상태0 방식보다 약한 상대 승률은 낮아질 수 있었습니다. 전술 성적과 경기 성적을 따로 판단합니다.")
    full_control = root / "step1_full_history/evaluation.json"
    if full_control.exists():
        cold = read(root.parent / "defense12_20261006/evaluation_history_control_validation.json")
        history = read(full_control)
        assert cold["data_sha256"] == history["data_sha256"]
        import ast
        current = ast.parse((Path(__file__).parent / "models.py").read_text(encoding="utf-8"))
        archived = ast.parse((root.parent / "defense12_20261006/source_final/models.py").read_text(encoding="utf-8"))
        for name in ("Policy", "parameter_formula", "matched_hidden"):
            assert ast.dump(next(n for n in current.body if getattr(n, "name", "") == name)) == ast.dump(
                next(n for n in archived.body if getattr(n, "name", "") == name)), name
        checks = []
        for kind in ("fly", "gru"):
            old = read(root.parent / "defense12_20261006" / f"{kind}_seed4201.json")
            new = read(root / "step1_full_history" / f"{kind}_seed4201.json")
            assert old["graph_sha256"] == new["graph_sha256"]
            assert old["parameters"] == new["parameters"]
            assert old["completed_updates"] == new["completed_updates"] == 1200
            for key, value in old["config"].items():
                assert new["config"][key] == value, key
            for label in ("cold", "history"):
                old_fresh = next(r for r in cold["tactical_test"] if r["kind"] == kind and r["stage"] == "fresh")
                new_fresh = next(r for r in history["tactical_test"] if r["kind"] == kind and r["stage"] == "fresh")
                assert old_fresh[label] == new_fresh[label]
            checks.append(dict(kind=kind, seed=4201, budgets_equal=True, source_initialization_equal=True,
                parameters=new["parameters"], cold_elapsed_seconds=old["elapsed_seconds"],
                history_elapsed_seconds=new["elapsed_seconds"], additional_history_frames=new["auxiliary_history_frames"]))
        (root / "full_budget_control.json").write_text(json.dumps(dict(runs=checks,
            same_validation_data_sha256=history["data_sha256"], fresh_predictions_equal=True,
            checkpoint_selection="last fixed budget"), indent=2), encoding="utf-8")
        sections.append("## 1단계 추가 확인: 원 실험1800/1200 예산\n\n"
            "기존 상태0 Fly·GRU 시드4201과, 같은 초기 시드·전역 구조·기존 학습 데이터·약한 상대를 사용하는 "
            "새 기보 모델을 비교했습니다. 기보 보조 학습 방식만 바꿨고 모델당 정답537,600개·결정614,400개입니다. "
            "두 모델 모두 기존 검증 세트와 동일한 CPU 고정 평가 경로를 사용했습니다. "
            "추가 확인은 네 모델 본실험의 설정이나 체크포인트를 바꾸지 않았습니다.\n\n"
            "### 기존 상태0 모델\n\n" + table(cold)
            + "\n\n### 새 기보 모델\n\n" + table(history))
        lines = ["| 방식 | 모델 | 약한 상대 승률 | 완전 전술 상대 승률 |", "|---|---|---:|---:|"]
        for label, measured in (("상태0", cold), ("기보", history)):
            for kind in ("fly", "gru"):
                values = [pct(next(r["win_rate"] for r in measured["fixed_opponents"]
                                  if r["kind"] == kind and r["stage"] == "final" and r["opponent"] == o))
                          for o in ("weak", "tactical")]
                lines.append(f"| {label} | {NAMES[kind]} | " + " | ".join(values) + " |")
        sections.append("### 원 예산 대조의 실제 경기\n\n" + "\n".join(lines)
            + "\n\n모델·조건마다256판(흑백128), 같은 평가 난수 시드. "
            "단일 학습 시드의 추가 대조이며 일반적인 인과 효과의 통계적 확정은 아닙니다.")
    final = read(root / "final/evaluation.json")
    records = [read(root / "final" / f"{kind}_seed{seed}.json")
               for seed in (4601, 4602, 4603) for kind in NAMES]
    assert len(records) == 12 and all(r["completed_updates"] == 1200 for r in records)
    assert all(r["learner_decisions"] == 614400 for r in records)
    assert len({r["config"]["tactical_data_sha256"] for r in records}) == 1
    sections.append("## 4. 네 모델 × 3시드 본실험\n\n"
        "새 시드4601·4602·4603, 보조1800×128 후 PPO1200×32×16, "
        "2에포크마다 보조128: 모델당 보조 정답537,600개와 PPO 결정614,400개. "
        "상대 규칙 착수 비율30%→60%→85%의 커리큘럼을 추가했습니다. "
        "고정된 마지막 체크포인트를 새 확대 시험2,048판으로 평가했습니다. "
        "추론은 점유 칸만 차단한 신경망 argmax이며 교사 착수와 탐색을 사용하지 않습니다. "
        "최종 레시피는 구조·기보·데이터·상대 커리큘럼 전체 효과입니다.\n\n"
        + table(final, kinds=tuple(NAMES)))
    final_rows = [r for r in final["tactical_test"] if r["stage"] == "final"]
    equal_counts = all(r["cold"][key]["correct"] == r["history"][key]["correct"]
                       for r in final_rows for key in ("win", "block", "create_fork", "prevent_fork"))
    if equal_counts:
        sections.append("### 현재 순환 기억 해석의 제한\n\n"
            "최종12개 실행 모두 네 범주의 상태0·기보 입력 정답 수가 같았습니다. "
            "현재 과제는 전체 판을 관측하므로 공간 패턴만으로도 전술 문제의 정보를 얻을 수 있습니다. "
            "정답 수 일치가 개별 착수나 내부 상태까지 같다는 뜻은 아닙니다. "
            "이번 개선을 과거 기보 기억의 필수 사용이나 순환 연결 구조의 고유 효과로 해석할 수 없습니다.")
    lines = ["| 모델 | 무작위 승률 | 약한 상대 승률 | 완전 전술 상대 승률 |",
             "|---|---:|---:|---:|"]
    for kind in NAMES:
        values = []
        for opponent in ("random", "weak", "tactical"):
            rows = [r for r in final["fixed_opponents"] if r["kind"] == kind
                    and r["stage"] == "final" and r["opponent"] == opponent]
            assert len(rows) == 3
            values.append(mean_sd([r["win_rate"] for r in rows]))
        lines.append(f"| {NAMES[kind]} | " + " | ".join(values) + " |")
    sections.append("### 경기 승률\n\n" + "\n".join(lines)
        + "\n\n평균 ± 시드 간 표준편차. 모델·시드·상대마다256판(흑백128). "
        "초기·보조 완료·최종을 각각 평가했으며 위 표는 최종만 표시합니다. "
        "전술 문제 정답률과 경기 승률은 별개입니다. "
        "완전 전술 상대는 매 수 즉시 승리·필수 방어·5칸 구간의 연속 돌 점수를 사용하는 얕은 휴리스틱 정책입니다. "
        "약한 상대는 같은 정책을30%, 무작위 착수를70% 사용합니다.")
    lines = ["| 모델 | 실제 경기 즉시 승리 선택 | 실제 경기 필수 방어 선택 |", "|---|---:|---:|"]
    for kind in NAMES:
        values = []
        for key in ("win", "block"):
            rows = [r["tactical_decisions"][key] for r in final["fixed_opponents"]
                    if r["kind"] == kind and r["stage"] == "final" and r["opponent"] == "tactical"]
            count, correct = sum(r["positions"] for r in rows), sum(r["correct"] for r in rows)
            values.append(f"{correct}/{count} ({pct(correct / count)})" if count else "기회0회")
        lines.append(f"| {NAMES[kind]} | " + " | ".join(values) + " |")
    sections.append("### 강한 상대 경기의 공격·방어 선택\n\n" + "\n".join(lines)
                    + "\n\n관측된 결정 횟수를 합산했습니다. 기회가 없으면 공격 능력 정답률을 추정할 수 없습니다.")
    match_audit = root / "final/strong_match_audit.json"
    if match_audit.exists():
        audit = read(match_audit)
        assert audit["same_primary_games_verified"]
        lines = ["| 모델 | 실제 복수 위협 생성 | 실제 복수 위협 예방 | 유일 방어 누락 패배 | 이미 복수 승리 칸이 생긴 패배 |",
                 "|---|---:|---:|---:|---:|"]
        for kind in NAMES:
            rows = [r for r in audit["runs"] if r["kind"] == kind]
            values = []
            for key in ("create_fork", "prevent_fork"):
                n = sum(r["tactical_decisions"][key]["positions"] for r in rows)
                c = sum(r["tactical_decisions"][key]["correct"] for r in rows)
                values.append(f"{c}/{n} ({pct(c/n)})" if n else "기회0회")
            values.extend(str(sum(r["terminal_loss_causes"][key] for r in rows))
                          for key in ("missed_unique_block", "already_multiple_winning_cells"))
            lines.append(f"| {NAMES[kind]} | " + " | ".join(values) + " |")
        sections.append("### 같은 강한 상대 경기의 복수 위협과 마지막 패배 상황\n\n" + "\n".join(lines)
            + "\n\n기존 평가와 같은 모델·시드·256판을 재현하고 승패와 즉시 승리/방어 횟수 일치를 확인했습니다. "
            "열린3/복수 위협 예방을 독립 문제에서 맞히더라도 경기 분포의 선택률은 다를 수 있습니다. "
            "패배 분류는 마지막 자기 수 직전의 판이며 최초 전략 오류를 증명하는 분류는 아닙니다. "
            "두 승리 칸이 이미 생긴 경우 현재 한 수로 방어할 수 없습니다.\n\n[강한 상대 기보96개](strong_replays.html)")
    games = sum(r["games"] for r in final["head_to_head"])
    assert games == 2304
    lines = ["| A | B | A 승 | 무승부 | A 패 | A 승점 |", "|---|---|---:|---:|---:|---:|"]
    for a, b in ((a, b) for i, a in enumerate(NAMES) for b in list(NAMES)[i + 1:]):
        rows = [r for r in final["head_to_head"] if r["a"] == a and r["b"] == b]
        w, d, l = (sum(r[k] for r in rows) for k in ("wins", "draws", "losses"))
        lines.append(f"| {NAMES[a]} | {NAMES[b]} | {w} | {d} | {l} | {pct((w + .5*d)/(w+d+l))} |")
    sections.append("### 흑백 교환 맞대결\n\n" + "\n".join(lines)
        + "\n\n6쌍×3시드×64공통 시작판×흑백 교환=2,304판. "
        "상대 비교는 같은 구조와 예산 안에서 해석합니다. 시드마다 학습 난수와 재배선도 달라 "
        "재배선 자체의 효과를 분리한 대규모 통계 실험은 아닙니다.\n\n[실제 기보 보기](replays.html)")
    for title, path in (("기존 시험에 대한 새 모델 성적", root / "final/evaluation_original_test.json"),
                        ("새 확대 시험에 대한 이전 모델 성적", root.parent / "defense12_20261006/evaluation_expanded_test.json")):
        if path.exists():
            sections.append(f"## {title}\n\n" + table(read(path), kinds=tuple(NAMES))
                            + "\n\n고정 체크포인트의 사후 비교입니다. 이 성적으로 학습 설정을 바꾸지 않았습니다.")
    diagnostic = root / "diagnostics.json"
    if diagnostic.exists():
        rows = read(diagnostic)["runs"]
        lines = ["| 모델 | 학습 판 승리 | 학습 판 방어 | 검증 판 승리 | 검증 판 방어 |", "|---|---:|---:|---:|---:|"]
        for kind in NAMES:
            group = [r for r in rows if r["kind"] == kind]
            values = [mean_sd([r[scope][key]["accuracy"] for r in group])
                      for scope in ("train_seen_history", "validation_history") for key in ("win", "block")]
            lines.append(f"| {NAMES[kind]} | " + " | ".join(values) + " |")
        sections.append("## 학습 판과 검증 판의 잔여 미학습 진단\n\n" + "\n".join(lines)
            + "\n\n최종 체크포인트, 기보 상태, 고정된 범주별512개 학습 판과 범주별256개 검증 판입니다. "
            "학습 설정과 체크포인트 선택에 사용하지 않은 사후 진단입니다. [원 측정값](diagnostics.json)")
    strong = [r for r in final["fixed_opponents"] if r["stage"] == "final" and r["opponent"] == "tactical"]
    wins, draws, losses = (sum(r[k] for r in strong) for k in ("wins", "draws", "losses"))
    sections.append(f"## 현재 결론\n\n완전 전술 상대 총 {wins}승·{draws}무·{losses}패입니다. "
        "기보와 공간 패턴을 이용한 전술 판독 개선은 경기 승률과 함께 판단해야 합니다. "
        "이 결과만으로 Fly 연결 구조의 우월성이나 장기 전략을 획득했다고 결론내릴 수 없습니다. "
        "다음 실험은 실제 패배 기보에서 방어 누락·다음 위협 예측·공격 연결 실패를 분리하고 "
        "2~3수의 강제 수순 지도와 경기 상태 분포의 학습을 비교하는 것이 적절합니다.")
    sections.append("## 실행 자원과 재현\n\n"
        f"학습 시간 합 {sum(r['elapsed_seconds'] for r in records)/60:.2f}분; "
        f"PyTorch 최대 할당 {max(r['peak_allocated_gib'] for r in records):.3f}GiB, "
        f"최대 예약 {max(r['peak_reserved_gib'] for r in records):.3f}GiB. "
        "예약량은 할당량과 다르며 Windows 전체 GPU 사용량을 뜻하지 않습니다. "
        "GPU 학습과 CPU 고정 평가를 겹쳐 실행했으므로 시간은 겹치는 실행의 실제 측정치입니다. "
        "시드4603의 RNN·GRU는 별도 GPU 프로세스에서 병행하고 완료 파일을 합쳤으며, "
        "주 실행은 해당 완료 기록을 재사용했습니다. 각 모델의 예산과 핵심 소스 일치를 확인했습니다. "
        "RTX 5060 Ti 16GB, PyTorch2.11.0+cu128, TF32off. "
        "[사전 실험 계획](../../advanced_protocol.md), [원 평가 JSON](final/evaluation.json), "
        "[실행 메타데이터](final/training_summary.json). 과거 모델 가중치를 이식하지 않고 새로 학습했습니다.")
    replays = read(root / "final/replays.json")["games"]
    validate_replays(replays)
    (root / "replays.html").write_text(viewer(replays).replace("9×9", "12×12"), encoding="utf-8")
    strong_replay = root / "final/strong_replays.json"
    if strong_replay.exists():
        import re
        strong_games = read(strong_replay)["games"]
        validate_replays(strong_games)
        html = viewer(strong_games).replace("9×9", "12×12")
        html = html.replace("gru:'GRU'", "gru:'GRU',tactical:'전술 규칙 상대'")
        html = re.sub(r"<p>저장된 마지막 체크포인트.*?</p>",
                      "<p>고정된 최종 정책과 강한 전술 상대의 실제 평가 기보입니다. 모델이 백이면 상대가 먼저 한 수를 둡니다.</p>", html)
        html = html.replace("ply<=2?' · 공통 시작판':''", "ply===1&&g.initial_opponent_move?' · 상대의 첫 수':''")
        html = html.replace(" · 시작판 ${g.opening_index+1}", " · 경기 ${g.opening_index+1}")
        html = html.replace("일반 오목 · 실제 모델 맞대결", "일반 오목 · 강한 전술 상대 평가")
        (root / "strong_replays.html").write_text(html, encoding="utf-8")
    (root / "results.md").write_text("\n\n".join(sections) + "\n", encoding="utf-8")
    tracked = [p for p in root.rglob("*") if p.is_file() and p.suffix in (".json", ".npz", ".pt", ".md", ".html")]
    manifest = dict(date="2026-10-07", runs=12, validated_replays=len(replays),
                    validated_strong_replays=len(strong_games) if strong_replay.exists() else 0, league_games=games,
                    files={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in tracked if p.name != "completion_manifest.json"})
    (root / "completion_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(dict(report=str(root / "results.md"), runs=12, league_games=games,
                          validated_replays=len(replays)), ensure_ascii=False))


if __name__ == "__main__":
    main()

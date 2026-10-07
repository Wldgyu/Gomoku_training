"""Write concise project navigation using completed evaluation measurements."""
from pathlib import Path
import json
import statistics

ROOT = Path(__file__).resolve().parents[1]


def main():
    data = json.loads((ROOT / "gomoku/runs/advanced_20261007/final/evaluation.json").read_text(encoding="utf-8"))
    kinds = {"fly": "실제 Fly", "rewired": "재배선 Fly", "rnn": "RNN", "gru": "GRU"}
    lines = ["# MaleCNS 순환망 비교 연구", "", "2026-10-07 · 최신 완료 과제: 12×12 일반 오목의 기보·공간 인식·전술 확대.", "",
        "실제 Fly 연결 그래프·차수 보존 재배선 Fly·RNN·GRU를 같은 조건에서 비교합니다. "
        "카드 기억·다중 단서·방/복도 연구에서 출발해 일반 오목으로 확장했습니다. "
        "오목은 과거 모델 가중치를 재사용하지 않고 새로 학습했습니다.", "", "## 최신 결과", "",
        "대표 Fly·GRU의 동일 예산 대조를 상태0 → 기보 입력 → 공통 공간 인코더/출력 → 확대 전술 순서로 진행했습니다. "
        "이후 네 모델×새 시드3개를 각각 보조1800업데이트와 경기1200업데이트로 학습하고 흑백 교환2,304판을 평가했습니다.", "",
        "아래는 기보를 입력한 새 시험 판의 정확도와 강한 전술 상대 승률입니다. 정확도는 시드 평균입니다.", "",
        "| 모델 | 즉시 승리 | 필수 방어 | 복수 위협 생성 | 복수 위협 예방 | 강한 상대 승률 |",
        "|---|---:|---:|---:|---:|---:|"]
    for kind, name in kinds.items():
        rows = [r for r in data["tactical_test"] if r["kind"] == kind and r["stage"] == "final"]
        values = [statistics.mean(r["history"][key]["accuracy"] for r in rows) for key in ("win", "block", "create_fork", "prevent_fork")]
        strong = [r for r in data["fixed_opponents"] if r["kind"] == kind and r["stage"] == "final" and r["opponent"] == "tactical"]
        values.append(statistics.mean(r["win_rate"] for r in strong))
        lines.append(f"| {name} | " + " | ".join(f"{100*v:.2f}%" for v in values) + " |")
    strong = [r for r in data["fixed_opponents"] if r["stage"] == "final" and r["opponent"] == "tactical"]
    lines += ["", f"강한 상대 총 {sum(r['wins'] for r in strong)}승·{sum(r['draws'] for r in strong)}무·{sum(r['losses'] for r in strong)}패. "
        "전술 판독 개선을 경기 전략 완성이나 Fly 구조 우월성으로 해석하지 않습니다. "
        "다음 우선순위는 실제 실패 기보의 여러 수 위협 예측과 공격 연결 학습입니다.", "",
        "- [최신 결과 보고서](gomoku/reports/advanced_20261007/results.md) · [맞대결 기보](gomoku/reports/advanced_20261007/replays.html)",
        "- [단계별 사전 계획](gomoku/advanced_protocol.md) · [데이터·예산 감사](gomoku/runs/advanced_20261007/audit.json)",
        "- [첫 12×12 결과](gomoku/reports/defense12_20261006/results.md) · [9×9 파일럿](gomoku/reports/pilot_20261006/results.md)", "",
        "## 폴더와 읽는 순서", "", "| 위치 | 내용 |", "|---|---|",
        "| [gomoku/](gomoku/) | 오목 코드·실험 계획·실행 안내 |",
        "| [gomoku/reports/](gomoku/reports/) | 결과 보고서·평가 JSON·기보 |",
        "| gomoku/runs/ | 원 실행 자료·전술 데이터·로컬 가중치·당시 코드 |",
        "| [previous_research/](previous_research/) | 카드게임·기억·방/복도 연구 |",
        "| [previous_research/code/](previous_research/code/) | 이전 연구 코드의 주제별 안내 |",
        "| [previous_research/reports/](previous_research/reports/) | 주제별 이전 보고서 |",
        "| [카드 웹게임](previous_research/games/card_memory/index.html) | 원본 index.html·game.js |",
        "| [docs/README_HISTORY.md](docs/README_HISTORY.md) | 이전 날짜별 연구 성과 기록 |",
        "| [tools/](tools/) | 이동 검증·정리·업로드용 사본 생성 도구 |", "",
        "README·원 기획서·계획·프로토콜은 최상위에 유지했습니다. "
        "원자료와 기존 학습 체크포인트는 로컬 previous_research/data/에 보존했습니다.", "",
        "- [원 기획서](drosophila_project_plan_flow.txt) · [연구 실행 계획](RESEARCH_PROTOCOL.md) · [핵심 질문 결과](previous_research/reports/foundation/core_research_questions_report.txt)",
        "- [이전 연구 이동 목록](previous_research/manifests/layout_20261007.json) · [오목 이동 목록](docs/gomoku_layout_manifest.json)", "",
        "## 실행과 검증", "", "오목은 저장소 최상위에서, 이전 코드는 previous_research에서 실행합니다. "
        "자세한 학습 명령은 각 README에 있습니다. 24개 오목 테스트와 43개 기존 연구 테스트를 통과했습니다.", "",
        "~~~powershell", ".venv/Scripts/python.exe -m unittest gomoku.test_gomoku gomoku.test_defense gomoku.test_history_spatial -v",
        "~~~", "", "[핵심 의존성](requirements.txt) · [당시 전체 환경](gomoku/environment_versions.txt). "
        "일반룰은 흑백5개 이상 연속 승리이며 렌주 금수는 없습니다. 15×15는 이번에 학습하지 않았습니다.", "",
        "## GitHub 업로드 준비", "", "[업로드 안내](docs/GITHUB_UPLOAD.md). "
        "검토용 github_upload 폴더에는 코드·보고서·작은 재현 자료를 담았고, 대형 원자료·가중치·가상환경은 로컬에 보존합니다. "
        "PUBLICATION_MANIFEST.json으로 사본 해시와 제외 파일을 확인할 수 있습니다. 실제 GitHub 게시는 사용자 검토 후 진행합니다.", ""]
    (ROOT / "README.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()

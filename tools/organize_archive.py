"""Reversible legacy layout migration, with byte hashes and runnable paths."""
from __future__ import annotations
import hashlib
import json
import re
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]
ARCHIVE = WORKSPACE / "previous_research"


def checked(path):
    resolved = path.resolve()
    if not resolved.is_relative_to(WORKSPACE):
        raise ValueError(f"Path outside workspace: {resolved}")
    return resolved


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def topic(name):
    if "card" in name or "rnn_recovery" in name or name.startswith(("evaluate_rnn", "measure_rnn")):
        return "card_game"
    if "minigrid" in name:
        return "minigrid"
    if "multi_cue" in name or "visual_aux" in name or "order_holdout" in name:
        return "memory"
    return "foundation"


def main():
    existing = ARCHIVE / "manifests/layout_20261007.json"
    if existing.exists():
        print("Legacy layout already migrated; existing manifest retained.")
        return
    files = sorted(p for p in ARCHIVE.iterdir() if p.is_file())
    reports = {p.name: topic(p.name) for p in files if p.suffix == ".txt" and not p.name.startswith("requirements")}
    data_before = {str(p.relative_to(ARCHIVE)): (p.stat().st_size, p.stat().st_mtime_ns)
                   for p in (ARCHIVE / "data").rglob("*") if p.is_file()}
    entries = []
    for source in files:
        if source.suffix in (".py", ".ps1"):
            target = ARCHIVE / "code" / source.name
        elif source.name.startswith("requirements"):
            target = ARCHIVE / "requirements" / source.name
        elif source.name in reports:
            target = ARCHIVE / "reports" / reports[source.name] / source.name
        elif source.suffix == ".json":
            target = ARCHIVE / "manifests" / source.name
        else:
            target = ARCHIVE / "documents" / source.name
        source, target = checked(source), checked(target)
        if target.exists():
            raise FileExistsError(target)
        before = sha(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        source.rename(target)
        entries.append(dict(old=str(source.relative_to(WORKSPACE)).replace("\\", "/"),
                            new=str(target.relative_to(WORKSPACE)).replace("\\", "/"),
                            sha256_before=before, sha256_after_move=sha(target), edited=False))
    for entry in entries:
        path = WORKSPACE / entry["new"]
        if path.suffix not in (".py", ".ps1"):
            continue
        before_bytes = path.read_bytes()
        text = before_bytes.decode("utf-8")
        text = text.replace("ROOT = Path(__file__).resolve().parent",
                            "ROOT = Path(__file__).resolve().parent.parent")
        text = re.sub(r'ROOT / ("[^"\n]+\.py")', r'ROOT / "code" / \1', text)
        text = text.replace("ROOT / script", 'ROOT / "code" / script')
        text = text.replace('(ROOT.parent if name.endswith("_protocol.txt") else ROOT) / name',
                            '(ROOT.parent if name.endswith("_protocol.txt") else ROOT / "code") / name')
        for name, category in reports.items():
            text = text.replace(f'ROOT / "{name}"', f'ROOT / "reports" / "{category}" / "{name}"')
        if path.suffix == ".ps1":
            text = text.replace('(Split-Path $PSScriptRoot -Parent) ".venv',
                                '(Split-Path (Split-Path $PSScriptRoot -Parent) -Parent) ".venv')
            text = text.replace('$w, $PSScriptRoot,', '$w, (Split-Path $PSScriptRoot -Parent),')
            text = text.replace("& $py multi_cue_memory.py", "& $py code\\multi_cue_memory.py")
        if text.encode("utf-8") != before_bytes:
            backup = ARCHIVE / "manifests" / "source_before_layout" / path.name
            checked(backup).parent.mkdir(parents=True, exist_ok=True)
            backup.write_bytes(before_bytes)
            path.write_bytes(text.encode("utf-8"))
            entry.update(edited=True, sha256_after_repair=sha(path),
                         original_source=str(backup.relative_to(WORKSPACE)).replace("\\", "/"))
    for name in ("__pycache__", ".ruff_cache"):
        source, target = checked(ARCHIVE / name), checked(ARCHIVE / ".cache" / name)
        if source.exists():
            if target.exists():
                raise FileExistsError(target)
            target.parent.mkdir(parents=True, exist_ok=True)
            source.rename(target)
    data_after = {str(p.relative_to(ARCHIVE)): (p.stat().st_size, p.stat().st_mtime_ns)
                  for p in (ARCHIVE / "data").rglob("*") if p.is_file()}
    if data_before != data_after:
        raise RuntimeError("Historical data inventory changed")
    checked(existing).parent.mkdir(parents=True, exist_ok=True)
    existing.write_text(json.dumps(dict(date="2026-10-07", entries=entries,
        data_files=len(data_before), data_bytes=sum(v[0] for v in data_before.values()),
        data_size_and_mtime_unchanged=True,
        moved_files_verified=all(e["sha256_before"] == e["sha256_after_move"] for e in entries)),
        ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for document in (WORKSPACE / "README.md", WORKSPACE / "RESEARCH_PROTOCOL.md"):
        original = document.read_text(encoding="utf-8")
        if document.name == "README.md":
            history = WORKSPACE / "docs" / "README_HISTORY.md"
            checked(history).parent.mkdir(parents=True, exist_ok=True)
            history.write_text(original, encoding="utf-8")
        updated = original
        for entry in entries:
            updated = updated.replace(entry["old"], entry["new"])
        document.write_text(updated, encoding="utf-8")
    labels = {"card_game": "카드게임", "minigrid": "방·복도 / MiniGrid", "memory": "다중 단서 기억", "foundation": "그래프·기초 검증"}
    lines = ["# 이전 연구 자료", "", "카드게임·방/복도·다중 단서 기억 실험의 코드와 보고서입니다.", "",
             "## 폴더", "", "| 경로 | 내용 |", "|---|---|",
             "| [code/](code/) | 기존 Python 코드와 실행 스크립트 |",
             "| [reports/](reports/) | 주제별 결과 보고서 |",
             "| [data/](data/) | 원자료·학습 기록·체크포인트·당시 소스 |",
             "| [requirements/](requirements/) | 단계별 의존성 목록 |",
             "| [manifests/](manifests/) | 이동 목록·해시·경로 수정 전 코드 |", "",
             "## 보고서", ""]
    for category, label in labels.items():
        lines += [f"### {label}", ""]
        for name, cat in sorted(reports.items()):
            if cat == category:
                lines.append(f"- [{name}](reports/{category}/{name})")
        lines.append("")
    lines += ["## 기존 코드 실행", "", "작업 폴더는 previous_research로 둡니다.", "",
              "~~~powershell", "Set-Location previous_research",
              r"..\.venv\Scripts\python.exe code\pilot_card_game.py --help",
              r"..\.venv\Scripts\python.exe -m unittest discover -s code -p 'test_*.py'",
              "~~~", "", "현재 경로는 수정했으며 당시 원본은 데이터의 source 폴더와 manifests/source_before_layout에 보존했습니다.", ""]
    (ARCHIVE / "README.md").write_text("\n".join(lines), encoding="utf-8")
    code_lines = ["# 이전 연구 코드 찾아보기", "", "기존 import를 유지하도록 코드는 한 폴더에 두고 주제별로 안내합니다.", ""]
    for category, label in labels.items():
        code_lines += [f"## {label}", ""]
        for p in sorted((ARCHIVE / "code").glob("*")):
            if p.is_file() and topic(p.name) == category:
                code_lines.append(f"- [{p.name}]({p.name})")
        code_lines.append("")
    (ARCHIVE / "code/README.md").write_text("\n".join(code_lines), encoding="utf-8")
    (ARCHIVE / "data/README.md").write_text(
        "# 이전 연구 데이터\n\n"
        "- game_pilot/: 카드게임 학습·평가·RNN 회복 실험\n"
        "- minigrid*/: 방·복도·관계 선택·자율 이동 실험\n"
        "- multi_cue*/와 grid_multi_cue/: 다중 단서 기억·보조 학습 실험\n"
        "- subgraphs/: 실제·재배선 Fly 부분 그래프와 출처 메타데이터\n"
        "- 최상위 feather 파일: 공식 MaleCNS 원자료\n\n"
        "pt는 학습 체크포인트, json은 설정·학습 곡선·평가, source는 당시 코드입니다.\n",
        encoding="utf-8")
    print(json.dumps(dict(moved=len(entries), repaired=sum(e["edited"] for e in entries),
                          data_files_unchanged=len(data_before)), ensure_ascii=False))


if __name__ == "__main__":
    main()

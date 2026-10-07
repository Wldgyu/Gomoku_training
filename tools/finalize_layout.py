"""Finalize Gomoku layout only after all authorized experiments complete."""
from pathlib import Path
import hashlib
import json
import re
import shutil

ROOT = Path(__file__).resolve().parents[1]


def checked(path):
    result = path.resolve()
    if not result.is_relative_to(ROOT):
        raise ValueError(result)
    return result


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    old, new = checked(ROOT / "gomoku9"), checked(ROOT / "gomoku")
    if new.exists():
        raise FileExistsError("Layout already finalized; preserve existing files.")
    final = old / "runs/advanced_20261007/final"
    for seed in (4601, 4602, 4603):
        for kind in ("fly", "rewired", "rnn", "gru"):
            r = json.loads((final / f"{kind}_seed{seed}.json").read_text())
            assert r["completed_updates"] == 1200
    for name in ("evaluation.json", "evaluation_original_test.json", "replays.json"):
        assert (final / name).is_file(), name
    assert (old / "runs/advanced_20261007/completion_manifest.json").is_file()
    assert (old / "runs/advanced_20261007/step1_full_history/evaluation.json").is_file()
    snapshot = old / "runs/advanced_20261007/source"
    snapshot.mkdir(exist_ok=True)
    for path in old.iterdir():
        if path.suffix in (".py", ".md", ".txt"):
            shutil.copy2(path, snapshot / path.name)
    before = {str(p.relative_to(old)): sha(p) for p in old.rglob("*")
              if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".log"}
    old.rename(new)
    assert all(sha(new / name) == value for name, value in before.items())
    test = new / "test_defense.py"
    content = test.read_text(encoding="utf-8")
    content = content.replace('patch("gomoku9.train.time.sleep")', 'patch(f"{__package__}.train.time.sleep")')
    test.write_text(content, encoding="utf-8")
    migration = json.loads((ROOT / "previous_research/manifests/layout_20261007.json").read_text(encoding="utf-8"))
    mapping = {r["old"]: r["new"] for r in migration["entries"]}
    history = ROOT / "docs/README_HISTORY.md"
    text = history.read_text(encoding="utf-8")
    for a, b in mapping.items():
        text = text.replace(a, b)
    text = text.replace("gomoku9", "gomoku")
    text = re.sub(r"\]\((?!https?://|#|\.\./)([^)]+)\)", r"](../\1)", text)
    history.write_text(text, encoding="utf-8")
    for path in [ROOT / "README.md", ROOT / "RESEARCH_PROTOCOL.md", *new.glob("*.md")]:
        if path.exists():
            text = path.read_text(encoding="utf-8")
            if path.name != "README.md" or path.parent != new:
                text = text.replace("gomoku9", "gomoku")
            for a, b in mapping.items():
                text = text.replace(a, b)
            path.write_text(text, encoding="utf-8")
    copies = []
    for run_name in ("pilot_20261006", "defense12_20261006", "advanced_20261007"):
        source = new / "runs" / run_name
        target = new / "reports" / run_name
        for path in source.rglob("*"):
            relative = path.relative_to(source)
            if not path.is_file() or path.suffix not in (".md", ".html", ".json", ".csv"):
                continue
            if any(part.startswith("source") for part in relative.parts):
                continue
            dest = checked(target / relative)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
            assert sha(path) == sha(dest)
            copies.append(dict(source=str(path.relative_to(ROOT)), report=str(dest.relative_to(ROOT)), sha256=sha(path)))
    reports = new / "reports/README.md"
    reports.write_text("# 오목 결과 보고서\n\n"
        "- [최신: 기보·공간 패턴·전술 확대](advanced_20261007/results.md) · [기보](advanced_20261007/replays.html)\n"
        "- [첫 12×12 방어 학습](defense12_20261006/results.md) · [기보](defense12_20261006/replays.html)\n"
        "- [9×9 파일럿](pilot_20261006/results.md) · [기보](pilot_20261006/replays.html)\n\n"
        "읽기용 보고서와 평가 JSON을 모았습니다. 원 실행 데이터·가중치·당시 소스는 ../runs/에 보존합니다. "
        "보고서 사본은 원본과 해시가 같습니다. 실험 당시 경로의 gomoku9는 현재 gomoku입니다.\n", encoding="utf-8")
    logs = new / "logs"
    logs.mkdir(exist_ok=True)
    for path in new.glob("*.log"):
        path.rename(checked(logs / path.name))
    cache = new / "__pycache__"
    if cache.exists():
        cache_target = checked(new / ".cache/__pycache__")
        cache_target.parent.mkdir(exist_ok=True)
        assert not cache_target.exists()
        cache.rename(cache_target)
    record = dict(date="2026-10-07", prefix_before="gomoku9", prefix_after="gomoku",
                  original_file_hashes=before, reports=copies,
                  note="Metrics and checkpoints unchanged; live docs and one test mock path repaired.")
    (ROOT / "docs/gomoku_layout_manifest.json").write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(dict(renamed=str(new), preserved_files=len(before), report_files=len(copies)), ensure_ascii=False))


if __name__ == "__main__":
    main()

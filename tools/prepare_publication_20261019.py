"""Non-destructive update of the user's existing Gomoku_training checkout (github_upload_20261013); never stage/commit/push.

2026-10-19: syncs the integrated research report, v5~v8 and controlled-comparison results, new teacher/data code and the
recorded run scripts. Writes a train/test key-overlap audit. Model weights, logs and engine binaries stay local.
"""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import hashlib
import json
import os

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "github_upload_20261013"
LIMIT = 8 * 2**20
RUNS = ("opening_followup_diverse_20261014", "balanced_curriculum_20261014",
        "defense_probe_20261014", "renju15_20261015")
SKIP_DIRS = {".git", "__pycache__", ".venv", ".cache", ".ruff_cache", "external", "source", "source_code",
             "before_v7", "before_v8", "before_priority_fix"}
CACHE_DIRS = {".git", "__pycache__", ".venv", ".cache", ".ruff_cache", "external"}
DOCS = ("RESEARCH_REPORT_20261019.md", "RESEARCH_STATUS_20261015.md",
        "RESEARCH_STATUS_20261016.md", "RESEARCH_STATUS_20261017.md", "RESULTS_AUDIT_20261015.json",
        "RESULTS_AUDIT_20261016.json", "RESULTS_AUDIT_20261019.json", "GITHUB_UPLOAD.md", "DATA_SOURCES.md",
        "OPENING_AND_NEXT_STEPS_20261014.md")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n", encoding="utf-8", newline="\n")


def portable(value):
    if isinstance(value, dict):
        return {portable(k): portable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [portable(v) for v in value]
    if isinstance(value, str):
        for prefix in (str(ROOT)+"\\", ROOT.as_posix()+"/"):
            if value.startswith(prefix):
                return value[len(prefix):].replace("\\", "/")
    return value


def audit_overlap():
    folder = ROOT / "gomoku/runs/renju15_20261015/data"
    summary, sets = {}, {}
    for path in sorted(folder.glob("*.npz")):
        with np.load(path) as data:
            if "keys" not in data.files:
                continue
            keys = data["keys"].astype(str)
            cats, counts = np.unique(data["category"], return_counts=True)
            sets[path.stem] = set(keys.tolist())
            summary[path.stem] = dict(records=len(keys), unique_keys=len(sets[path.stem]),
                categories={str(c): int(n) for c, n in zip(cats, counts)}, sha256=digest(path), bytes=path.stat().st_size)
    overlaps = []
    for train in (n for n in sets if n.endswith("train")):
        for test in (n for n in sets if n.endswith("test")):
            count = len(sets[train] & sets[test])
            overlaps.append(dict(train=train, test=test, overlap=count))
    bad = [o for o in overlaps if o["overlap"]]
    audit = dict(date="2026-10-19", generated_at=datetime.now(timezone(timedelta(hours=9))).isoformat(),
                 scope="Train/test canonical-position-key intersections of every stored 15x15 array; no training or evaluation was run",
                 files=summary, train_test_key_intersections=overlaps, pairs_with_overlap=bad,
                 note="Stored position keys only; not an exhaustive trajectory-independence proof. Files without a keys array are skipped.")
    write(ROOT / "docs/RESULTS_AUDIT_20261019.json", audit)
    return bad


def main():
    if not (DEST/".git").is_dir():
        raise RuntimeError("Expected the user's existing Git repository; refusing to create or replace it")
    git_files = [DEST/".git"/name for name in ("HEAD", "index", "config")]
    before = {str(p): digest(p) for p in git_files if p.exists()}
    bad = audit_overlap()
    selected = {Path(r) for r in ("README.md", "requirements.txt", ".gitattributes")}
    selected |= {Path("docs")/name for name in DOCS}
    for path in (ROOT/"gomoku").iterdir():
        if path.is_file() and path.suffix in (".py", ".md", ".html", ".txt"):
            selected.add(path.relative_to(ROOT))
    for path in (ROOT/"gomoku/renju").iterdir():
        if path.is_file() and path.suffix in (".py", ".md", ".html"):
            selected.add(path.relative_to(ROOT))
    for folder in RUNS:
        base = ROOT/"gomoku/runs"/folder
        for directory, dirs, files in os.walk(base):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
            in_scripts = Path(directory).name == "scripts"
            for name in files:
                path = Path(directory)/name
                allowed = (".json", ".md", ".npz") + ((".ps1", ".py") if in_scripts else ())
                if path.suffix not in allowed or name in ("manifest.json", "completion_manifest.json"):
                    continue
                if path.stat().st_size <= LIMIT:
                    selected.add(path.relative_to(ROOT))
    for rel in ("previous_research/data/subgraphs/malecns_cx_2950.npz", "previous_research/data/subgraphs/malecns_cx_2950.json",
                "previous_research/data/subgraphs/malecns_cx_2950_rewired_seed0.npz",
                "previous_research/data/subgraphs/malecns_cx_2950_rewired_seed1.npz",
                "previous_research/code/build_subgraphs.py", "previous_research/code/verify_subgraphs.py",
                "previous_research/code/rewire_subgraphs.py", "previous_research/code/project_config.py",
                "tools/prepare_publication_20261015.py", "tools/prepare_publication_20261019.py"):
        selected.add(Path(rel))
    changed = []
    for rel in sorted(selected):
        source, target = ROOT/rel, DEST/rel
        if not source.is_file():
            raise FileNotFoundError(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        raw = source.read_bytes()
        if source.suffix == ".json":
            raw = (json.dumps(portable(json.loads(raw.decode("utf-8"))), ensure_ascii=False, indent=2)+"\n").encode("utf-8")
        if source.suffix in (".py", ".md", ".json", ".html", ".txt", ".ps1") or source.name == ".gitattributes":
            raw = raw.replace(b"\r\n", b"\n")
        if not target.exists() or target.read_bytes() != raw:
            target.write_bytes(raw)
            changed.append(rel.as_posix())
    ignore = DEST/".gitignore"
    text = ignore.read_text(encoding="utf-8")
    for pattern in ("/gomoku/external/", "*.pth", "*.onnx", "*.exe", "*.dll", "*.7z"):
        if pattern not in text.splitlines():
            text += "\n"+pattern
    ignore.write_text(text.rstrip()+"\n", encoding="utf-8", newline="\n")
    items = []
    for directory, dirs, files in os.walk(DEST):
        dirs[:] = [d for d in dirs if d not in CACHE_DIRS]
        for name in sorted(files):
            path = Path(directory)/name
            if name == "PUBLICATION_MANIFEST.json":
                continue
            rel = path.relative_to(DEST).as_posix()
            if path.suffix in (".pt", ".pth", ".feather", ".pyc", ".log", ".tmp", ".exe", ".dll", ".7z"):
                raise RuntimeError(f"Unexpected non-public artifact in checkout: {rel}")
            if path.stat().st_size > 20*2**20:
                raise RuntimeError(f"File exceeds publication size policy: {rel}")
            items.append(dict(path=rel, bytes=path.stat().st_size, sha256=digest(path)))
    items.sort(key=lambda r: r["path"])
    manifest = dict(date="2026-10-19", snapshot=True, included=items, total_included_bytes=sum(r["bytes"] for r in items),
        excluded_files_remain_local=True, automatic_github_upload=False, git_metadata_excluded=True,
        excluded_policy=[".git internals, virtual environments, caches", "model checkpoints (.pt/.pth), raw feather data",
                         "logs and temporary files", "external engine executables/weights and bundled dependencies",
                         "large proof trees and historical local artifact manifests",
                         "raw human game files (analysis summaries and position arrays are included)"],
        updated_source_files=len(selected), changed_source_files=changed,
        portability="Current workspace absolute paths in copied JSON are made repository-relative; copied text uses LF; numeric results unchanged",
        note="Existing checkout and historical published files preserved; no stage/commit/push; manifest excludes itself")
    write(DEST/"PUBLICATION_MANIFEST.json", manifest)
    if before != {str(p): digest(p) for p in git_files if p.exists()}:
        raise RuntimeError("Git metadata unexpectedly changed")
    print(json.dumps(dict(updated=len(changed), included_files=len(items), MiB=round(manifest["total_included_bytes"]/2**20, 2),
                          overlap_pairs=len(bad), git_metadata_unchanged=True), ensure_ascii=False))


if __name__ == "__main__":
    main()

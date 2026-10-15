"""Non-destructive update of the user's existing GitHub checkout; never stage/commit/push."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import hashlib
import json
import os
import shutil

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "github_upload_20261013"
LIMIT = 8 * 2**20
RUNS = ("opening_followup_diverse_20261014", "balanced_curriculum_20261014",
        "defense_probe_20261014", "renju15_20261015")
SKIP_DIRS = {".git", "__pycache__", ".venv", ".cache", ".ruff_cache", "external", "source", "source_code"}
CACHE_DIRS = SKIP_DIRS - {"source", "source_code"}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n", encoding="utf-8", newline="\n")


def portable(value):
    if isinstance(value, dict):
        return {portable(k): portable(v) for k,v in value.items()}
    if isinstance(value, list):
        return [portable(v) for v in value]
    if isinstance(value, str):
        for prefix in (str(ROOT)+"\\", ROOT.as_posix()+"/"):
            if value.startswith(prefix):
                return value[len(prefix):].replace("\\", "/")
    return value


def audit_results():
    r15 = ROOT / "gomoku/runs/renju15_20261015"
    r12 = ROOT / "gomoku/runs/defense_probe_20261014"
    audit = dict(date="2026-10-15", generated_at=datetime.now(timezone(timedelta(hours=9))).isoformat(),
                 scope="Saved-result audit; no additional model training or match evaluation",
                 source_tests=dict(count=35, status="passed", suites=["gomoku.renju.test_rules", "gomoku.renju.test_tactics",
                    "gomoku.renju.test_env", "gomoku.renju.test_data", "gomoku.renju.test_play", "gomoku.test_defense_probe", "gomoku.test_external_engine"]),
                 publication_copy_tests=dict(count=35,status="passed",date="2026-10-15",
                    scope="Manually verified from github_upload_20261013 during this snapshot; helper does not run tests"),
                 renju15=dict(training={}, live=load(r15/"live_v1.json"),league=load(r15/"league_v1.json")),
                 freestyle12={}, datasets={}, source_files=[], local_checkpoint_inventory=[])
    for name in ("fly2950_v1", "rnn3000_v1"):
        data = load(r15/"models"/f"{name}.json")
        audit["renju15"]["training"][name] = dict(updates=data["curve"][-1]["update"],
            seconds=data["curve"][-1]["seconds"],parameters=data["parameters"],tests=data["tests"],
            validation_first=data["curve"][0],validation_last=data["curve"][-1])
    for kind in ("fly", "gru", "rnn", "rewired"):
        seed = load(r12/f"{kind}_seed5101_live.json")["results"]
        quiet = load(r12/f"{kind}_quiet_live.json")["results"]["quiet"]
        def totals(modes):
            games=sum(r["games"] for r in modes.values()); wins=sum(r["wins"] for r in modes.values())
            losses=sum(r["losses"] for r in modes.values())
            return dict(games=games,wins=wins,losses=losses,draws=games-wins-losses)
        audit["freestyle12"][kind] = dict(original=totals(seed["original"]),wide=totals(seed["seed5101"]),quiet=totals(quiet))
    for folder in (r12/"data",r15/"data"):
        summary={}; sets={}
        for path in sorted(folder.glob("*.npz")):
            with np.load(path) as data:
                keys=data["keys"].astype(str); cats,counts=np.unique(data["category"],return_counts=True)
                sets[path.stem]=set(keys.tolist())
                summary[path.stem]=dict(records=len(keys),unique_keys=len(sets[path.stem]),
                    categories={str(c):int(n) for c,n in zip(cats,counts)},sha256=digest(path),bytes=path.stat().st_size)
        overlaps=[]
        for train in (n for n in sets if n.endswith("train")):
            for test in (n for n in sets if n.endswith("test")):
                count=len(sets[train]&sets[test])
                if count: raise RuntimeError(f"Train/test overlap: {folder}/{train}/{test}: {count}")
                overlaps.append(dict(train=train,test=test,overlap=count))
        audit["datasets"][folder.relative_to(ROOT).as_posix()]=dict(files=summary,train_test_key_intersections=overlaps,
                scope="stored canonical position keys; not an exhaustive trajectory independence proof")
    for folder in (r12,r15):
        for path in sorted(folder.rglob("*.json")):
            if "smoke" in path.parts: continue
            audit["source_files"].append(dict(path=path.relative_to(ROOT).as_posix(),sha256=digest(path)))
    checkpoints=list((r15/"models").glob("*.pt"))+list(r12.glob("scratch_*_w64x6_atk_q.pt"))
    for path in sorted(checkpoints):
        audit["local_checkpoint_inventory"].append(dict(path=path.relative_to(ROOT).as_posix(),bytes=path.stat().st_size,
                                                        sha256=digest(path),included_in_publication=False))
    write(ROOT/"docs/RESULTS_AUDIT_20261015.json",audit)


def main():
    if not (DEST/".git").is_dir():
        raise RuntimeError("Expected the user's existing Git repository; refusing to create or replace it")
    git_files=[DEST/".git"/name for name in ("HEAD","index","config")]
    before={str(p):digest(p) for p in git_files if p.exists()}
    audit_results()
    selected=set()
    for rel in ("README.md","requirements.txt",".gitattributes"):
        selected.add(Path(rel))
    for rel in ("docs/RESEARCH_STATUS_20261015.md","docs/RESULTS_AUDIT_20261015.json","docs/GITHUB_UPLOAD.md",
                "docs/OPENING_AND_NEXT_STEPS_20261014.md"):
        selected.add(Path(rel))
    for path in (ROOT/"gomoku").iterdir():
        if path.is_file() and path.suffix in (".py",".md",".html",".txt"):
            selected.add(path.relative_to(ROOT))
    for path in (ROOT/"gomoku/renju").iterdir():
        if path.is_file() and path.suffix in (".py",".md",".html"):
            selected.add(path.relative_to(ROOT))
    for folder in RUNS:
        base=ROOT/"gomoku/runs"/folder
        for directory,dirs,files in os.walk(base):
            dirs[:]=[d for d in dirs if d not in SKIP_DIRS]
            for name in files:
                path=Path(directory)/name
                if path.suffix not in (".json",".md",".npz") or name in ("manifest.json","completion_manifest.json"):
                    continue
                if path.stat().st_size<=LIMIT:
                    selected.add(path.relative_to(ROOT))
    for rel in ("previous_research/data/subgraphs/malecns_cx_2950.npz","previous_research/data/subgraphs/malecns_cx_2950.json",
                "previous_research/code/build_subgraphs.py","previous_research/code/verify_subgraphs.py",
                "previous_research/code/rewire_subgraphs.py","previous_research/code/project_config.py",
                "tools/prepare_publication_20261015.py"):
        selected.add(Path(rel))
    changed=[]
    for rel in sorted(selected):
        source,target=ROOT/rel,DEST/rel
        if not source.is_file(): raise FileNotFoundError(source)
        target.parent.mkdir(parents=True,exist_ok=True)
        raw=source.read_bytes()
        if source.suffix==".json":
            raw=(json.dumps(portable(json.loads(raw.decode("utf-8"))),ensure_ascii=False,indent=2)+"\n").encode("utf-8")
        if source.suffix in (".py", ".md", ".json", ".html", ".txt") or source.name==".gitattributes":
            raw=raw.replace(b"\r\n", b"\n")
        if not target.exists() or target.read_bytes()!=raw:
            target.write_bytes(raw); changed.append(rel.as_posix())
    ignore=DEST/".gitignore"
    text=ignore.read_text(encoding="utf-8")
    for pattern in ("/gomoku/external/","*.pth","*.onnx","*.exe","*.dll","*.7z"):
        if pattern not in text.splitlines(): text+="\n"+pattern
    ignore.write_text(text.rstrip()+"\n",encoding="utf-8",newline="\n")
    items=[]
    for directory,dirs,files in os.walk(DEST):
        dirs[:]=[d for d in dirs if d not in CACHE_DIRS]
        for name in sorted(files):
            path=Path(directory)/name
            if name=="PUBLICATION_MANIFEST.json": continue
            rel=path.relative_to(DEST).as_posix()
            if path.suffix in (".pt",".pth",".feather",".pyc",".log",".tmp",".exe",".dll",".7z"):
                raise RuntimeError(f"Unexpected non-public artifact in checkout: {rel}")
            if path.stat().st_size>20*2**20: raise RuntimeError(f"File exceeds publication size policy: {rel}")
            items.append(dict(path=rel,bytes=path.stat().st_size,sha256=digest(path)))
    items.sort(key=lambda r:r["path"])
    manifest=dict(date="2026-10-15",snapshot=True,included=items,total_included_bytes=sum(r["bytes"] for r in items),
        excluded_files_remain_local=True,automatic_github_upload=False,git_metadata_excluded=True,
        excluded_policy=[".git internals, virtual environments, caches", "model checkpoints (.pt/.pth), raw feather data",
                         "logs and temporary files", "external engine executables/weights and bundled dependencies",
                         "large proof trees and historical local artifact manifests"],
        updated_source_files=len(selected),changed_source_files=changed,
        portability="Current workspace absolute paths in copied JSON are made repository-relative; copied text uses LF; numeric results unchanged",
        note="Existing checkout and historical published files preserved; no stage/commit/push; manifest excludes itself")
    write(DEST/"PUBLICATION_MANIFEST.json",manifest)
    if before!={str(p):digest(p) for p in git_files if p.exists()}:
        raise RuntimeError("Git metadata unexpectedly changed")
    print(json.dumps(dict(updated=len(changed),included_files=len(items),MiB=round(manifest["total_included_bytes"]/2**20,2),
                          git_metadata_unchanged=True),ensure_ascii=False))


if __name__=="__main__": main()

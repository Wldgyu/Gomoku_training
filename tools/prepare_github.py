"""Build a reviewable GitHub upload tree; preserve large local artifacts."""
from pathlib import Path
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "github_upload"
SKIP_DIRS = {".venv", ".git", "__pycache__", ".ruff_cache", ".cache", "github_upload", "node_modules"}
SKIP_EXT = {".pt", ".feather", ".log", ".tmp", ".pyc"}


def checked(path):
    result = path.resolve()
    if not result.is_relative_to(ROOT):
        raise ValueError(f"Outside workspace: {result}")
    return result


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if DEST.exists():
        raise FileExistsError("Review folder exists; do not overwrite manually edited publication files.")
    files = sorted(p for p in ROOT.rglob("*") if p.is_file()
                   and not any(part in SKIP_DIRS for part in p.relative_to(ROOT).parts))
    included, excluded = [], []
    checked(DEST).mkdir()
    for source in files:
        relative = source.relative_to(ROOT)
        size = source.stat().st_size
        if source.suffix.lower() in SKIP_EXT or size > 20 * 2**20:
            excluded.append(dict(path=relative.as_posix(), bytes=size,
                                 reason="local checkpoint/raw data/log/temp or >20MiB"))
            continue
        target = checked(DEST / relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(checked(source), target)
        hash_value = digest(source)
        if digest(target) != hash_value:
            raise RuntimeError(f"Copy mismatch: {relative}")
        included.append(dict(path=relative.as_posix(), bytes=size, sha256=hash_value))
    manifest = dict(date="2026-10-07", included=included, excluded=excluded,
                    excluded_files_remain_local=True, automatic_github_upload=False,
                    total_included_bytes=sum(r["bytes"] for r in included),
                    total_excluded_bytes=sum(r["bytes"] for r in excluded))
    (DEST / "PUBLICATION_MANIFEST.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (ROOT / "docs/publication_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(dict(folder=str(DEST), files=len(included), excluded=len(excluded),
                          included_mib=manifest["total_included_bytes"] / 2**20), ensure_ascii=False))


if __name__ == "__main__":
    main()

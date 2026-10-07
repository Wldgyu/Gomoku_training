"""Check portable publication bytes, omissions and live document links."""
from pathlib import Path
import hashlib
import json
import re
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
PUBLICATION = ROOT / "github_upload"


def main():
    manifest = json.loads((PUBLICATION / "PUBLICATION_MANIFEST.json").read_text(encoding="utf-8"))
    for record in manifest["included"]:
        path = PUBLICATION / record["path"]
        assert path.resolve().is_relative_to(PUBLICATION.resolve())
        assert path.is_file(), path
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"], path
    bad, checked = [], 0
    for path in PUBLICATION.rglob("*.md"):
        relative = path.relative_to(PUBLICATION)
        if any(part.startswith("source") for part in relative.parts):
            continue
        text = path.read_text(encoding="utf-8")
        for match in re.finditer(r"\]\(([^)]+)\)", text):
            target = match[1].strip().strip("<>")
            if "://" in target or target.startswith("#"):
                continue
            target = unquote(target.split("#")[0])
            if not target:
                continue
            checked += 1
            if not (path.parent / target).exists():
                bad.append(dict(document=relative.as_posix(), target=target))
    if bad:
        print(json.dumps(dict(broken_links=bad), ensure_ascii=False, indent=2))
        raise SystemExit(1)
    forbidden = [str(p.relative_to(PUBLICATION)) for p in PUBLICATION.rglob("*")
                 if p.is_file() and (p.suffix in (".pt", ".feather", ".log", ".pyc") or p.stat().st_size > 20 * 2**20)]
    assert not forbidden, forbidden
    print(json.dumps(dict(verified_files=len(manifest["included"]), checked_local_links=checked,
                          missing_links=0, large_or_cache_files=0)))


if __name__ == "__main__":
    main()

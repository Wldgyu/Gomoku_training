"""Move fully completed independent runs into the common final directory."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
CORE = ("train.py", "env.py", "models.py", "spatial.py", "history_aux.py", "tactics.py", "expanded_tactics.py")


def checked(path):
    resolved = path.resolve()
    if not resolved.is_relative_to(ROOT):
        raise ValueError(resolved)
    return resolved


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    base = ROOT / "gomoku9/runs/advanced_20261007"
    source, final = checked(base / "parallel_seed4603"), checked(base / "final")
    reference = json.loads((final / "rnn_seed4601.json").read_text(encoding="utf-8"))
    for kind in ("rnn", "gru"):
        record = json.loads((source / f"{kind}_seed4603.json").read_text(encoding="utf-8"))
        assert record["completed_updates"] == 1200 and record["learner_decisions"] == 614400
        for key, value in reference["config"].items():
            if key not in ("kind", "seed", "rewire_seed", "hidden"):
                assert record["config"][key] == value, key
        assert all(record["source_hashes"][name] == reference["source_hashes"][name] for name in CORE)
    assert (source / "training_summary.json").is_file()
    entries = []
    for path in source.iterdir():
        if path.name == "training_summary.json":
            target = checked(base / "parallel_training_summary.json")
        else:
            assert path.name.startswith(("rnn_seed4603", "gru_seed4603"))
            target = checked(final / path.name)
        assert path.is_file() and not target.exists(), target
        value = sha(path)
        entry = dict(source=str(path.relative_to(ROOT)), destination=str(target.relative_to(ROOT)), sha256=value)
        path.rename(target)
        assert sha(target) == value
        entries.append(entry)
    source.rmdir()
    record = dict(date="2026-10-07", independent_gpu_process=True,
                  models=["rnn_seed4603", "gru_seed4603"], budgets_and_core_source_equal=True,
                  main_training_reuses_completed_runs=True, files=entries)
    (base / "parallel_execution.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(dict(merged=len(entries), models=record["models"])))


if __name__ == "__main__":
    main()

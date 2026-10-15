"""Print a compact table of every defense-probe live evaluation and static result."""
from pathlib import Path
import json

RUN = Path(__file__).resolve().parent / "runs/defense_probe_20261014"


def main():
    print("== live (wins/losses/games, multi-threat reach, block, prevent_fork) ==")
    for path in sorted(RUN.glob("*_live.json")):
        data = json.loads(path.read_text())
        for tag, modes in data["results"].items():
            cells = []
            for mode, r in modes.items():
                cells.append(f"{mode}: {r['wins']}W/{r['losses']}L/{r['games']} reach={r['reached_multiple_threat']} "
                             f"blk={r['block'][0]}/{r['block'][1]} prv={r['prevent_fork'][0]}/{r['prevent_fork'][1]}")
            print(f"{path.name:36s} seeds={data['seeds']} {tag:14s} | " + " | ".join(cells))
    print("== static (accuracy) ==")
    for path in sorted(RUN.glob("scratch_*.json")):
        data = json.loads(path.read_text())
        tests = {n: round(r["all"]["accuracy"], 3) for n, r in data["tests"].items()}
        print(f"{path.name:36s} {tests}")


if __name__ == "__main__":
    main()

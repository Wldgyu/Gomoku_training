"""Replay saved human-vs-model Renju games and check every AI move against the exact tactical teacher."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import rules, tactics
from .env import BatchBoard
from .rules import CELLS, SIZE


def label(action):
    return f"{action // SIZE + 1}행{action % SIZE + 1}열"


def render(board, last=None):
    rows = []
    for r in range(SIZE):
        row = []
        for c in range(SIZE):
            v = board[r * SIZE + c]
            ch = "X" if v == 1 else "O" if v == -1 else "."
            row.append(f"[{ch}]" if last == r * SIZE + c else f" {ch} ")
        rows.append(f"{r + 1:2d}" + "".join(row))
    return "   " + "".join(f"{c + 1:^3d}" for c in range(SIZE)) + "\n" + "\n".join(rows)


def analyze(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    moves = [(m["row"] - 1) * SIZE + m["column"] - 1 for m in data["moves"]]
    human = 1 if data["human_color"] == "black" else -1
    board = BatchBoard(1)
    mistakes, ai_turns, ai_correct = [], 0, 0
    human_threats = []
    for ply, action in enumerate(moves):
        color = 1 if ply % 2 == 0 else -1
        labels, category = tactics.analyze(board.board[0], color)
        cat = int(category)
        if color != human and cat:
            ai_turns += 1
            ok = bool(labels[action])
            ai_correct += ok
            if not ok:
                mistakes.append(dict(ply=ply + 1, category=tactics.NAMES[cat], played=label(action),
                                     expected=[label(int(a)) for a in np.flatnonzero(labels)][:6]))
        elif color == human and cat in (1, 3, 5) and labels[action]:
            human_threats.append((ply + 1, tactics.NAMES[cat]))
        board.place(np.array([action]), color)
    winner = int(board.winner[0])
    result = ("사람 승" if winner == human else "AI 승") if winner else "무승부/미종료"
    if board.forbidden_loss[0]:
        result += " (흑 금수패)"
    return dict(file=path.name, model=data["model"], human=data["human_color"], plies=len(moves), result=result,
                ai_tactical_turns=ai_turns, ai_tactical_correct=ai_correct, ai_mistakes=mistakes,
                human_attacks=human_threats, moves=moves,
                last_moves=[f"{'흑' if i % 2 == 0 else '백'} {label(a)}" for i, a in enumerate(moves)][-8:],
                final_board=render(board.board[0], moves[-1]))


def game_positions(folder, reserved=frozenset()):
    """Every position with an exact label (either side to move) from the saved games, as dataset records."""
    from .data import canonical, position_obs
    records, seen = [], set(reserved)
    for path in sorted(folder.glob("renju-*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        moves = [(m["row"] - 1) * SIZE + m["column"] - 1 for m in data["moves"]]
        board = BatchBoard(1)
        history = np.full(CELLS, -1, dtype=np.int16)
        for ply, action in enumerate(moves):
            color = 1 if ply % 2 == 0 else -1
            labels, category = tactics.analyze(board.board[0], color)
            key = canonical(board.board[0], color)
            if category and labels.any() and key not in seen:
                seen.add(key)
                last = moves[ply - 1] if ply else -1
                records.append((position_obs(board.board[0], color, last), labels, rules.legal_mask(board.board[0], color),
                                int(category), key, history.copy()))
            history[ply] = action
            board.place(np.array([action]), color)
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--positions", type=Path, help="save exact-label positions of these games as a dataset")
    args = parser.parse_args()
    if args.positions:
        import numpy as _np
        from .data import save
        reserved = set()
        for name in ("test", "quiet_test", "onpolicy_test"):
            p = args.positions.parent / f"{name}.npz"
            if p.exists():
                with _np.load(p) as saved:
                    reserved |= set(saved["keys"].tolist())
        records = game_positions(args.folder, reserved)
        save(args.positions, records)
        print("saved", len(records), "positions", {c: sum(r[3] == c for r in records) for c in range(1, 7)})
    results = [analyze(p) for p in sorted(args.folder.glob("renju-*.json"))]
    for r in results:
        print(f"\n=== {r['file']} | {r['model']} | 사람={r['human']} | {r['result']} | {r['plies']}수")
        print(f"AI 전술 판단 {r['ai_tactical_correct']}/{r['ai_tactical_turns']} | 사람의 공격 수: {r['human_attacks']}")
        print("마지막 수:", ", ".join(r["last_moves"]))
        for m in r["ai_mistakes"]:
            print(f"  AI 실수 {m['ply']}수째 [{m['category']}] 둔 곳 {m['played']} / 정답 예시 {m['expected']}")
        print(r["final_board"])
    if args.out:
        args.out.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

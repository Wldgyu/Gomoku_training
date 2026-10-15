"""Replay saved human-vs-model games and check every AI move against the exact tactical teachers."""
from pathlib import Path
import argparse
import json

import numpy as np

from .env import BatchBoard
from .history_aux import CATEGORIES
from .multistep_tactics import multi_targets

NAMES = dict(CATEGORIES) | {0: "quiet(no exact label)"}


def label(action):
    return f"{action // 12 + 1}행{action % 12 + 1}열"


def render(board, last=None):
    rows = []
    for r in range(12):
        row = []
        for c in range(12):
            v = board[r * 12 + c]
            ch = "X" if v == 1 else "O" if v == -1 else "."
            row.append(f"[{ch}]" if last == r * 12 + c else f" {ch} ")
        rows.append(f"{r + 1:2d}" + "".join(row))
    return "    " + "".join(f"{c + 1:^3d}" for c in range(12)) + "\n" + "\n".join(rows)


def analyze(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    moves = [m["row"] * 12 - 12 + m["column"] - 1 for m in data["moves"]]
    human = 1 if data["human_color"] == "black" else -1
    board = BatchBoard(1, 12)
    mistakes, ai_turns, ai_correct = [], 0, 0
    for ply, action in enumerate(moves):
        color = 1 if ply % 2 == 0 else -1
        labels, category = multi_targets(board, np.array([color]))
        cat = int(category[0])
        if color != human and cat:
            ai_turns += 1
            ok = bool(labels[0, action])
            ai_correct += ok
            if not ok:
                mistakes.append(dict(ply=ply + 1, category=NAMES[cat], played=label(action),
                                     expected=[label(int(a)) for a in np.flatnonzero(labels[0])][:6]))
        board.place(np.array([action]), color)
    winner = int(board.winner[0])
    return dict(file=path.name, model=data["model"], human=data["human_color"], plies=len(moves),
                result="사람 승" if winner == human else "AI 승" if winner else "무승부/미종료",
                ai_tactical_turns=ai_turns, ai_tactical_correct=ai_correct, ai_mistakes=mistakes,
                last_moves=[f"{'흑' if i % 2 == 0 else '백'} {label(a)}" for i, a in enumerate(moves)][-8:],
                final_board=render(board.board[0], moves[-1]), moves=moves)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    results = [analyze(p) for p in sorted(args.folder.glob("*.json"))]
    for r in results:
        print(f"\n=== {r['file']} | {r['model']} | 사람={r['human']} | {r['result']} | {r['plies']}수")
        print(f"AI의 전술 판단 {r['ai_tactical_correct']}/{r['ai_tactical_turns']}, 마지막 수: {', '.join(r['last_moves'])}")
        for m in r["ai_mistakes"]:
            print(f"  AI 실수 {m['ply']}수째 [{m['category']}] 둔 곳 {m['played']} / 정답 예시 {m['expected']}")
        print(r["final_board"])
    if args.out:
        args.out.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

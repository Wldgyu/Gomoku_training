"""Bounded Piskvork process adapter; coordinates are x=column, y=row."""
from pathlib import Path
import queue
import re
import subprocess
import threading
import time

import numpy as np


class PiskvorkEngine:
    def __init__(self, command, size=12, turn_ms=200, cwd=None, rule=0):
        self.size, self.turn_ms = size, turn_ms
        self.lines = queue.Queue()
        flags = subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0
        self.process = subprocess.Popen([str(x) for x in command], cwd=cwd, stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                        text=True, bufsize=1, creationflags=flags)
        def reader():
            for line in self.process.stdout:
                self.lines.put(line.strip())
            self.lines.put(None)
        self.reader_thread = threading.Thread(target=reader, daemon=True)
        self.reader_thread.start()
        self.transcript = []
        try:
            self.send(f"START {size}")
            self.wait(lambda s: s.startswith("OK"), 15)
            for line in (f"INFO rule {rule}", f"INFO timeout_turn {turn_ms}", "INFO time_left 60000",
                         "INFO max_memory 67108864", "INFO thread_num 1"):
                self.send(line)
        except Exception:
            self.close()
            raise

    def send(self, line):
        self.process.stdin.write(line+"\n")
        self.process.stdin.flush()

    def wait(self, predicate, timeout):
        end = time.monotonic()+timeout
        rows = []
        while time.monotonic() < end:
            try:
                line = self.lines.get(timeout=max(0.01, end-time.monotonic()))
            except queue.Empty as exc:
                raise TimeoutError("External engine response deadline exceeded") from exc
            if line is None:
                raise RuntimeError("External engine exited before replying")
            self.transcript.append(line)
            rows.append(line)
            if line.startswith("ERROR"):
                raise RuntimeError(line)
            if predicate(line):
                return line, rows
        raise TimeoutError("External engine response deadline exceeded")

    def recommend(self, cells, color, history=None):
        cells = np.asarray(cells, dtype=np.int8).reshape(-1)
        if len(cells) != self.size**2 or color not in (-1, 1) or not np.isin(cells, [-1,0,1]).all():
            raise ValueError("Invalid board/color")
        if not (cells == 0).any():
            raise ValueError("Cannot analyze a full board")
        black = np.flatnonzero(cells == 1).tolist()
        white = np.flatnonzero(cells == -1).tolist()
        if not (len(black) == len(white) or len(black) == len(white)+1):
            raise ValueError("Stone counts do not describe a legal alternating game")
        expected_color = 1 if len(black) == len(white) else -1
        if color != expected_color:
            raise ValueError("Requested color is not the side to move")
        if history is None:
            # Rapfi replays BOARD entries, including passes for consecutive sides.
            # Alternating absolute colors keeps the final side to move correct.
            moves = [a for pair in zip(black, white) for a in pair]
            if len(black) > len(white):
                moves.append(black[-1])
        else:
            moves = [int(a) for a in np.asarray(history).reshape(-1) if int(a) >= 0]
            rebuilt = np.zeros_like(cells)
            for ply, action in enumerate(moves):
                if not 0 <= action < len(cells) or rebuilt[action]:
                    raise ValueError("Invalid or duplicate history move")
                rebuilt[action] = 1 if ply % 2 == 0 else -1
            if not np.array_equal(rebuilt, cells):
                raise ValueError("History does not reconstruct the supplied board")
        self.send("RESTART")
        self.wait(lambda s:s.startswith("OK"), 5)
        self.send("BOARD")
        for action in moves:
            row, column = divmod(int(action), self.size)
            self.send(f"{column},{row},{1 if cells[action]==color else 2}")
        self.send("DONE")
        started = time.perf_counter()
        line, messages = self.wait(lambda s:re.fullmatch(r"\d+\s*,\s*\d+", s) is not None,
                                   max(5, self.turn_ms/1000+3))
        column, row = map(int, line.split(","))
        if not (0 <= row < self.size and 0 <= column < self.size):
            raise RuntimeError("Engine returned an out-of-board move")
        action = row*self.size+column
        if cells[action] != 0:
            raise RuntimeError("Engine returned an occupied move")
        evaluations = [int(v) for s in messages for v in re.findall(r"\bEval\s+(-?\d+)", s)]
        disabled = any("Evaluator" in s and "disabled" in s for s in self.transcript)
        return dict(action=action, row=row, column=column,
                    evaluation=evaluations[-1] if evaluations else None,
                    seconds=time.perf_counter()-started, messages=messages,
                    evaluator_disabled=disabled)

    def close(self):
        if self.process.poll() is None:
            try:
                self.send("END")
                self.process.wait(timeout=3)
            except (OSError, subprocess.TimeoutExpired):
                self.process.kill()
                self.process.wait(timeout=3)
        if self.process.stdin:
            self.process.stdin.close()
        self.reader_thread.join(timeout=1)
        if self.process.stdout:
            self.process.stdout.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

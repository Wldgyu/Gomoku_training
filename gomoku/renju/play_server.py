"""Local human-versus-checkpoint 15x15 Renju. No training, teacher or search at runtime.

  cd C:\\Users\\a\\Desktop\\MaleCNS
  .venv\\Scripts\\python.exe -m gomoku.renju.play_server --port 8766
"""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import argparse
import json
import threading
import uuid

import numpy as np
import torch

from ..models import load_policy
from . import rules
from .env import BatchBoard, force_black_center
from .rules import CELLS, CENTER, SIZE
from .train import burn_in

HTML = Path(__file__).resolve().parent / "play.html"
MODELS = Path(__file__).resolve().parents[1] / "runs/renju15_20261015/models"
KIND_NAMES = {"fly": "Fly", "rnn": "RNN", "gru": "GRU", "rewired": "재배열 Fly"}


def checkpoints():
    found = {}
    for path in sorted(MODELS.glob("*.pt")) if MODELS.exists() else []:
        found[path.stem] = path
    return found


def display_name(stem):
    kind = stem.split("_")[0].rstrip("0123456789")
    return f"{KIND_NAMES.get(kind, kind.upper())} · {stem}"


class Game:
    def __init__(self, kind, human):
        if kind not in checkpoints() or human not in (-1, 1):
            raise ValueError("모델과 흑백을 선택해주세요.")
        self.kind, self.human = kind, human
        self.moves, self.board = [], BatchBoard(1)

    @property
    def turn(self):
        return 1 if len(self.moves) % 2 == 0 else -1

    def forbidden(self):
        if self.board.finished[0] or self.turn != 1:
            return []
        return np.flatnonzero(rules.forbidden_mask(self.board.board[0])).tolist()

    def place(self, action, by_human=False):
        if self.board.finished[0]:
            raise ValueError("대국이 끝났습니다. 새 대국을 시작해주세요.")
        if isinstance(action, bool) or not isinstance(action, int) or not 0 <= action < CELLS:
            raise ValueError("판 안의 교차점을 선택해주세요.")
        if self.board.board[0, action]:
            raise ValueError("이미 돌이 놓인 자리입니다.")
        if not self.moves and action != CENTER:
            raise ValueError("흑의 첫 수는 중앙의 표시된 자리에 놓아주세요.")
        if by_human and self.turn == 1 and action in self.forbidden():
            raise ValueError("금수 자리입니다(장목·쌍삼·쌍사). 흑은 이 자리에 둘 수 없습니다.")
        self.board.place(np.array([action]), self.turn)
        self.moves.append(action)

    @torch.no_grad()
    def reply(self, model):
        if self.board.finished[0] or self.turn == self.human:
            return
        history = torch.full((1, CELLS), -1, dtype=torch.long)
        history[0, :len(self.moves)] = torch.tensor(self.moves, dtype=torch.long)
        obs = torch.from_numpy(self.board.observe(self.turn))
        state = burn_in(model, history)
        logits, _, _ = model.step(obs, state, torch.zeros(1, dtype=torch.bool))
        legal = torch.from_numpy(self.board.legal(self.turn))
        raw = logits.masked_fill(~legal, -1e9).argmax(1).numpy()
        self.place(int(force_black_center(self.board, self.turn, raw)[0]))

    def undo(self):
        if not any((1 if i % 2 == 0 else -1) == self.human for i in range(len(self.moves))):
            return
        last_human = max(i for i in range(len(self.moves)) if (1 if i % 2 == 0 else -1) == self.human)
        self.moves = self.moves[:last_human]
        self.board = BatchBoard(1)
        for ply, action in enumerate(self.moves):
            self.board.place(np.array([action]), 1 if ply % 2 == 0 else -1)

    def state(self):
        return dict(kind=self.kind, name=display_name(self.kind), human=self.human,
                    board=self.board.board[0].tolist(), moves=self.moves.copy(), turn=self.turn,
                    finished=bool(self.board.finished[0]), winner=int(self.board.winner[0]),
                    forbidden_loss=bool(self.board.forbidden_loss[0]), forbidden=self.forbidden(), size=SIZE)


class Application:
    def __init__(self):
        self.models, self.games = {}, {}
        self.lock = threading.Lock()

    def model(self, kind):
        if kind not in self.models:
            self.models[kind] = load_policy(checkpoints()[kind], torch.device("cpu"))[0]
        return self.models[kind]

    def process(self, endpoint, data):
        with self.lock:
            if endpoint == "/api/new":
                kind, human = data.get("kind"), data.get("human", 1)
                game = Game(kind, human)
                game.reply(self.model(kind))
                if len(self.games) >= 100:
                    self.games.pop(next(iter(self.games)))
                identifier = uuid.uuid4().hex
                self.games[identifier] = game
            else:
                identifier = data.get("id")
                if identifier not in self.games:
                    raise ValueError("새 대국을 시작해주세요.")
                game = self.games[identifier]
                if endpoint == "/api/move":
                    if game.turn != game.human:
                        raise ValueError("AI의 차례입니다.")
                    game.place(data.get("action"), by_human=True)
                    game.reply(self.model(game.kind))
                elif endpoint == "/api/undo":
                    game.undo()
                else:
                    raise ValueError("지원하지 않는 요청입니다.")
            return dict(id=identifier, **game.state())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    torch.set_num_threads(2)
    application = Application()

    class Handler(BaseHTTPRequestHandler):
        def send(self, status, body, content_type="application/json; charset=utf-8"):
            payload = body.encode("utf-8") if isinstance(body, str) else json.dumps(body, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            if self.path == "/":
                self.send(200, HTML.read_text(encoding="utf-8"), "text/html; charset=utf-8")
            elif self.path == "/api/health":
                self.send(200, dict(status="ready", models={k: display_name(k) for k in checkpoints()}))
            elif self.path == "/favicon.ico":
                self.send(204, "")
            else:
                self.send(404, dict(error="페이지를 찾을 수 없습니다."))

        def do_POST(self):
            origin = self.headers.get("Origin")
            if origin and origin not in (f"http://127.0.0.1:{self.server.server_port}", f"http://localhost:{self.server.server_port}"):
                self.send(403, dict(error="대국 화면에서 요청해주세요."))
                return
            try:
                length = int(self.headers.get("Content-Length", 0))
                if not 0 < length <= 4096 or not self.headers.get("Content-Type", "").startswith("application/json"):
                    raise ValueError("잘못된 요청입니다.")
                data = json.loads(self.rfile.read(length))
                if not isinstance(data, dict):
                    raise ValueError("잘못된 요청입니다.")
                self.send(200, application.process(self.path, data))
            except (ValueError, TypeError) as error:
                self.send(400, dict(error=str(error)))
            except Exception:
                self.send(500, dict(error="모델 실행에 실패했습니다. 대국을 다시 시작해주세요."))

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Renju ready: http://127.0.0.1:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

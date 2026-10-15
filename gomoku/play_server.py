"""Local human-versus-checkpoint Gomoku. No training, teacher, or search at runtime."""
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import argparse
import json
import threading
import uuid

import numpy as np
import torch

from .env import BatchBoard, black_center_action, force_black_center
from .history_aux import burn_in
from .models import load_policy
from .opening_followup import PATHS

HTML = Path(__file__).resolve().parent / "play.html"
NAMES = {"gru": "GRU", "fly": "Fly", "rnn": "RNN", "rewired": "재배열 Fly"}
# 넓은 트렁크 후보(2026-10-14 방어 점검): 원본과 별도 체크포인트이며 원본은 교체하지 않는다.
WIDE_DIR = Path(__file__).resolve().parent / "runs/defense_probe_20261014"
CHECKPOINTS = dict(PATHS)
for _kind, _label in list(NAMES.items()):
    _path = WIDE_DIR / f"scratch_{_kind}_w64x6_atk.pt"
    if _path.exists():
        CHECKPOINTS[f"{_kind}_wide"] = _path
        NAMES[f"{_kind}_wide"] = f"{_label} (넓은 트렁크)"
# 초중반 공격 자료(Rapfi 라벨)를 더한 후보. 파일이 있는 모델만 노출한다.
for _kind, _label in [(k, v) for k, v in NAMES.items() if not k.endswith("_wide")]:
    _path = WIDE_DIR / f"scratch_{_kind}_w64x6_atk_q.pt"
    if _path.exists():
        CHECKPOINTS[f"{_kind}_quiet"] = _path
        NAMES[f"{_kind}_quiet"] = f"{_label} (넓은 트렁크 + 초중반 공격 자료)"


class Game:
    def __init__(self, kind, human):
        if kind not in CHECKPOINTS or human not in (-1, 1):
            raise ValueError("모델과 흑백을 선택해주세요.")
        self.kind, self.human = kind, human
        self.moves = []
        self.board = BatchBoard(1, 12)

    @property
    def turn(self):
        return 1 if len(self.moves) % 2 == 0 else -1

    def place(self, action):
        if self.board.finished[0]:
            raise ValueError("대국이 끝났습니다. 새 대국을 시작해주세요.")
        if isinstance(action, bool) or not isinstance(action, int) or not 0 <= action < 144:
            raise ValueError("판 안의 교차점을 선택해주세요.")
        if self.board.board[0, action]:
            raise ValueError("이미 돌이 놓인 자리입니다.")
        if not self.moves and action != black_center_action(12):
            raise ValueError("흑의 첫 수는 중앙의 표시된 자리에 놓아주세요.")
        self.board.place(np.array([action]), self.turn)
        self.moves.append(action)

    @torch.no_grad()
    def reply(self, model):
        if self.board.finished[0] or self.turn == self.human:
            return
        history = torch.full((1, 144), -1, dtype=torch.long)
        history[0, :len(self.moves)] = torch.tensor(self.moves, dtype=torch.long)
        obs = torch.from_numpy(self.board.observe(self.turn))
        state = burn_in(model, history)
        logits, _, _ = model.step(obs, state, torch.zeros(1, dtype=torch.bool))
        legal = torch.from_numpy(self.board.legal())
        raw = logits.masked_fill(~legal, -1e9).argmax(1).numpy()
        action = int(force_black_center(self.board, self.turn, raw)[0])
        self.place(action)

    def undo(self):
        minimum = 1 if self.human == -1 else 0
        if len(self.moves) <= minimum:
            return
        last_human = max(i for i in range(len(self.moves)) if (1 if i%2==0 else -1)==self.human)
        self.moves = self.moves[:last_human]
        self.board = BatchBoard(1, 12)
        for ply, action in enumerate(self.moves):
            self.board.place(np.array([action]), 1 if ply%2==0 else -1)

    def state(self):
        return dict(kind=self.kind, name=NAMES[self.kind], human=self.human,
                    board=self.board.board[0].tolist(), moves=self.moves.copy(), turn=self.turn,
                    finished=bool(self.board.finished[0]), winner=int(self.board.winner[0]), size=12)


class Application:
    def __init__(self):
        self.models, self.games = {}, {}
        self.lock = threading.Lock()

    def model(self, kind):
        if kind not in self.models:
            self.models[kind] = load_policy(CHECKPOINTS[kind], torch.device("cpu"))[0]
        return self.models[kind]

    def process(self, endpoint, data):
        with self.lock:
            if endpoint == "/api/new":
                kind, human = data.get("kind", "gru"), data.get("human", 1)
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
                    game.place(data.get("action"))
                    game.reply(self.model(game.kind))
                elif endpoint == "/api/undo":
                    game.undo()
                else:
                    raise ValueError("지원하지 않는 요청입니다.")
            return dict(id=identifier, **game.state())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
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
                self.send(200, dict(status="ready", models=list(NAMES)))
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
    print(f"Gomoku ready: http://127.0.0.1:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

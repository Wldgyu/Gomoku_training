"""Regenerate the pilot report and an offline viewer of actual match replays."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np

from .env import BatchBoard
from .models import KINDS
from .train import DEFAULT_OUT

NAMES = {"fly": "실제 Fly", "rewired": "재배선 Fly", "rnn": "RNN", "gru": "GRU"}


def mean_sd(values: list[float]) -> str:
    return f"{statistics.mean(values) * 100:.2f} ± {statistics.stdev(values) * 100:.2f}%" if len(values) > 1 else f"{values[0] * 100:.2f}%"


def validate_replays(games: list[dict]) -> None:
    for game in games:
        board = BatchBoard(1, game.get("size", 9))
        for i, cell in enumerate(game["moves"]):
            if board.finished[0]:
                raise RuntimeError("Replay contains a move after the game ended")
            board.place(np.array([cell]), 1 if i % 2 == 0 else -1)
        if not board.finished[0] or int(board.winner[0]) != game["winner"]:
            raise RuntimeError("Replay winner does not match the actual moves")


def viewer(games: list[dict]) -> str:
    data = json.dumps(games, ensure_ascii=False).replace("</", "<\\/")
    return '''<!doctype html>
<html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>9×9 오목 모델 맞대결 기보</title>
<style>
body{margin:0;background:#f4f4f1;color:#24302d;font-family:system-ui,sans-serif}main{max-width:960px;margin:28px auto;padding:20px}
h1{font-size:24px}p{line-height:1.6}.layout{display:flex;gap:28px;flex-wrap:wrap}canvas{width:min(90vw,480px);height:min(90vw,480px);border-radius:12px;background:#dfbf85;box-shadow:0 6px 20px #0002}
aside{flex:1;min-width:240px}select{width:100%;padding:10px;border:1px solid #bbb;border-radius:6px}button{padding:10px 16px;border:1px solid #bcc5c0;border-radius:6px;background:white;margin:8px 4px 0 0;cursor:pointer}input{width:100%}.card{background:white;padding:16px;border-radius:10px;margin:16px 0}a{color:#16685a}#status{font-weight:600}
</style><main><h1>9×9 일반 오목 · 실제 모델 맞대결</h1>
<p>저장된 마지막 체크포인트의 실제 착수입니다. 같은 무작위 시작 2수를 두고 모델의 흑백을 바꿔 평가했습니다. 처음 2수는 모델이 선택한 수가 아닙니다.</p>
<div class="layout"><canvas id="board" width="600" height="600" aria-label="오목 기보"></canvas><aside>
<label for="game">경기 선택</label><select id="game"></select><div class="card"><div id="players"></div><p id="status"></p><div id="result"></div></div>
<input id="seek" type="range" min="0" value="0" aria-label="수순"><div><button id="first">처음</button><button id="prev">이전</button><button id="play">재생</button><button id="next">다음</button><button id="last">끝</button></div>
<p>돌 안 숫자는 착수 순서이며, 붉은 테두리는 마지막 착수입니다.</p><a href="results.md">결과 보고서</a></aside></div></main>
<script>
const games=__DATA__, names={fly:'실제 Fly',rewired:'재배선 Fly',rnn:'RNN',gru:'GRU'};
const canvas=document.getElementById('board'),ctx=canvas.getContext('2d'),sel=document.getElementById('game'),seek=document.getElementById('seek');
let gi=0,ply=0,timer=null;
games.forEach((g,i)=>{let o=document.createElement('option');o.value=i;o.textContent=`${names[g.a]} / ${names[g.b]} · 시드 ${g.seed} · ${g.a_color===1?'A 흑':'A 백'} · 시작판 ${g.opening_index+1}`;sel.append(o)});
function stop(){clearInterval(timer);timer=null;document.getElementById('play').textContent='재생'}
function draw(){const g=games[gi],black=g.a_color===1?g.a:g.b,white=g.a_color===-1?g.a:g.b;seek.max=g.moves.length;seek.value=ply;
ctx.clearRect(0,0,600,600);ctx.fillStyle='#dfbf85';ctx.fillRect(0,0,600,600);ctx.strokeStyle='#604a2a';ctx.lineWidth=1.5;const start=50,size=g.size||9,gap=500/(size-1);
for(let i=0;i<size;i++){ctx.beginPath();ctx.moveTo(start,start+gap*i);ctx.lineTo(550,start+gap*i);ctx.stroke();ctx.beginPath();ctx.moveTo(start+gap*i,start);ctx.lineTo(start+gap*i,550);ctx.stroke()}
for(let i=0;i<ply;i++){const cell=g.moves[i],x=start+(cell%size)*gap,y=start+Math.floor(cell/size)*gap;ctx.beginPath();ctx.arc(x,y,gap*.4,0,Math.PI*2);ctx.fillStyle=i%2===0?'#202421':'#fafbf9';ctx.fill();ctx.strokeStyle=i===ply-1?'#bb3930':'#777';ctx.lineWidth=i===ply-1?3:1;ctx.stroke();ctx.fillStyle=i%2===0?'white':'#222';ctx.font='14px system-ui';ctx.textAlign='center';ctx.textBaseline='middle';ctx.fillText(String(i+1),x,y)}
document.getElementById('players').textContent=`흑: ${names[black]} / 백: ${names[white]}`;
document.getElementById('status').textContent=`${ply} / ${g.moves.length}수${ply<=2?' · 공통 시작판':''}`;
document.getElementById('result').textContent=ply===g.moves.length?(g.winner===0?'무승부':`${g.winner===1?'흑':'백'} 승리`):`${ply%2===0?'흑':'백'} 차례`;}
sel.onchange=()=>{stop();gi=Number(sel.value);ply=0;draw()};seek.oninput=()=>{stop();ply=Number(seek.value);draw()};
document.getElementById('first').onclick=()=>{stop();ply=0;draw()};document.getElementById('prev').onclick=()=>{stop();ply=Math.max(0,ply-1);draw()};
document.getElementById('next').onclick=()=>{stop();ply=Math.min(games[gi].moves.length,ply+1);draw()};document.getElementById('last').onclick=()=>{stop();ply=games[gi].moves.length;draw()};
document.getElementById('play').onclick=()=>{if(timer){stop();return}if(ply===games[gi].moves.length)ply=0;document.getElementById('play').textContent='정지';timer=setInterval(()=>{ply++;draw();if(ply===games[gi].moves.length)stop()},450)};draw();
</script></html>'''.replace("__DATA__", data)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    evaluation = json.loads((args.input / "evaluation.json").read_text(encoding="utf-8"))
    training = json.loads((args.input / "training_summary.json").read_text(encoding="utf-8"))["runs"]
    replays = json.loads((args.input / "replays.json").read_text(encoding="utf-8"))["games"]
    validate_replays(replays)
    seeds = sorted({r["config"]["seed"] for r in training})
    config = training[0]["config"]
    decisions = config["updates"] * config["envs"] * config["rollout"]
    if any(r["completed_updates"] != config["updates"] or r["learner_decisions"] != decisions for r in training):
        raise RuntimeError("All runs must have the same completed training budget")
    lines = ["# 9×9 일반 오목 첫 실험 결과", "", "2026-10-06. 전체 판 관측, 금수 없음, 양쪽 모두 5개 이상 연속 승리.", "",
             "[고정 프로토콜](../../protocol.md) · [실제 대결 기보 재생](replays.html) · [평가 원자료](evaluation.json)", "",
             f"4모델 × {len(seeds)}개 새 학습 시드. 동일 CNN·정책/가치 출력 구성과 파라미터 수를 맞췄다.",
             f"모델마다 무작위 상대에게 PPO {config['updates']}업데이트, {config['envs']}환경×{config['rollout']}결정, 총 {decisions:,} 학습자 결정을 사용했다.",
             "승리 +1·패배 -1·무승부 0만 사용했다. 기존 기억 과제 가중치, 교사 착수, 중간 보상 및 탐색은 사용하지 않았다.",
             "시험은 고정 예산의 마지막 체크포인트이며 시험 성적에 따른 체크포인트 선택은 없다.", "",
             "## 공통 상대 평가", "", "각 시드의 새 상대 시드에서 무작위·규칙 상대 각각 256경기, 흑백 각 128경기. 평균 ± 학습 시드 표본표준편차다.", "",
             "| 모델 | 무작위 상대 승률: 새 초기화 → 학습 후 | 규칙 상대 승률: 새 초기화 → 학습 후 |",
             "|---|---:|---:|"]
    for kind in KINDS:
        groups = []
        for opp in ("random", "tactical"):
            before = [r["win_rate"] for r in evaluation["fresh_initialization"] if r["kind"] == kind and r["opponent"] == opp]
            after = [r["win_rate"] for r in evaluation["fixed_opponents"] if r["kind"] == kind and r["opponent"] == opp]
            groups.append(mean_sd(before) + " → " + mean_sd(after))
        lines.append(f"| {NAMES[kind]} | {groups[0]} | {groups[1]} |")
    lines += ["", "규칙 상대는 자기의 즉시 승리를 우선하고 상대의 즉시 승리를 막은 뒤 줄을 늘리는 얕은 정책이다.",
              "이 상대에게는 학습하지 않았다. 무작위 상대 성적만으로 방어·전략 학습 성공을 해석하지 않는다.",
              "무작위 플레이어 기준: " + "; ".join(f"{opp} 상대 승률 {row['win_rate'] * 100:.2f}%" for opp, row in evaluation["random_baseline"].items()),
              "", "## 모델끼리 맞대결", "",
              f"같은 학습 시드끼리 6쌍 × {len(seeds)}시드. 각 시드의 공통 무작위 시작 2수 "
              f"{evaluation['paired_two_ply_openings_per_seed']}개에서 흑백을 바꿔 대결했다.",
              "표는 행 모델의 점수율 (승 + 무승부×0.5) / 경기수, 평균 ± 시드 표준편차다.", "",
              "| 행 모델 / 열 상대 | " + " | ".join(NAMES[k] for k in KINDS) + " |",
              "|---|" + "---:|" * len(KINDS)]
    for kind in KINDS:
        cells = []
        for other in KINDS:
            if kind == other:
                cells.append("—")
                continue
            scores = [r["score"] if r["a"] == kind else 1 - r["score"] for r in evaluation["head_to_head"]
                      if {r["a"], r["b"]} == {kind, other}]
            cells.append(mean_sd(scores))
        lines.append("| " + NAMES[kind] + " | " + " | ".join(cells) + " |")
    lines += ["", "승·무·패 집계 (A 모델 기준):", "", "| A | B | 승 | 무 | 패 | 시드별 A 점수율 |", "|---|---|---:|---:|---:|---|"]
    for a, b in ((a, b) for i, a in enumerate(KINDS) for b in KINDS[i + 1:]):
        rows = [r for r in evaluation["head_to_head"] if r["a"] == a and r["b"] == b]
        lines.append(f"| {NAMES[a]} | {NAMES[b]} | {sum(r['wins'] for r in rows)} | {sum(r['draws'] for r in rows)} | {sum(r['losses'] for r in rows)} | "
                     + "; ".join(f"{r['seed']}: {r['score'] * 100:.2f}%" for r in rows) + " |")
    lines += ["", "## 자원과 해석 범위", "", "| 모델 | 은닉 크기 | 학습 파라미터 | 학습 시간 평균/실행 | 최대 PyTorch 할당/예약 |", "|---|---:|---:|---:|---:|"]
    for kind in KINDS:
        runs = [r for r in training if r["config"]["kind"] == kind]
        lines.append(f"| {NAMES[kind]} | {runs[0]['config']['hidden']} | {runs[0]['parameters']:,} | "
                     f"{statistics.mean(r['elapsed_seconds'] for r in runs):.1f}초 | "
                     f"{max(r['peak_allocated_gib'] for r in runs):.3f}/{max(r['peak_reserved_gib'] for r in runs):.3f} GiB |")
    lines += ["", "VRAM 수치는 PyTorch가 추적한 할당/예약 피크이며 CUDA 컨텍스트·디스플레이·다른 프로세스 사용량을 포함하지 않는다.",
              "모델을 하나씩 학습했다. 이 측정은 실제 9×9 작은 파일럿의 결과이며 15×15와 큰 배치의 메모리 보장은 아니다.",
              "전체 판을 보므로 이 실험만으로 기억 능력을 비교할 수 없다. 공통 상대는 무작위이고 자기 대국 학습은 아직 수행하지 않았다.",
              "동일 시드끼리의 대결과 재배선 3개 결과이며 여러 시드·재배선의 교차 리그가 아니다.",
              "실제 연결 구조의 고유한 우위와 강한 오목 전략은 후속 실험에서 확인해야 한다.",
              "", "다음 단계는 같은 예산의 더 강한 공통 상대 및 과거 정책을 상대로 하는 학습과 새 시작판 평가다.",
              "9×9 전략과 대결이 안정화된 뒤 부분관측 및 15×15 렌주를 각각 추가한다.",
              "", f"실제 대결 기보 {len(replays)}개를 돌 순서와 승패 판정으로 다시 검증했다. 환경·학습 검증 테스트 14개와 이전 코드 테스트 43개 통과."]
    (args.input / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (args.input / "replays.html").write_text(viewer(replays), encoding="utf-8")
    print(args.input / "results.md")


if __name__ == "__main__":
    main()

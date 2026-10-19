# 15x15 Renju 학습 기록 (2026-10-15)

모든 명령은 저장소 최상위(MaleCNS)에서 `$env:PYTHONPATH="."` 로 실행한다.

## 규칙·코드 (gomoku/renju/)
- `rules.py` 정확한 렌주 규칙: 흑 장목·3-3·4-4 금수, 정확히 5개는 금수여도 승리, 백은 5개 이상 승리.
  흑 첫 수는 중앙(7,7) 고정. 금수 판정은 선 문자열 캐시 + 재귀(3-3 에서 열린 4를 만드는 수가 금수면 3이 아님).
- `env.py` 배치 환경(흑 금수는 합법수 마스크에서 제외, 금수를 두면 즉시 패), 규칙 기반 `tactical` 상대.
- `tactics.py` 정확한 교사 6범주(승리/막기/포크 만들기/포크 막기/강제 연쇄 만들기/연쇄 막기). 흑이 막을 칸이 금수면 라벨 없음.
- `data.py` 롤아웃 자료(관측, 라벨 집합, 합법 마스크, 범주, D4 정규 키, 수순). 훈련·검증 키 중복 제거.
- `train.py` 반복 신경망 학습: 과거 자기 차례 관측으로 상태 복원(기울기 없음, 1536개씩 나눠 통과), 마지막 한 수만 학습.
- `live.py` 규칙 상대와 고정 시드 대국 + 전술 정확도, `league.py` 모델끼리 대국, `onpolicy.py` 모델이 실제로 도달한 위치의 정확한 라벨.
- `play_server.py` / `play.html` 사람 대국(금수 ✕ 표시). 실행: `.venv\Scripts\python.exe -m gomoku.renju.play_server --port 8766`.
- 테스트: `python -m unittest gomoku.renju.test_rules gomoku.renju.test_tactics gomoku.renju.test_env gomoku.renju.test_play`.

## 모델
- Fly: `previous_research/data/subgraphs/malecns_cx_2950.npz` (CX 내부 뉴런 전부, 2950개, 연결 207,827개; `build_subgraphs.py --sizes 2950`).
- RNN: 은닉 3000.
- 공통: 64채널 x 6 잔차 블록(GroupNorm) 넓은 트렁크, 4채널 관측(내 돌/상대 돌/마지막 수/흑 여부). 금수 칸은 합법 마스크로만 전달.
- 12x12 체크포인트 로딩은 그대로(`graph_neurons` 기본 1000). 원본 체크포인트는 건드리지 않는다.

## 실행 규칙
- 학습은 한 번에 1개만 돌린다 (`logs/run_renju_train.ps1` 이 Fly, RNN 순서로 순차 실행).
- 자료 수집은 워커마다 BLAS 스레드를 1개로 제한해야 한다(`renju/__init__.py`). 제한이 없으면 10배 느려졌다.

## 진행
- [x] 규칙·교사·환경·자료·학습·평가·대국 서버 구현 및 테스트
- [x] 자료 수집: data/train.npz, test.npz = 범주 1~6 각 6000/600 (정확한 라벨, 롤아웃 144만 위치 중 선별)
- [x] 조용한 수: data/quiet_train.npz(6000), quiet_test.npz(400) — Rapfi 렌주 규칙 200ms 라벨, 금수 불일치 0건
- [x] 1차 학습 (한 번에 1개): `--extra quiet_train`, 12000 업데이트, 로그 logs/renju_fly_v1.log → logs/renju_rnn_v1.log
      fly2950_v1 (약 38분) → rnn3000_v1 → models/ (VRAM 최대 약 1.1GB 할당/3GB 점유)
- [x] 규칙 상대 평가, 두 모델 대결 (`live_v1.json`, `league_v1.json`에서 완료 확인)
- [ ] 사용자 대국 후: 약점 자료 보강(onpolicy), Rapfi 렌주 라벨(조용한 수) 추가

2026-10-15 결과 정리: [최신 종합 보고서](../../../docs/RESEARCH_STATUS_20261015.md). 사람 기보 분석과79개 교정 판은 저장됐지만, 해당 자료를 적용한 추가 학습 결과는 아직 확인되지 않았다.

## Round 2/3 (2026-10-16)
- v2: fine-tune v1 with human_train(79) + selfplay_train + onpolicy_train (lr 3e-4, 6000 updates).
- v3: Rapfi DAgger (`renju/dagger.py`): policy-vs-Rapfi games (opp 60/100 ms; <=30 ms Rapfi returns junk moves), beta in {0,.3,.6}, rule-AI blunders {0,.25,.5};
  positions labelled by exact teacher else Rapfi (cat 7). Data: rapfi_train (30.8k), rapfi_test (3.8k held-out), humanr_train (208, human games, not held-out).
  Fine-tune v2 -> v3, lr 4e-4, 8000 updates, extras quiet/selfplay/onpolicy/human/humanr/rapfi.
- Results: rapfi_test Rapfi agreement 30% -> 34.5%; vs weakened Rapfi (blunder .5) wins Fly 41->53, RNN 29->59 of 180;
  vs rule AI Fly 164->193 W, RNN 169->219 W (white 128/0); league v3 > v2 (Fly .60, RNN .67 head-to-head).
- Human test of v3 pending. Do not leave play_server running.

## Round 4 (2026-10-16): forbidden-block chains
- `renju/forbid.py`: white-to-move positions where a four's only block cell is black-forbidden (category 5, exact labels): forbid_train 1000 / forbid_test 150 (moderate on purpose).
- v4 = v3 + forbid_train + refreshed human_train (297) / humanr_train (319), 5000 updates from v3.
  forbid_test accuracy: v3 ~49% -> v4 ~69-70%; other suites unchanged.
- fly2950_v4 (lr 3e-4): fine. rnn3000_v4 (lr 3e-4, seed 5302) REGRESSED vs rule AI as white (127/0 -> ~55/65 over 3 seeds), hidden from play server.
  rnn3000_v4b (lr 2e-4, seed 5303): no regression (white 124/0), use it as the RNN v4.
- Hidden from play server: *_v1, rnn3000_v4. Play: fly2950_v4, rnn3000_v4b (also v2/v3).

## Round 5 (2026-10-17): seed study from v4 (new human games + Rapfi DAgger on v4)
- Data: human_train 482 / humanr_train 510 (all saved games), rapfi2_train 30.4k / rapfi2_test 4.0k (v4 models vs Rapfi).
- Trained 2 seeds per kind from v4, 5000 updates (Fly lr 3e-4: v5a seed 5401, v5b 5402; RNN lr 2e-4: v5a 5403, v5b 5404).
- Static suites: all within ~1pp of v4 (seed gap <= 0.7pp). Rule AI (256 games, 2 seeds): fly v5a 253/256 (!), fly v5b ~205 (= v4);
  rnn v5a 171-176 (REGRESSED as white 78/42), rnn v5b 230-236 (v4b 221-223). Seed variance is large.
- League (6 models): all 0.48-0.52. Weak Rapfi (blunder .5, 180 games): fly v4 60 / v5a 72 / v5b 68; rnn v4b 62 / v5a 55 / v5b 48.
- v5 = fly2950_v5 (copy of v5a) and rnn3000_v5 (copy of v5b). Play server shows only v4 (fly2950_v4, rnn3000_v4b) and v5.
  Hidden: *_v1, *_v2, *_v3, rnn3000_v4, *_v5a, *_v5b (files kept).

## Round 5/6 (2026-10-17)
- Teacher fix: create_forcing_chain (forced win) now outranks prevent_fork; stored data relabeled (`renju/relabel.py`, originals in data/before_priority_fix/, ~370 positions).
- v5 = fly2950_v5a / rnn3000_v5b (v4 + rapfi2 DAgger + human games). Rule-AI score (253/256 for Fly) did not transfer to humans.
- v6 = fly2950_v6a / rnn3000_v6a (v5 + relabeled data + 47 game files' positions), picked by weak-Rapfi (74/70 wins of 180) and league (~0.51); rule AI only a reference.
- Server shows v5 and v6 only. Not yet in training: the 4 games added at 14:52-14:57 (incl. AI black walking into a forbidden-block chain), plan v7: forbidden-block defence for black.

## Round 7 (2026-10-17 evening): Fly v7 (RNN not trained, keep v6a)
- Code: `tactics.deep_chains/deep_defence/analyze_deep` (VCF up to 4 fours), `renju/deep.py` (modes deep, forbid_black), `dagger.py --hunt/--relabel-ms` (opponent plays any forced win it finds; lost games' last engine labels relabelled at 600 ms), `forbid.py --prefix`.
- Data (all checked: 0 overlap with any *_test): deep 3000/300, forbid_black 1500/200, forbid2 1000/50, rapfi3 9486/1158 (v6a vs Rapfi), human 924, humanr 951 (all 60 game files). Old human files in data/before_v7/.
- Train: fly v6a -> v7a (seed 7401), v7b (7402), lr 3e-4, 5000 updates, extras = quiet selfplay onpolicy human humanr rapfi rapfi2 forbid forbid2 deep forbid_black rapfi3. Logs logs/renju_fly2950_v7*.log, scripts logs/run_v7_*.ps1.
- Static: deep_test 65.0 -> 77.3/79.7, forbid_black_test 76.0 -> 81.5/83.5, forbid_test 72.7 -> 81.3/79.3, others unchanged.
- Play: weak Rapfi 720 games: v5 268, v6a 257, v7a 259, v7b 265 (no difference); league v6a .513, v7a .509, v7b .500.
- v7 = fly2950_v7 (copy of v7b). play_server hides *_v7a/_v7b. Report: docs/RESEARCH_STATUS_20261017.md. Next: user plays v7; no play server running.
- v7 human test (5 games): human 1 / AI 4; AI (white) won 2 games with a four whose only block cell was black-forbidden (plies 22, 24). Server now hides *_v5 too: shows fly2950_v6, rnn3000_v6, fly2950_v7.

## Round 8 (2026-10-18): Fly v8 (RNN not trained)
- v7 human test, 16 games: human 4 / AI 12 (14 distinct). Human wins come from open-three attacks the teacher cannot label (no exact category in AI's last 14 plies).
- Data: human 1089 / humanr 865 (tails of human-won games excluded) / humanlost 259 (tails, Rapfi 1500 ms) / rapfi4 9850 (v7 vs Rapfi+hunter, relabel last 10 lost positions at 1000 ms). Old human files in data/before_v8/.
- Train: v7 -> v8a (8401), v8b (8402), lr 3e-4, 5000 updates. Scripts logs/run_v8_*.ps1.
- Weak Rapfi 720: v6a 272, v7 294, v8a 248, v8b 268; league v6a .510 v7 .502 v8a .508 v8b .498 (league_v8.json). No clear gain. fly2950_v8 = copy of v8b for human testing; server shows fly2950_v6, rnn3000_v6, fly2950_v7, fly2950_v8. No server running.
- v8 human test (9 games): human 3 / AI 6 (v7 16 games: human 4 / AI 12). Not distinguishable. AI(black) 3 of 5 wins, AI(white) 3 of 4; one forbidden-block four (ply 18).

## Round 9 (2026-10-18): controlled comparison, from scratch
- User decision: stop recipe work; compare rewired Fly / real Fly / RNN (GRU and 12x12 league context: rewired Fly 58.9, RNN 58.7, GRU 49.9, Fly 32.6).
- Same recipe for all: data = train + 14 extras (all current npz), 12000 updates, lr 1e-3, batch 256, seeds 9101 (a) / 9102 (b); rewired graphs malecns_cx_2950_rewired_seed0/1 (a/b, degree preserving, made with previous_research/code/rewire_subgraphs.py).
- Order: ctl_rewired2950_a,b -> ctl_fly2950_a,b -> ctl_rnn3000_a,b (logs/run_ctl_train.ps1, one GPU job at a time, ~70 min each). Eval auto-starts after: logs/run_ctl_eval.ps1 (static suites, weak Rapfi 720 incl. rnn3000_v6a, league_ctl.json with fly2950_v7 + rnn3000_v6a).
- Caveat: extras contain positions from earlier real-Fly/RNN models and human games vs them. ctl_* hidden from play server until chosen.
- 18:45 change (user request): two training lanes in parallel on the GPU. Lane 1 (run_ctl_lane1.ps1): rewired a -> rewired b -> RNN a. Lane 2 (run_ctl_lane2.ps1): real Fly a -> Fly b -> RNN b. Placeholder files (text, tiny) exist for not-yet-trained ctl_*.pt until their runs finish; eval (run_ctl_eval.ps1) waits for logs/renju_ctl_all.done.
- Round 9 results (eval finished 00:00): weak Rapfi 720 wins: rewired a/b 260/245, Fly a/b 288/268, RNN a/b 234/239, rnn v6a 262 (fly v7 294 earlier, same seeds). League (8 models, league_ctl.json): rnn v6a .525, rewired b .514, Fly b .514, RNN a .503, Fly v7 .497, Fly a .484, rewired a .482, RNN b .480. Static: RNN a best on forbid/deep (.86/.863), rewired a best forbid_black (.93). No clear winner; seed gaps as large as kind gaps. ctl_* now visible in play server; no server running.

## Round 10 (2026-10-19): human test of ctl_*, integrated report, upload sync
- Human games vs ctl_* (12 files in 기보저장, 11 finished + 1 abandoned 2-move game): AI 9 / human 2 (ctl_fly2950_a as white, ctl_rnn3000_b as black). By kind: Fly 1/3, rewired 0/4, RNN 1/4 human wins. Summary: human_games_summary_ctl.json. Not a ranking.
- First white reply to the centre is flat for all six ctl_* (top-1 prob 3-22%; ctl_fly2950_b 3% -> picks (11,11)); fly v6-v8 34-47%, rnn v6 24%. Cause not investigated.
- Aggregated controlled-comparison data: controlled_comparison_20261019.json (static suites, weak Rapfi per eval seed, validation curve points, league, by-kind means). Per kind: weak Rapfi Fly 38.6% / rewired 35.1% / RNN 32.8% (1440 games each), league .499/.498/.492, deep .827/.790/.838. Params: Fly and rewired 1,257,627, RNN 10,062,450.
- Report: docs/RESEARCH_REPORT_20261019.md (RQ1-4 structure). Scripts copied (portable) to runs/renju15_20261015/scripts/.
- Next candidates (need user go-ahead, all need training): low-data comparison (10%/25%), 5 seeds per kind, data without earlier Fly/RNN-origin positions, open-three (VCT) teacher for v9. No server or job is running.


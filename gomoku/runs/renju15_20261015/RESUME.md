# 15x15 Renju 학습 기록 (2026-10-15)

모든 명령은 `C:\Users\a\Desktop\MaleCNS`에서 `$env:PYTHONPATH="."` 로 실행한다.

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

# 일반 오목 순환망 비교

[내 모델과 직접 대국하기](PLAY.md): GRU·Fly·RNN·재배열 Fly 원본을 선택해 로컬 화면에서 흑백 대국할 수 있다.

9×9 파일럿과 12×12 후속 실험을 포함합니다. 흑백 모두 5개 이상 이어지면 승리하고 렌주 금수는 없습니다.
전체 판을 관측하며 합법 칸만 출력에서 허용합니다. Fly·재배선 Fly·RNN·GRU는 과거 과제의 가중치를
이식하지 않고 새로 학습했습니다. 실제 Fly는 실측 연결 부분 그래프 위의 학습 가능한 순환망입니다.

## 최신 연구 상태 (2026-10-15)

[통합 연구 보고서(2026-10-19)](../docs/RESEARCH_REPORT_20261019.md)를 먼저 참고하세요. 이어서 [10월17~18일 v5~v8 정리](../docs/RESEARCH_STATUS_20261017.md)와 [10월15일 종합 보고서](../docs/RESEARCH_STATUS_20261015.md)를 보면 됩니다. 12×12 확대 학습과15×15 렌주를 구분해 정리했습니다.

- 12×12 넓은 트렁크·Rapfi 초중반 자료: 네 후보 모두 규칙 상대 승리가 증가했습니다. 일부 모델은 패배도 증가했으며 구조별 우열은 미확정입니다.
- 15×15 렌주: `renju/`에서 Fly2950·RNN3000을 각12000회 학습했습니다. 규칙 상대256국에서 각각171·142승, 직접 대결은130:126입니다.
- [15×15 실행 기록](runs/renju15_20261015/RESUME.md), [12×12 확대 실험](runs/defense_probe_20261014/results.md).

### 10월14일까지의 기록

[최신 요약](../docs/RESEARCH_STATUS_20261013.md)과 [10월14일 첫 수 점검·다음 진행 방향](../docs/OPENING_AND_NEXT_STEPS_20261014.md)을 참고하세요. 아래 초기 학습 명령은 10월7일 실험을 재현하는 예시입니다.

- [10월14일 중앙 시작·네 모델 교정 학습](runs/opening_followup_diverse_20261014/results.md): PPO 첫 수 확률 계산 수정, 네 모델 동일 예산 학습·재평가 완료. 새 후보는 모두 미채택하며 원본 가중치를 유지한다.
- [10월14일 방어 보존 커리큘럼·Rapfi 연결](runs/balanced_curriculum_20261014/results.md): 방어·흑백 균형과 보존 손실을 넣은 2단계 실험 완료, 원본 유지. Rapfi 12×12 탐색·15×15 NNUE 로딩 확인, 초중반 교사 자료128개 확보. 엔진 자료 학습은 아직 하지 않았다.

- [10월14일 방어 정적 진단·트렁크 확대](runs/defense_probe_20261014/results.md): 공통 공간 트렁크(16채널·3층)가 방어의 병목임을 같은 자료·예산 비교로 확인. 넓은 트렁크 신규 학습 후보는 새 시드 평가에서 패배가 크게 줄었으나 원본은 아직 교체하지 않았다.
- [F3 장기 준비·방어 보존](runs/long_plan_f_20261009/results.md): GRU 개선, 낮은 절대 승률과 시드 변동성 존재.
- [네 모델 직접 대결](runs/model_league_20261010/results.md): 강한 전술 상대 평가와 분리해서 해석.
- [정적 공간 처리](runs/spatial_state_20261012/report.md): 이번 방식 미채택.
- [후보3/6/12 비교](runs/candidate_width_nested_20261013/comparison.json): 완료. 고유 출발 판30개 확보는 진행 중이며 새 자료 재학습은 아직 없음.
- 새 탐색 코드: robust_long_plan.py, robust_collect.py, wide_search_worker.py, coverage_teacher.py.
- 출발 판 균형 학습: history_aux.py의 root_keys 기반 샘플링.

## 자료 위치

| 경로 | 내용 |
|---|---|
| [reports/](reports/) | 읽기용 결과 보고서·평가 수치·기보 |
| runs/pilot_20261006/ | 9×9 파일럿 원 실행 기록 |
| runs/defense12_20261006/ | 첫 12×12 승리·방어 보조 학습 |
| runs/advanced_20261007/ | 기보·공간 인식·전술 확대와 4모델×3시드 |
| [runs/multistep_20261007/results.md](runs/multistep_20261007/results.md) | 실패 기보·강제 공격 연결 동일 예산 파일럿 |
| [multistep_protocol.md](multistep_protocol.md) | 해당 파일럿 사전 예산·평가·확대 기준 |
| [runs/midgame_20261007/results.md](runs/midgame_20261007/results.md) | 중간 판 커리큘럼 → 공격 다양화·범주 균형 비교 |
| [midgame_protocol.md](midgame_protocol.md) | 두 단계의 사전 실행 순서·예산·평가 |
| 각 실행의 source/ | 당시 코드 사본 |
| logs/ | 로컬 실행 로그 |
| [advanced_protocol.md](advanced_protocol.md) | 최신 실험의 사전 계획 |
| [resource_benchmark.md](resource_benchmark.md) | 9·12·15 크기별 합성 메모리 측정 |

코드와 읽기용 보고서를 구분했고 runs는 재현에 필요한 원 실행 자료입니다.
GitHub용 사본에는 .pt 가중치와 로그가 제외됩니다. 전체 자료는 원 작업 폴더에 보존합니다.
실험 당시 JSON과 source의 gomoku9 경로는 현재 gomoku에 해당하며 수치와 가중치는 그대로입니다.

## 코드 찾기

- 환경·규칙·상대: env.py
- 기존 순환 정책·그래프·가중치 로딩: models.py
- 공통 공간 인코더·공유 칸별 출력: spatial.py
- PPO·보조 학습·체크포인트: train.py
- 즉시 승리·필수 방어 교사: tactics.py
- 열린3·복수 승리 위협 교사: expanded_tactics.py
- 기보 입력과 상태 복원: history_aux.py
- 경기·맞대결: evaluate.py, advanced_evaluate.py
- 데이터·예산 감사와 학습 판 진단: advanced_audit.py
- 강한 상대의 실제 복수 위협 선택·패배 상황·기보: advanced_match_audit.py
- 최신 보고서 생성: advanced_report.py
- 실패 기보 추출·강제 공격 연결: mine_multistep.py, multistep_tactics.py, chain_curriculum.py
- 실패 기보 비교 평가·보고서: multistep_evaluate.py, multistep_report.py
- 중간 판 커리큘럼·자료: midgame_env.py, midgame_data.py
- 중간 판/빈 판 평가·감사·보고서: midgame_evaluate.py, midgame_audit.py, midgame_report.py
- 반복 방문 수집·위협 예측: dagger.py, threat_aux.py
- 반복 실험 감사·청정 부분집합·보고서: iterative_audit.py, iterative_clean_subset.py, iterative_report.py
- 반복 실험 최종 보고서: runs/iterative_20261008/results.md
- 테스트: test_dagger_threat.py, test_midgame.py, test_multistep.py, test_gomoku.py, test_defense.py, test_history_spatial.py

## 실행

작업 폴더는 저장소 최상위입니다. 아래 명령은 정리 후 gomoku 패키지 경로입니다.
새 환경에는 ../requirements.txt와 당시 environment_versions.txt를 참고하세요.
Fly 그래프는 ../previous_research/data/subgraphs/에 포함됩니다.

~~~powershell
$env:OPENBLAS_NUM_THREADS = '1'
.venv/Scripts/python.exe -m unittest gomoku.test_gomoku gomoku.test_defense gomoku.test_history_spatial -v
.venv/Scripts/python.exe -m gomoku.train --size 12 --models fly rewired rnn gru --seeds 4601 4602 4603 --opponent weak --warmup-updates 1800 --updates 1200 --auxiliary-mode history --architecture spatial --curriculum --tactical-data gomoku/runs/advanced_20261007/data/train.npz --output gomoku/runs/advanced_20261007/final
.venv/Scripts/python.exe -m gomoku.advanced_evaluate --input gomoku/runs/advanced_20261007/final --data gomoku/runs/advanced_20261007/data/test.npz --models fly rewired rnn gru --seeds 4601 4602 4603 --games 256 --league
.venv/Scripts/python.exe -m gomoku.advanced_report
~~~

동일 설정의 체크포인트가 있으면 이어서 학습하며 완료된 실행은 재사용합니다.
새 실험은 새 시드와 출력 폴더를 지정해야 합니다. 최종 시험 결과로 설정과 예산을 선택하지 않습니다.
학습 시드는4601·4602·4603이고 대표 단계 검증은4500의 단일 시드입니다.
기보 학습은 앞선 자기 차례 관측을 역전파 없이 처리하고 정답 턴에서 학습합니다.
공간 변경은 인코더와 출력 방식의 결합 효과이며 기억 능력이나 Fly 구조의 우월성을 직접 증명하지 않습니다.


# 일반 오목 순환망 비교

9×9 파일럿과 12×12 후속 실험을 포함합니다. 흑백 모두 5개 이상 이어지면 승리하고 렌주 금수는 없습니다.
전체 판을 관측하며 합법 칸만 출력에서 허용합니다. Fly·재배선 Fly·RNN·GRU는 과거 과제의 가중치를
이식하지 않고 새로 학습했습니다. 실제 Fly는 실측 연결 부분 그래프 위의 학습 가능한 순환망입니다.

## 자료 위치

| 경로 | 내용 |
|---|---|
| [reports/](reports/) | 읽기용 결과 보고서·평가 수치·기보 |
| runs/pilot_20261006/ | 9×9 파일럿 원 실행 기록 |
| runs/defense12_20261006/ | 첫 12×12 승리·방어 보조 학습 |
| runs/advanced_20261007/ | 기보·공간 인식·전술 확대와 4모델×3시드 |
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
- 테스트: test_gomoku.py, test_defense.py, test_history_spatial.py

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

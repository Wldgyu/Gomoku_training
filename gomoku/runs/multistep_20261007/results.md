# 12×12 실패 기보·강제 공격 연결 학습 결과 — 2026-10-07

Fly·GRU 단일 학습 시드4700의 동일 예산 파일럿입니다. 모델의 실제 착수는 점유 칸만 차단한 신경망 argmax입니다.

실패 기보 학습은 새 시험에서 방어 누락률을37.28%→26.80%로 줄였지만 두 방식 모두0/512승, 실제 안전한 복수 위협 생성0회였습니다. 실전 승률 개선이 확인되지 않아 네 모델×3시드 확대는 실행하지 않았습니다.

| 평가 | 모델 | 방식 | 승리/256 | 방어 누락/기회 | 복수 위협 도달 경기 | 실제 안전한 복수 위협 생성 |
|---|---|---|---:|---:|---:|---:|
| validation | fly | baseline | 0 | 125/328 | 131/256 | 0 |
| test | fly | baseline | 0 | 127/316 | 129/256 | 0 |
| validation | gru | baseline | 0 | 101/307 | 155/256 | 0 |
| test | gru | baseline | 0 | 103/301 | 153/256 | 0 |
| validation | fly | multistep | 0 | 87/366 | 169/256 | 0 |
| test | fly | multistep | 0 | 96/355 | 160/256 | 0 |
| validation | gru | multistep | 0 | 90/322 | 166/256 | 0 |
| test | gru | multistep | 0 | 94/354 | 162/256 | 0 |

| 평가 | 방식 | 합산 승률 | 방어 누락률 | 복수 위협 도달 비율 | 공격 복수 위협 선택 성공/기회 | 실제 생성/512경기 |
|---|---|---:|---:|---:|---:|---:|
| validation | baseline | 0/512 (0.00%) | 35.59% | 55.86% | 0/0 | 0 |
| validation | multistep | 0/512 (0.00%) | 25.73% | 65.43% | 0/0 | 0 |
| test | baseline | 0/512 (0.00%) | 37.28% | 55.08% | 0/2 | 0 |
| test | multistep | 0/512 (0.00%) | 26.80% | 62.89% | 0/0 | 0 |

확대 기준 충족: **아니오**.
검증·시험 모두 합산 승률 +5%p 이상, 개별 모델 승수 감소 없음, 방어 누락률·복수 위협 도달 비율 감소, 실제 경기당 안전한 복수 위협 생성 수 감소 없음을 사전에 요구했습니다.
기준을 충족하지 않으면 네 모델×3시드로 확대하지 않습니다.

## 데이터와 예산

기존 3,072판의 1,388건 유일 방어 누락 패배와 1,675건 복수 위협 패배를 학습 원자료로 전환했습니다. 중복·평가 판 제외 후 2,323문제를 사용했고 현재 교사가 비어 있지 않은 정답을 만들지 못한 818판은 별도 보존했습니다. 이는 모든 해당 판이 일반적으로 방어 불가능하다는 증명이 아닙니다.
실제 패배에서 얻은 범주별 문제 수: {'1': 1, '2': 1359, '3': 0, '4': 904, '5': 1, '6': 58}. 1=즉시 승리, 2=방어, 3=복수 위협 생성, 4=예방, 5=공격 연결, 6=연결 예방.
공격 연결 사례 부족으로 512개 생성·512개 예방 문제를 독립 합법 기보로 보충했습니다. 교차 공격 한 모티프의 회전·반사·이동·무작위 주변 돌 변형이므로 다양한 공격 연결의 일반화를 증명하지 않습니다.
각 방식 학습 20,000예제. 새 평가 기보 1,024판의 모든 차례 12,891개 판+차례를 학습에서 제외했습니다.
검증/시험 경기 은행은 학습 전에 기존 고정 모델로 생성했습니다. 실제 비교는 새 모델을 같은 환경 시드로 평가하며, 종료·재설정에 따른 난수 소비 차이 때문에 완전히 동일한 상대 응수 기보의 쌍별 비교는 아닙니다.
기존 3,072판과 원 전술 시험은 학습 원자료가 되었으므로 이후 독립 시험 성적처럼 해석하지 않습니다.

| 모델 | 방식 | 보조 예제 제시 수 | PPO 결정 수 | 추가 기보 관측 | 시간(초) | 최대 할당 GiB |
|---|---|---:|---:|---:|---:|---:|
| fly | baseline | 153,600 | 153,600 | 1,627,627 | 63.1 | 0.293 |
| gru | baseline | 153,600 | 153,600 | 1,627,627 | 59.8 | 0.284 |
| fly | multistep | 153,600 | 153,600 | 1,656,121 | 64.4 | 0.301 |
| gru | multistep | 153,600 | 153,600 | 1,656,121 | 60.6 | 0.293 |

보조600×128, PPO300×32×16, 에포크2 및 에포크별 보조128입니다. 기존 대표 단계와 동일한600/300 예산이며 최종1800/1200 예산과는 다릅니다. 모든 모델은 새 초기화·동일 공간 구조·기보 burn-in·30→60→85% 규칙 상대 커리큘럼을 사용했습니다.

## 전술 정답률

| 평가 | 모델 | 방식 | 입력 | 범주 | 정답/문제 |
|---|---|---|---|---|---:|
| validation | fly | baseline | cold | block | 62/180 |
| validation | fly | baseline | cold | prevent_fork | 44/123 |
| validation | fly | baseline | cold | prevent_forcing_chain | 0/6 |
| validation | fly | baseline | history | block | 62/180 |
| validation | fly | baseline | history | prevent_fork | 44/123 |
| validation | fly | baseline | history | prevent_forcing_chain | 0/6 |
| test | fly | baseline | cold | block | 68/180 |
| test | fly | baseline | cold | prevent_fork | 33/116 |
| test | fly | baseline | cold | prevent_forcing_chain | 2/5 |
| test | fly | baseline | history | block | 68/180 |
| test | fly | baseline | history | prevent_fork | 33/116 |
| test | fly | baseline | history | prevent_forcing_chain | 2/5 |
| validation | gru | baseline | cold | block | 66/180 |
| validation | gru | baseline | cold | prevent_fork | 42/123 |
| validation | gru | baseline | cold | prevent_forcing_chain | 0/6 |
| validation | gru | baseline | history | block | 66/180 |
| validation | gru | baseline | history | prevent_fork | 42/123 |
| validation | gru | baseline | history | prevent_forcing_chain | 0/6 |
| test | gru | baseline | cold | block | 66/180 |
| test | gru | baseline | cold | prevent_fork | 30/116 |
| test | gru | baseline | cold | prevent_forcing_chain | 2/5 |
| test | gru | baseline | history | block | 66/180 |
| test | gru | baseline | history | prevent_fork | 30/116 |
| test | gru | baseline | history | prevent_forcing_chain | 2/5 |
| validation | fly | multistep | cold | block | 91/180 |
| validation | fly | multistep | cold | prevent_fork | 54/123 |
| validation | fly | multistep | cold | prevent_forcing_chain | 1/6 |
| validation | fly | multistep | history | block | 91/180 |
| validation | fly | multistep | history | prevent_fork | 54/123 |
| validation | fly | multistep | history | prevent_forcing_chain | 1/6 |
| test | fly | multistep | cold | block | 89/180 |
| test | fly | multistep | cold | prevent_fork | 50/116 |
| test | fly | multistep | cold | prevent_forcing_chain | 2/5 |
| test | fly | multistep | history | block | 89/180 |
| test | fly | multistep | history | prevent_fork | 50/116 |
| test | fly | multistep | history | prevent_forcing_chain | 2/5 |
| validation | gru | multistep | cold | block | 99/180 |
| validation | gru | multistep | cold | prevent_fork | 59/123 |
| validation | gru | multistep | cold | prevent_forcing_chain | 1/6 |
| validation | gru | multistep | history | block | 99/180 |
| validation | gru | multistep | history | prevent_fork | 59/123 |
| validation | gru | multistep | history | prevent_forcing_chain | 1/6 |
| test | gru | multistep | cold | block | 99/180 |
| test | gru | multistep | cold | prevent_fork | 52/116 |
| test | gru | multistep | cold | prevent_forcing_chain | 2/5 |
| test | gru | multistep | history | block | 99/180 |
| test | gru | multistep | history | prevent_fork | 52/116 |
| test | gru | multistep | history | prevent_forcing_chain | 2/5 |

## 해석과 재현

전술 교사의 연결 범위는 공격→유일 방어→복수 승리 위협→방어→승리, 즉 자기 착수3번/5ply의 강제 전술입니다. 중간 상대 즉시 승리를 배제합니다. 조용한 2수 계획이나 일반적 장기 전략을 완전히 해결하지 않습니다.
상대 다음 위협은 예방 착수 정답을 통해 간접 학습합니다. 별도 위협 예측 헤드나 예측 손실을 추가한 비교는 아닙니다.
새 정답률 세트는 검증309문제·시험301문제이고 모두 방어/예방 범주입니다. 공격 연결 생성의 독립 문제 정답률은 이번 세트에서 검증할 수 없으며 실전 공격 생성은0회였습니다.
복수 위협 도달 비율의 증가는 방어 누락으로 즉시 끝나는 패배가 줄어 이후 복수 위협 패배까지 진행한 구성 변화와도 양립합니다. 이 지표만으로 예방 능력의 악화를 확정할 수 없습니다.
cold/history의 범주별 정답 수는 이번 비교에서도 모두 같았습니다.
이번 비교는 데이터·강제 전술 보충의 결합 효과를 보는 단일 시드 파일럿입니다. 순환 기억이나 Fly 연결 구조의 우월성을 주장할 수 없습니다.
실전 위협 생성은 자기 착수 뒤 상대 즉시 승리가 없고 새로 두 개 이상의 자기 승리 칸이 생긴 경우를 집계합니다. 방어 누락은 기회 수가 달라질 수 있으므로 횟수와 비율을 함께 봅니다.
30개 단위·회귀 테스트 통과. 데이터 감사에서 키 중복 없음, 학습/평가 기보 전체 판 교집합0, 합법·비어 있지 않은 정답을 확인했습니다.

- [사전 프로토콜](../../multistep_protocol.md)
- [수치 원본](evaluation.json)
- [데이터·해시 기록](data/manifest.json)
- [분리 감사](data/split_audit.json)
- [실제 평가2,048경기 비중복 감사](actual_evaluation_split_audit.json)
- [기보 관측 복원 감사](data/history_audit.json)
- [동일 예산 감사](budget_audit.json)
- [추출 문제 원 기보 위치](data/mined.origins.json)
- [정답 미부여 판](data/unresolved.json)

~~~powershell
.venv/Scripts/python.exe -m unittest gomoku.test_multistep gomoku.test_gomoku gomoku.test_defense gomoku.test_history_spatial -q
.venv/Scripts/python.exe -m gomoku.multistep_evaluate --input gomoku/runs/multistep_20261007
.venv/Scripts/python.exe -m gomoku.multistep_report
~~~

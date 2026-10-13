# 12×12 중간 판 커리큘럼 → 다양한 공격·범주 균형 — 2026-10-07

새 시험 결과(baseline → step1 → step2): 자기 착수1번 내 완승 71.1% → 77.0% → 82.4%; 자기 착수2번 내 완승 39.1% → 54.7% → 67.2%; 자기 착수3번 내 완승 5.1% → 16.0% → 31.2%. 빈 판 강한 상대 승리는 0/512 → 0/512 → 0/512입니다.

중간 판에서의 개선 여부와 별개로 빈 판 전체 경기로의 승률 전이는 확인되지 않았습니다.

사용자 요청대로 1단계의 학습·검증을 완료한 뒤 2단계의 자료 생성·학습을 실행했습니다. Fly·GRU, 독립 새 학습 시드4810의 탐색적 비교입니다.

baseline은 빈 판에서만 학습, step1은 기존 자료+중간 판 커리큘럼, step2는 다양한 공격·실제 후속 응수 자료+범주 균형+새 시작 풀입니다.

## 빈 판에서 시작한 강한 상대 경기

| 평가 | 방식 | 모델 | 승/256 | 방어 누락/기회 | 복수 위협 도달/256 | 실제 안전한 복수 위협 생성 |
|---|---|---|---:|---:|---:|---:|
| validation | baseline | fly | 0 | 73/208 | 183 | 0 |
| validation | baseline | gru | 0 | 112/266 | 144 | 0 |
| validation | step1 | fly | 0 | 91/183 | 165 | 0 |
| validation | step1 | gru | 0 | 84/217 | 172 | 0 |
| validation | step2 | fly | 0 | 106/173 | 150 | 0 |
| validation | step2 | gru | 0 | 104/157 | 152 | 0 |
| test | baseline | fly | 0 | 92/214 | 164 | 0 |
| test | baseline | gru | 0 | 114/259 | 142 | 0 |
| test | step1 | fly | 0 | 86/207 | 170 | 0 |
| test | step1 | gru | 0 | 85/229 | 171 | 0 |
| test | step2 | fly | 0 | 99/162 | 157 | 0 |
| test | step2 | gru | 0 | 101/165 | 155 | 0 |

| 평가 | 방식 | 합산 승률 | 방어 누락률 | 복수 위협 도달 비율 | 실제 안전한 복수 위협 생성 |
|---|---|---:|---:|---:|---:|
| validation | baseline | 0/512 (0.00%) | 39.03% | 63.87% | 0 |
| validation | step1 | 0/512 (0.00%) | 43.75% | 65.82% | 0 |
| validation | step2 | 0/512 (0.00%) | 63.64% | 58.98% | 0 |
| test | baseline | 0/512 (0.00%) | 43.55% | 59.77% | 0 |
| test | step1 | 0/512 (0.00%) | 39.22% | 66.60% | 0 |
| test | step2 | 0/512 (0.00%) | 61.16% | 60.94% | 0 |

## 실제 기보 겹침을 제외한 공통 부분집합

회전·반사를 고려한 전체 기보 사후 감사에서 초기5ply 예방 판 한 종류가 step1/Fly 검증3기보·시험1기보에 자연 재등장했습니다. 중간 판 평가의 모든 기보에는 겹침이0입니다.
기존 고정 평가를 보존하고, 해당 시작 인덱스를 모든 방식·두 모델에서 공통 제외한 보조 결과를 아래에 표시합니다. 사후 제외에 따른 선택 편향 가능성이 있으므로 별도의 사전 고정 평가라고 주장하지 않습니다. 모델·설정·학습은 바꾸지 않았습니다.

| 평가 | 방식 | 승리/경기 | 방어 누락률 | 복수 위협 도달 비율 |
|---|---|---:|---:|---:|
| validation | baseline | 0/506 | 39.52% | 63.83% |
| validation | step1 | 0/506 | 44.30% | 65.42% |
| validation | step2 | 0/506 | 63.53% | 58.70% |
| test | baseline | 0/510 | 43.83% | 59.61% |
| test | step1 | 0/510 | 39.31% | 66.47% |
| test | step2 | 0/510 | 61.73% | 60.78% |

원 시험0/512승과 겹침을 제외한0/510승은 모든 방식에서 같은 결론입니다. 보조 결과 원본은 clean_full_game_subset.json, 재등장 위치는 overlap_details.json입니다.

## 중간 판의 첫 수와 마지막 승리

각 평가 시작 판은 실제 교대 기보로 도달 가능하고, 동일 시작 판·상대 난수 스트림을 사용합니다. 초기 정답을 맞혔는지와 지정한 자기 착수 수 안에 승리했는지, 경기 종료까지 승리했는지를 분리합니다.

| 평가 | 방식 | 모델 | 자기 착수 깊이 | 첫 수 정답/128 | 깊이 내 승리/128 | 최종 승리/128 |
|---|---|---|---:|---:|---:|---:|
| validation | baseline | fly | 1 | 87 | 87 | 89 |
| validation | baseline | fly | 2 | 90 | 50 | 51 |
| validation | baseline | fly | 3 | 97 | 3 | 5 |
| validation | baseline | gru | 1 | 93 | 93 | 100 |
| validation | baseline | gru | 2 | 93 | 61 | 65 |
| validation | baseline | gru | 3 | 102 | 7 | 8 |
| validation | step1 | fly | 1 | 102 | 102 | 105 |
| validation | step1 | fly | 2 | 99 | 77 | 79 |
| validation | step1 | fly | 3 | 107 | 14 | 15 |
| validation | step1 | gru | 1 | 105 | 105 | 109 |
| validation | step1 | gru | 2 | 101 | 83 | 85 |
| validation | step1 | gru | 3 | 106 | 18 | 21 |
| validation | step2 | fly | 1 | 106 | 106 | 114 |
| validation | step2 | fly | 2 | 101 | 90 | 95 |
| validation | step2 | fly | 3 | 112 | 34 | 39 |
| validation | step2 | gru | 1 | 107 | 107 | 114 |
| validation | step2 | gru | 2 | 103 | 92 | 96 |
| validation | step2 | gru | 3 | 113 | 38 | 42 |
| test | baseline | fly | 1 | 92 | 92 | 93 |
| test | baseline | fly | 2 | 89 | 49 | 51 |
| test | baseline | fly | 3 | 103 | 5 | 7 |
| test | baseline | gru | 1 | 90 | 90 | 93 |
| test | baseline | gru | 2 | 93 | 51 | 54 |
| test | baseline | gru | 3 | 106 | 8 | 10 |
| test | step1 | fly | 1 | 97 | 97 | 101 |
| test | step1 | fly | 2 | 98 | 68 | 69 |
| test | step1 | fly | 3 | 113 | 19 | 21 |
| test | step1 | gru | 1 | 100 | 100 | 105 |
| test | step1 | gru | 2 | 99 | 72 | 75 |
| test | step1 | gru | 3 | 119 | 22 | 28 |
| test | step2 | fly | 1 | 103 | 103 | 106 |
| test | step2 | fly | 2 | 100 | 85 | 89 |
| test | step2 | fly | 3 | 118 | 35 | 40 |
| test | step2 | gru | 1 | 108 | 108 | 110 |
| test | step2 | gru | 2 | 103 | 87 | 90 |
| test | step2 | gru | 3 | 118 | 45 | 50 |

| 평가 | 방식 | 깊이 | 합산 첫 수 정답률 | 합산 깊이 내 완승률 | 합산 최종 승률 |
|---|---|---:|---:|---:|---:|
| validation | baseline | 1 | 180/256 (70.31%) | 180/256 (70.31%) | 189/256 (73.83%) |
| validation | baseline | 2 | 183/256 (71.48%) | 111/256 (43.36%) | 116/256 (45.31%) |
| validation | baseline | 3 | 199/256 (77.73%) | 10/256 (3.91%) | 13/256 (5.08%) |
| validation | step1 | 1 | 207/256 (80.86%) | 207/256 (80.86%) | 214/256 (83.59%) |
| validation | step1 | 2 | 200/256 (78.12%) | 160/256 (62.50%) | 164/256 (64.06%) |
| validation | step1 | 3 | 213/256 (83.20%) | 32/256 (12.50%) | 36/256 (14.06%) |
| validation | step2 | 1 | 213/256 (83.20%) | 213/256 (83.20%) | 228/256 (89.06%) |
| validation | step2 | 2 | 204/256 (79.69%) | 182/256 (71.09%) | 191/256 (74.61%) |
| validation | step2 | 3 | 225/256 (87.89%) | 72/256 (28.12%) | 81/256 (31.64%) |
| test | baseline | 1 | 182/256 (71.09%) | 182/256 (71.09%) | 186/256 (72.66%) |
| test | baseline | 2 | 182/256 (71.09%) | 100/256 (39.06%) | 105/256 (41.02%) |
| test | baseline | 3 | 209/256 (81.64%) | 13/256 (5.08%) | 17/256 (6.64%) |
| test | step1 | 1 | 197/256 (76.95%) | 197/256 (76.95%) | 206/256 (80.47%) |
| test | step1 | 2 | 197/256 (76.95%) | 140/256 (54.69%) | 144/256 (56.25%) |
| test | step1 | 3 | 232/256 (90.62%) | 41/256 (16.02%) | 49/256 (19.14%) |
| test | step2 | 1 | 211/256 (82.42%) | 211/256 (82.42%) | 216/256 (84.38%) |
| test | step2 | 2 | 203/256 (79.30%) | 172/256 (67.19%) | 179/256 (69.92%) |
| test | step2 | 3 | 236/256 (92.19%) | 80/256 (31.25%) | 90/256 (35.16%) |

## 첫 수가 맞았던 경기의 연결 성공

| 평가 | 방식 | 모델 | 깊이 | 첫 수 정답 경기 중 깊이 내 완승 |
|---|---|---|---:|---:|
| validation | baseline | fly | 1 | 87/87 |
| validation | baseline | fly | 2 | 50/90 |
| validation | baseline | fly | 3 | 3/97 |
| validation | baseline | gru | 1 | 93/93 |
| validation | baseline | gru | 2 | 61/93 |
| validation | baseline | gru | 3 | 7/102 |
| validation | step1 | fly | 1 | 102/102 |
| validation | step1 | fly | 2 | 77/99 |
| validation | step1 | fly | 3 | 14/107 |
| validation | step1 | gru | 1 | 105/105 |
| validation | step1 | gru | 2 | 83/101 |
| validation | step1 | gru | 3 | 18/106 |
| validation | step2 | fly | 1 | 106/106 |
| validation | step2 | fly | 2 | 90/101 |
| validation | step2 | fly | 3 | 34/112 |
| validation | step2 | gru | 1 | 107/107 |
| validation | step2 | gru | 2 | 92/103 |
| validation | step2 | gru | 3 | 38/113 |
| test | baseline | fly | 1 | 92/92 |
| test | baseline | fly | 2 | 49/89 |
| test | baseline | fly | 3 | 5/103 |
| test | baseline | gru | 1 | 90/90 |
| test | baseline | gru | 2 | 51/93 |
| test | baseline | gru | 3 | 8/106 |
| test | step1 | fly | 1 | 97/97 |
| test | step1 | fly | 2 | 68/98 |
| test | step1 | fly | 3 | 19/113 |
| test | step1 | gru | 1 | 100/100 |
| test | step1 | gru | 2 | 72/99 |
| test | step1 | gru | 3 | 22/119 |
| test | step2 | fly | 1 | 103/103 |
| test | step2 | fly | 2 | 85/100 |
| test | step2 | fly | 3 | 35/118 |
| test | step2 | gru | 1 | 108/108 |
| test | step2 | gru | 2 | 87/103 |
| test | step2 | gru | 3 | 45/118 |

## 유형별 끝까지 승리

| 평가 | 방식 | 모델 | 깊이 | 유형 | 깊이 내 승리/경기 | 최종 승리/경기 |
|---|---|---|---:|---|---:|---:|
| validation | baseline | fly | 1 | broken_four | 54/64 | 54/64 |
| validation | baseline | fly | 1 | four | 33/64 | 35/64 |
| validation | baseline | fly | 2 | broken_three | 7/32 | 7/32 |
| validation | baseline | fly | 2 | counterattack | 19/32 | 19/32 |
| validation | baseline | fly | 2 | cross | 17/32 | 18/32 |
| validation | baseline | fly | 2 | open_three | 7/32 | 7/32 |
| validation | baseline | fly | 3 | broken_chain | 0/32 | 2/32 |
| validation | baseline | fly | 3 | counterattack_chain | 0/32 | 0/32 |
| validation | baseline | fly | 3 | diagonal_chain | 0/32 | 0/32 |
| validation | baseline | fly | 3 | straight_chain | 3/32 | 3/32 |
| validation | baseline | gru | 1 | broken_four | 56/64 | 57/64 |
| validation | baseline | gru | 1 | four | 37/64 | 43/64 |
| validation | baseline | gru | 2 | broken_three | 12/32 | 13/32 |
| validation | baseline | gru | 2 | counterattack | 20/32 | 20/32 |
| validation | baseline | gru | 2 | cross | 18/32 | 20/32 |
| validation | baseline | gru | 2 | open_three | 11/32 | 12/32 |
| validation | baseline | gru | 3 | broken_chain | 1/32 | 2/32 |
| validation | baseline | gru | 3 | counterattack_chain | 2/32 | 2/32 |
| validation | baseline | gru | 3 | diagonal_chain | 1/32 | 1/32 |
| validation | baseline | gru | 3 | straight_chain | 3/32 | 3/32 |
| validation | step1 | fly | 1 | broken_four | 57/64 | 57/64 |
| validation | step1 | fly | 1 | four | 45/64 | 48/64 |
| validation | step1 | fly | 2 | broken_three | 16/32 | 17/32 |
| validation | step1 | fly | 2 | counterattack | 27/32 | 27/32 |
| validation | step1 | fly | 2 | cross | 21/32 | 22/32 |
| validation | step1 | fly | 2 | open_three | 13/32 | 13/32 |
| validation | step1 | fly | 3 | broken_chain | 2/32 | 3/32 |
| validation | step1 | fly | 3 | counterattack_chain | 5/32 | 5/32 |
| validation | step1 | fly | 3 | diagonal_chain | 1/32 | 1/32 |
| validation | step1 | fly | 3 | straight_chain | 6/32 | 6/32 |
| validation | step1 | gru | 1 | broken_four | 59/64 | 60/64 |
| validation | step1 | gru | 1 | four | 46/64 | 49/64 |
| validation | step1 | gru | 2 | broken_three | 16/32 | 17/32 |
| validation | step1 | gru | 2 | counterattack | 30/32 | 30/32 |
| validation | step1 | gru | 2 | cross | 23/32 | 24/32 |
| validation | step1 | gru | 2 | open_three | 14/32 | 14/32 |
| validation | step1 | gru | 3 | broken_chain | 4/32 | 6/32 |
| validation | step1 | gru | 3 | counterattack_chain | 6/32 | 6/32 |
| validation | step1 | gru | 3 | diagonal_chain | 1/32 | 2/32 |
| validation | step1 | gru | 3 | straight_chain | 7/32 | 7/32 |
| validation | step2 | fly | 1 | broken_four | 61/64 | 62/64 |
| validation | step2 | fly | 1 | four | 45/64 | 52/64 |
| validation | step2 | fly | 2 | broken_three | 15/32 | 16/32 |
| validation | step2 | fly | 2 | counterattack | 29/32 | 30/32 |
| validation | step2 | fly | 2 | cross | 26/32 | 27/32 |
| validation | step2 | fly | 2 | open_three | 20/32 | 22/32 |
| validation | step2 | fly | 3 | broken_chain | 10/32 | 12/32 |
| validation | step2 | fly | 3 | counterattack_chain | 11/32 | 11/32 |
| validation | step2 | fly | 3 | diagonal_chain | 1/32 | 3/32 |
| validation | step2 | fly | 3 | straight_chain | 12/32 | 13/32 |
| validation | step2 | gru | 1 | broken_four | 61/64 | 62/64 |
| validation | step2 | gru | 1 | four | 46/64 | 52/64 |
| validation | step2 | gru | 2 | broken_three | 16/32 | 17/32 |
| validation | step2 | gru | 2 | counterattack | 31/32 | 31/32 |
| validation | step2 | gru | 2 | cross | 26/32 | 27/32 |
| validation | step2 | gru | 2 | open_three | 19/32 | 21/32 |
| validation | step2 | gru | 3 | broken_chain | 10/32 | 11/32 |
| validation | step2 | gru | 3 | counterattack_chain | 13/32 | 13/32 |
| validation | step2 | gru | 3 | diagonal_chain | 1/32 | 4/32 |
| validation | step2 | gru | 3 | straight_chain | 14/32 | 14/32 |
| test | baseline | fly | 1 | broken_four | 53/64 | 53/64 |
| test | baseline | fly | 1 | four | 39/64 | 40/64 |
| test | baseline | fly | 2 | broken_three | 7/32 | 8/32 |
| test | baseline | fly | 2 | counterattack | 21/32 | 22/32 |
| test | baseline | fly | 2 | cross | 16/32 | 16/32 |
| test | baseline | fly | 2 | open_three | 5/32 | 5/32 |
| test | baseline | fly | 3 | broken_chain | 1/32 | 1/32 |
| test | baseline | fly | 3 | counterattack_chain | 1/32 | 1/32 |
| test | baseline | fly | 3 | diagonal_chain | 0/32 | 1/32 |
| test | baseline | fly | 3 | straight_chain | 3/32 | 4/32 |
| test | baseline | gru | 1 | broken_four | 53/64 | 54/64 |
| test | baseline | gru | 1 | four | 37/64 | 39/64 |
| test | baseline | gru | 2 | broken_three | 8/32 | 9/32 |
| test | baseline | gru | 2 | counterattack | 21/32 | 22/32 |
| test | baseline | gru | 2 | cross | 16/32 | 17/32 |
| test | baseline | gru | 2 | open_three | 6/32 | 6/32 |
| test | baseline | gru | 3 | broken_chain | 1/32 | 2/32 |
| test | baseline | gru | 3 | counterattack_chain | 2/32 | 2/32 |
| test | baseline | gru | 3 | diagonal_chain | 0/32 | 1/32 |
| test | baseline | gru | 3 | straight_chain | 5/32 | 5/32 |
| test | step1 | fly | 1 | broken_four | 56/64 | 57/64 |
| test | step1 | fly | 1 | four | 41/64 | 44/64 |
| test | step1 | fly | 2 | broken_three | 13/32 | 13/32 |
| test | step1 | fly | 2 | counterattack | 26/32 | 26/32 |
| test | step1 | fly | 2 | cross | 21/32 | 21/32 |
| test | step1 | fly | 2 | open_three | 8/32 | 9/32 |
| test | step1 | fly | 3 | broken_chain | 3/32 | 3/32 |
| test | step1 | fly | 3 | counterattack_chain | 6/32 | 7/32 |
| test | step1 | fly | 3 | diagonal_chain | 0/32 | 1/32 |
| test | step1 | fly | 3 | straight_chain | 10/32 | 10/32 |
| test | step1 | gru | 1 | broken_four | 58/64 | 58/64 |
| test | step1 | gru | 1 | four | 42/64 | 47/64 |
| test | step1 | gru | 2 | broken_three | 15/32 | 15/32 |
| test | step1 | gru | 2 | counterattack | 25/32 | 25/32 |
| test | step1 | gru | 2 | cross | 22/32 | 24/32 |
| test | step1 | gru | 2 | open_three | 10/32 | 11/32 |
| test | step1 | gru | 3 | broken_chain | 5/32 | 7/32 |
| test | step1 | gru | 3 | counterattack_chain | 6/32 | 7/32 |
| test | step1 | gru | 3 | diagonal_chain | 0/32 | 3/32 |
| test | step1 | gru | 3 | straight_chain | 11/32 | 11/32 |
| test | step2 | fly | 1 | broken_four | 54/64 | 55/64 |
| test | step2 | fly | 1 | four | 49/64 | 51/64 |
| test | step2 | fly | 2 | broken_three | 19/32 | 19/32 |
| test | step2 | fly | 2 | counterattack | 27/32 | 27/32 |
| test | step2 | fly | 2 | cross | 25/32 | 28/32 |
| test | step2 | fly | 2 | open_three | 14/32 | 15/32 |
| test | step2 | fly | 3 | broken_chain | 10/32 | 12/32 |
| test | step2 | fly | 3 | counterattack_chain | 10/32 | 11/32 |
| test | step2 | fly | 3 | diagonal_chain | 0/32 | 2/32 |
| test | step2 | fly | 3 | straight_chain | 15/32 | 15/32 |
| test | step2 | gru | 1 | broken_four | 55/64 | 56/64 |
| test | step2 | gru | 1 | four | 53/64 | 54/64 |
| test | step2 | gru | 2 | broken_three | 17/32 | 18/32 |
| test | step2 | gru | 2 | counterattack | 28/32 | 28/32 |
| test | step2 | gru | 2 | cross | 27/32 | 29/32 |
| test | step2 | gru | 2 | open_three | 15/32 | 15/32 |
| test | step2 | gru | 3 | broken_chain | 12/32 | 14/32 |
| test | step2 | gru | 3 | counterattack_chain | 15/32 | 15/32 |
| test | step2 | gru | 3 | diagonal_chain | 0/32 | 3/32 |
| test | step2 | gru | 3 | straight_chain | 18/32 | 18/32 |

## 학습 예산과 혼합

각 군은 새 초기화와 같은 보조600×128, PPO300×32×16, 에포크2 및 에포크별 보조128을 사용했습니다. 보조 제시153,600개·정책 결정153,600개입니다. 중간 판에 필요한 기보 연산량·시간은 추가되므로 같은 계산량이라고 주장하지 않습니다.

새 경기50%는 빈 판,50%는 중간 판입니다. 비율은 경기 시작 수 기준이며 정책 결정 수 비율과 다릅니다. 중간 판 깊이는 업데이트1~100:1착수,101~200:2착수,201~300:3착수입니다. 상대는30→60→85% 규칙 커리큘럼, 평가는100% 규칙 상대입니다. 혼합 학습 승률을 빈 판 실전 승률로 해석하지 않습니다.

| 방식 | 모델 | 파라미터 | 보조 제시 | 정책 결정 | 보조 기보 관측 | 추가 수집 기보 관측 | 시간(초) | 최대 할당 GiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | fly | 300,370 | 153,600 | 153,600 | 1,660,963 | 0 | 61.2 | 0.305 |
| baseline | gru | 300,386 | 153,600 | 153,600 | 1,660,963 | 0 | 56.8 | 0.296 |
| step1 | fly | 300,370 | 153,600 | 153,600 | 1,660,963 | 86,487 | 139.1 | 0.305 |
| step1 | gru | 300,386 | 153,600 | 153,600 | 1,660,963 | 86,363 | 127.2 | 0.296 |
| step2 | fly | 300,370 | 153,600 | 153,600 | 1,597,040 | 98,633 | 138.9 | 0.294 |
| step2 | gru | 300,386 | 153,600 | 153,600 | 1,597,040 | 99,141 | 128.3 | 0.284 |

step1/fly 경기 시작 수: {'0': 8492, '1': 1948, '2': 3006, '3': 3480}. 깊이0은 빈 판입니다.
학습 중 시작 유형별 종료 결과: {'0': {'games': 8471, 'wins': 655, 'within_horizon_wins': 0}, '1': {'games': 1948, 'wins': 960, 'within_horizon_wins': 75}, '2': {'games': 3006, 'wins': 265, 'within_horizon_wins': 30}, '3': {'games': 3469, 'wins': 543, 'within_horizon_wins': 363}}.
범주별 보조 제시 횟수: {'1': 31788, '2': 42492, '3': 32623, '4': 38584, '5': 3819, '6': 4294}.


step1/gru 경기 시작 수: {'0': 8422, '1': 1955, '2': 2927, '3': 3516}. 깊이0은 빈 판입니다.
학습 중 시작 유형별 종료 결과: {'0': {'games': 8408, 'wins': 661, 'within_horizon_wins': 0}, '1': {'games': 1955, 'wins': 894, 'within_horizon_wins': 78}, '2': {'games': 2927, 'wins': 269, 'within_horizon_wins': 14}, '3': {'games': 3498, 'wins': 529, 'within_horizon_wins': 323}}.
범주별 보조 제시 횟수: {'1': 31788, '2': 42492, '3': 32623, '4': 38584, '5': 3819, '6': 4294}.


step2/fly 경기 시작 수: {'0': 10285, '1': 2634, '2': 3524, '3': 4170}. 깊이0은 빈 판입니다.
학습 중 시작 유형별 종료 결과: {'0': {'games': 10273, 'wins': 994, 'within_horizon_wins': 0}, '1': {'games': 2634, 'wins': 2030, 'within_horizon_wins': 937}, '2': {'games': 3524, 'wins': 1577, 'within_horizon_wins': 860}, '3': {'games': 4150, 'wins': 846, 'within_horizon_wins': 509}}.
범주별 보조 제시 횟수: {'1': 25623, '2': 25560, '3': 25592, '4': 25437, '5': 25838, '6': 25550}.


step2/gru 경기 시작 수: {'0': 10348, '1': 2685, '2': 3440, '3': 4284}. 깊이0은 빈 판입니다.
학습 중 시작 유형별 종료 결과: {'0': {'games': 10334, 'wins': 999, 'within_horizon_wins': 0}, '1': {'games': 2685, 'wins': 2118, 'within_horizon_wins': 993}, '2': {'games': 3440, 'wins': 1556, 'within_horizon_wins': 812}, '3': {'games': 4266, 'wins': 934, 'within_horizon_wins': 577}}.
범주별 보조 제시 횟수: {'1': 25623, '2': 25560, '3': 25592, '4': 25437, '5': 25838, '6': 25550}.


2단계 새 시작 판 1,536개, 공격·후속·예방 보조 예제 3,730개(예방 658개). 전체 보조 자료20,000개를 유지했습니다. 자료별 범주 빈도: {'1': 4999, '2': 4806, '3': 4568, '4': 4717, '5': 513, '6': 397}.
2단계는 자료의 빈도와 상관없이6범주를 같은 확률로 제시합니다. 실제 제시 횟수는 위 기록 및 run_audit.json에 포함됩니다.

## 범위·검증·해석

35개 테스트 및 실제 짧은 학습 검증 통과. 중간 판 재시작의 돌·기보·차례·증분 줄 개수 일치, 기보 burn-in의 수집/PPO 일치와 순환 가중치 기울기, 인증 깊이 내 실제 상대 완승, 범주 균형 제시를 검증했습니다.
새 검증/시험 판은 각384개(깊이별128개, 흑백 균형)이며 모든 이전 차례·현재 판을 D4 정규화 해시로 학습에서 제외했습니다. 기보와 현재 관측은 모든 자료에서 정확히 일치합니다.
시작 판 생성은 여러 구조를 갖지만 여전히 정해진 강제 전술 가족과 무작위 주변 돌의 조합입니다. 전체 오목의 장기 전략·조용한 계획을 해결했다는 의미가 아닙니다.
2단계는 공격 다양화·실제 후속 응수 지도·범주 균형·시작 풀 교체의 결합 효과이며 각 요소의 단독 인과 효과는 구분할 수 없습니다.
중간 판 성공률이 오르더라도 빈 판 경기 승률이 오르지 않으면 전체 경기로의 전이가 확인되지 않은 것입니다. 단일 학습 시드이므로 Fly 연결 구조나 순환 기억의 우월성·통계적 확증을 주장하지 않습니다.
이전 실험의 시험을 설정 선택에 재사용하지 않았으며, 두 단계가 완료된 뒤 새 시험을 고정 체크포인트로 평가합니다.
2단계의 새 시험 방어 누락률은61.16%로 대조43.55% 및1단계39.22%보다 높았습니다. 공격 연결 향상이 방어 약화를 동반했으므로 전체 경기 전략이 개선됐다고 볼 수 없습니다.
실제 평가 기보 전체와 각 군 학습 판의 겹침 여부는 run_audit.json에 별도 기록합니다. 복수 위협 도달 비율은 방어 누락 감소로 패배 양상이 바뀌어 증가할 수도 있으므로 단독으로 예방 능력 악화의 증거로 해석하지 않습니다.

- [사전 프로토콜](../../midgame_protocol.md)
- [평가 수치](evaluation.json)
- [1단계 자료 감사](data_audit_step1.json)
- [예산·실제 기보 감사](run_audit.json)
- [겹침 기보를 공통 제외한 보조 평가](clean_full_game_subset.json)
- [첫 수 정답 경기의 조건부 완승](conditional_completion.json)
- [중간 판 체크포인트 복원 검증](resume_audit.json)

~~~powershell
.venv/Scripts/python.exe -m gomoku.midgame_evaluate --root gomoku/runs/midgame_20261007 --arms baseline step1 step2 --splits validation test
.venv/Scripts/python.exe -m gomoku.midgame_audit --root gomoku/runs/midgame_20261007 --phase runs
.venv/Scripts/python.exe -m gomoku.midgame_report
~~~

# MaleCNS 소규모 순환망 비교 연구

2026-10-06: 현재 과제는 **12×12 일반 오목 방어 학습과 실력 재평가**다. 흑백 모두 5개 이상 연속이면 승리하며 금수와 시작 수 제한이 없다. 전체 판 관측으로 실제 Fly·재배선 Fly·RNN·GRU를 새로 학습하고 맞대결한다. 15×15 렌주와 부분관측은 후속 단계다.

- [12×12 방어 학습 프로토콜](../gomoku/defense12_protocol.md): 공통 전술 보조 학습, 약화 규칙 상대 PPO, 독립 전술 시험과 새 상대 평가.
- [12×12 방어 학습 결과](../gomoku/runs/defense12_20261006/results.md) · [12×12 맞대결 기보](../gomoku/runs/defense12_20261006/replays.html).
- 12×12 실행 폴더: `gomoku/runs/defense12_20261006/`. 9×9 체크포인트와 당시 코드는 `gomoku/runs/pilot_20261006/` 및 하위 `source/`에 보존했다.

- [오목 실험 프로토콜](../gomoku/protocol.md): 규칙, 공통 구조, 학습 예산, 평가 방법.
- [첫 오목 결과 보고서](../gomoku/runs/pilot_20261006/results.md) · [모델 맞대결 기보 재생](../gomoku/runs/pilot_20261006/replays.html).
- [9×9·12×12·15×15 GPU 자원 진단](../gomoku/resource_benchmark.md): 배치32·역전파16/64의 합성 계산 측정. 큰 판의 실제 게임 학습 결과와 구분한다.
- [오목 코드](../gomoku/): 환경·정책·PPO 학습·대결 평가.
- [원 기획서](../drosophila_project_plan_flow.txt) · [연구 실행 계획](../RESEARCH_PROTOCOL.md) · [기존 핵심 질문 보고서](../previous_research/reports/foundation/core_research_questions_report.txt).
- [이전 실험 보관](../previous_research/): 기존 코드 78개, 실험 보고서·의존성 파일, 원자료·체크포인트 전체. 기획서·프로토콜은 최상위에 유지했다. [이동 검증 목록](../previous_research/manifests/relocation_manifest.json).
- `.venv/`: 공통 실행 환경. [사용 버전 기록](../gomoku/environment_versions.txt).

**12×12 완료:** 4모델 × 새 시드 3개, 각각 전술 보조 학습 1,800업데이트 후 PPO 1,200업데이트(614,400 학습자 결정)를 완료했다. 새 약화 규칙 상대 승률은 실제 Fly/재배선 Fly/RNN/GRU 순서로 88.02/86.20/86.72/89.32%였지만 완전 규칙 상대는 모두 0%였다. 기보를 입력한 상태의 독립 필수 방어 정확도는 3.87/3.78/0.88/1.17%로 낮았다. 학습 판 방어 정확도도 약22~27%여서 기본 전술의 미학습과 일반화·순환 상태 영향을 함께 보완해야 한다. 맞대결 2,304경기와 기보 288개 재판정을 완료했다. 실제 Fly는 재배선 Fly에게 176승 208패로 9×9 우세가 재현되지 않았다. 학습 PyTorch 최대 allocated 약0.164GiB, reserved 약0.195GiB. 테스트19개가 통과했다. 다음 우선순위는 기보 상태를 포함한 보조 학습 → 공통 공간 인코더/출력 대조 → 복수 위협·상대 리그 강화이며 15×15 확장은 뒤에 둔다.

9×9 기준 실험은 최상위 폴더에서 아래 명령으로 실행한다. 새 12×12 명령은 방어 학습 프로토콜에 있다:

```powershell
.\.venv\Scripts\python.exe -m unittest gomoku.test_gomoku -v
.\.venv\Scripts\python.exe -m gomoku.train
.\.venv\Scripts\python.exe -m gomoku.evaluate
.\.venv\Scripts\python.exe -m gomoku.make_report
```

모델별 새 학습 시드 3개, 최종 예산 체크포인트를 `gomoku/runs/pilot_20261006/`에 저장한다. 동일 설정은 체크포인트에서 이어서 실행하고 완료한 실행은 재사용한다. 아래는 이전 연구의 날짜별 기록이며 당시의 '오목 보류' 문구는 과거 결정이다.

첫 파일럿 12개 학습과 맞대결 2,304경기를 완료했다. 새 무작위 상대 승률은 실제 Fly 100%, 재배선 Fly·RNN·GRU 각각 99.87%였으며, 즉시 승리/방어 규칙 상대에게는 모두 0%였다. 맞대결에서 실제 Fly는 재배선 Fly에게 369승 15패였지만, 규칙 상대에게 대응하지 못하는 초급 정책이므로 실제 연결 구조의 고유한 우위로 해석하지 않는다. 다음 단계는 더 강한 공통 상대와 과거 정책을 상대로 하는 학습이다. 환경/학습 검증 14개·이전 코드 검증 43개가 통과했고 기보 288개를 다시 판정했다.

---

[원 기획서](../drosophila_project_plan_flow.txt) · [실행 프로토콜](../RESEARCH_PROTOCOL.md)

## 최신 완료: aux 일반화·실패 분리·방/복도 적용

2026-10-05: [후속 결과 보고서](../previous_research/reports/memory/multi_cue_aux_generalization_results.txt)에 고정 가중치의 새 조건 평가, 진단, 방·복도 적용을 정리했다. 기존 실험은 가중치를 저장하지 않았으므로 단서2·지연16·방해2 `aux` 모델 5종×5시드를 동일 절차로 재현해 체크포인트를 저장했다. 원 시험 정확도와 재현 정확도는 25개 모두 정확히 일치했다.

1. **새 조건 일반화:** 새 4,096개/시드와 고정 가중치에서 지연32 정확도(5시드 평균)는 실제 Fly 83.18%, 재배선 Fly 83.79%, 누설 기억 84.57%, RNN 49.06%, GRU 99.98%였다. 방해10에서는 각각 84.81/77.96/98.29/100/100%, 3단서에서는 89.41/85.63/100/56.18/79.99%였다. aux의 훈련 조건 성과가 모든 새 조건에 유지되지는 않았다.
2. **엄격한 새 단서 순서:** 색 쌍마다 한 제시 방향만 학습한 별도 3시드 대조에서, 반대 방향 정확도는 실제 Fly 99.88%, 재배선 Fly 95.52%, 누설 기억 100%, RNN 67.15%, GRU 73.71%였다. 다섯 모델 모두 학습 방향과 동일 에피소드를 그 방향으로 되돌린 대조는 100%였다. 이 별도 학습 결과는 원 체크포인트의 고정 가중치 전이와 구분한다.
3. **92.3% 재배선 시드 원인:** 시드3104의 지연 후 상태에서는 색별 종류를 선형 판독기로 100%, 학습된 aux 판독기로 99.85% 읽어냈다. 새 표본에서 원 답 출력은 92.53%였지만 aux 출력에서 질문 색을 명시 선택하면 99.90%였다. 이 시드에서는 정보 소실보다 질문에 맞는 답 출력 선택이 주된 문제다.
4. **방·복도:** 7×7 MiniGrid 물체·색·상태 관측의 3시드 짝 비교에서 aux만으로 마지막 답의 일관된 개선은 없었다. 별도 학습한 공통 화면 기반 전진·정지 정책은 미사용 복도 새 512개에서 이동 성공률 100%였고, 같은 자율 궤적에서 실제 Fly의 종단 답은 base 67.06%·aux 66.15%였다. 이 이동 과제는 직선 복도의 전진·정지이며 회전·탐색은 포함하지 않는다.

원자료와 체크포인트는 `data/multi_cue_aux_transfer/`, `data/multi_cue_visual_aux/`에 보존했다. 집계는 `summarize_multi_cue_aux_transfer.py`로 재생성한다. 오목과 5,000개 확장은 계속 보류한다.

## 이전 완료: 다중 단서 후속 연구 III — 결합 학습과 방해 입력 기억 유지 개선

2026-10-04: [후속 연구 III 프로토콜](../multi_cue_binding_protocol.txt)에 따라 조정(88회)과 최종 실험(200회, 새 시드 3101~3105, 8,000 업데이트)을 완료했다. [후속 연구 III 결과 보고서](../previous_research/reports/memory/multi_cue_binding_results.txt)에 조건과 원자료를 정리했다.

1. **조정 및 개입 선정:** 조정 시드(901·902)에서 그래프 계열(실제 Fly, 누설 기억)의 base 모델은 평균 검증 67.40%에 머물렀으나, 모든 단서의 종류를 보고하는 보조 손실(`aux`), 결합 입력(`bind`), 쓰기 게이트(`gate`) 및 그 조합 모두 100.00%를 기록했다. 사전 규칙에 따라 최고 성능(100%)을 달성한 최소 구성 단일 개입인 **`aux`**(모든 단서 색의 종류를 보조 학습하는 손실)가 선정되었다.
2. **최종 200회 비교 (새 5시드, 4개 난이도 조건):**
   - **방해 6개 (단서2·지연8·방해6):** base 모델은 실제 Fly(66.11%), 재배선 Fly(66.51%), RNN(66.44%), 누설 기억(66.43%) 모두 66%대에 정체(GRU만 100%)했으나, `aux`를 사용해 처음부터 별도 학습한 모델은 **5개 모델 전 시드 시험 100.00%**였다. 앞의 네 모델은 +33~34pp, GRU는 +0pp였다.
   - **긴 지연 (단서2·지연16·방해2):** base GRU의 2개 시드는 34.0%·33.1%였으나, 같은 시드의 새 `aux` 학습은 최종 시험100%였다. 실제 Fly 평균99.99%, 재배선98.47%, RNN·누설 기억100%였다. 실제 Fly 시드3104는99.951%, 재배선 시드3104는약92.3%로, 모두100%인 것은 아니다. GRU의 500회 수치는 검증90% 이상 도달 시점이며 붕괴 체크포인트 복구나 시험100% 도달 시점이 아니다.
   - **다중 단서 부하 (단서3·지연8·방해2 및 단서4·지연8·방해2):** `aux`에서 **5개 모델 전 시드 시험100%**였다. 단서4 base 평균은 실제 Fly67.22%(한 시드는99.9%), 재배선59.05%, RNN59.19%, 누설 기억59.13%였으며 개선은 +32~41pp였다. GRU base는 이미100%였다.
3. **진단과 해석:** 실패한 실제 Fly는 첫·마지막 단서 정확도가66.5%·65.7%로 비슷해, '마지막 단서만 기억'이나 '새 입력이 앞 단서를 지움'을 원인으로 확정할 수 없다. 누설 기억 일부 조건에는 뒤 단서가 유리한 경향이 있다. `aux`의 높은 위치별 답 정확도는 단서 활용 개선의 근거이며, 상태의 모든 정보가 완전히 보존됨을 직접 측정한 결과는 아니다. 질문 색 재사용에 따른 일관된 성능 저하는 없었지만, base 일부 차이는1pp를 넘었고 방해 간섭 전반을 배제할 수 없다.
4. **현재 결론:** 추가 정답 감독(`aux`)으로 이 3종류 다중 단서 과제의 성능 정체를 크게 개선했다. 조정에서 `bind`·`gate` 단독도100%였으므로 aux의 유일한 효과나 실제 Fly 연결 구조의 우수성을 입증한 것은 아니다. 다음에는 더 긴 지연·새 방해 조건·방/복도 관측으로의 일반화와 보존·결합·질문 활용의 분리 진단이 필요하다.

원자료는 `data/multi_cue_binding/`에 보존되었으며, `summarize_multi_cue_binding.py report`로 언제든 재요약할 수 있다. 오목 및 5,000개 확장은 계속 보류한다.

## 이전 완료: 다중 단서 후속 연구 II (출력층·관측 분리·어려운 조건 반복)

2026-10-04: [후속 연구 II 프로토콜](../multi_cue_recovery_protocol.txt)에 따라 1→2→3을 순서대로 완료했다. 고정 기억15개에서 출력층60개를 비교한 1단계와 방·복도 진단15개·관측 요약 조정30회·최종15회의 2단계를 완료했다. 출력층 재학습으로 붕괴한 GRU는 새 시험100%로 복구했지만 실패한 Fly는 약66~67%였다. 화면의 보이는 속성을 직접 요약한 별도 학습에서 실제 Fly는 학습 길이 시험97.35%, 미사용 길이96.65%, 대기 추가82.73%였다. 3단계 지연16/방해6 두 조건의 본체90회와 출력층 보수90회 원자료는 [후속 연구 II 결과](../previous_research/reports/memory/multi_cue_recovery_results.txt)에 정리했다.

## 최신 진행: 2026-10-04 다중 단서 1→2→3 완료

[결과 보고서](../previous_research/reports/memory/multi_cue_followup_results.txt) · [기준·진단·부하 프로토콜](../multi_cue_followup_protocol.txt) · [방·복도 프로토콜](../multi_cue_visual_protocol.txt)

1. **기준 조건 재현:** 단서2·지연4·방해2, 새 시드3개, 16,000 업데이트. 실제 Fly 77.83%, 재배선5개 평균 68.98%, RNN 100%, GRU 77.91%, 간선 없는 누설 기억 100%. 실제 Fly는 100/66.36/67.14%로 시드 차이가 컸다.
2. **원인 진단:** 실패한 Fly 두 시드는 질문 색을 바꿔도 같은 답을 냈다. 단서 직후의 상태에서는 두 단서 모두 판독되지만 지연 뒤 판독 정확도가 떨어졌다. 붕괴한 GRU 한 시드는 출력층 ReLU 128개가 시험 전체에서 비활성이었다. 따라서 기억 보관과 질문에 맞는 출력 선택을 함께 개선해야 한다.
3. **부하·방·복도 비교:** 9조건 135회와 MiniGrid 관측 조정30회·최종15회를 완료했다. 방·복도 시험 평균은 실제 Fly/재배선/누설 기억 66.73/69.74/66.02%, 미사용 길이는 66.73/69.30/67.07%였다. 이 관측 비교는 고정 전진 경로이며 자율 주행 학습 결과가 아니다.

**현재 판단:** 실제 연결 배치의 일관된 우위는 확인되지 않았다. 다음 권장 순서는 출력층 안정화 → 화면 인식과 기억 실패의 분리 → 어려운 지연·방해 조건의 새 시드 반복이다. 약66.7%만으로 마지막 단서만 기억한다고 단정할 수 없다. 오목과 5,000개 확장은 계속 보류한다. 기존14개와 추가6개, 총20개 테스트를 통과했다.

## 이전 진행 기록

2026-10-03 다중 단서 실험 점검: [코드·결과 점검 보고서](../previous_research/reports/memory/multi_cue_audit_results.txt). 조정된 RNN·GRU는 2단서 검증에서 100%, Fly는 조정 시드 901의 16,000 업데이트에서도 약 65~67%였다. 간선 없는 누설 기억은 lr=0.003에서 11,000회부터 검증 100%였다. 최종 9조건 비교는 아직 완료되지 않았다. 실행 스크립트의 조건 배열 전달 오류를 수정했고 `run_multi_cue_grid.ps1 -DryRun`으로 9조건을 검증했다. 다음 권장 순서는 2단서 기준 조건의 새 시드 재현·단서 순서별 평가, Fly 고원 원인 진단, 부하 조건 확대다. 오목과 5,000개 확장은 계속 보류한다.

현재까지 **Phase 0: 1,000개·2,000개 뉴런의 실제 MaleCNS 연결망 추출, GPU 자원 확인, 짧은 기억 과제 진단**, 기존 카드 게임의 1·2단계 핵심 규칙 파일럿, MiniGrid Memory의 첫 탐색적 파일럿과 S11 기억·무작위 시작·복도 길이 후속 진단, S9 실패 분해와 동일 단계 Fly·재배선 Fly 5시드 비교, 미사용 길이의 단서·물체 관계 선택 실험을 수행했다. 오목 게임 학습 결과는 아직 없다.

기본 실험 규모는 [`project_config.py`](../previous_research/code/project_config.py)의 `PILOT_NEURON_COUNTS = (1000, 2000)`에 고정했다. 그래프 파일의 간선은 `pre_index → post_index`이고, 모델 계산 행렬은 `W[post, pre]`로 구성해 원본 방향을 유지한다.

## 데이터와 추출 방식

[Janelia MaleCNS v1.0 공식 배포처](https://male-cns.janelia.org/download/)의 뉴런 주석과 연결 테이블을 `data/`에 저장했다. 1.1GB 원본 연결 파일은 Git에 넣지 않도록 `.gitignore`에서 제외했다. 파일 해시는 생성된 JSON에 기록된다.

`build_subgraphs.py`는 `cb_intrinsic` / `CX`로 주석된 2,950개 뉴런 사이의 연결을 원본에서 배치 단위로 읽는다. 자기 연결을 제외하고 시냅스 3개 이상의 간선을 유지한다. 내부 연결의 가중 차수가 높은 순서로 1,000개, 2,000개를 선정한다. 두 그래프는 중첩되며, 게임 결과는 선정에 쓰지 않는다. 이 기준은 초기 구현용 잠정 선택이다.

| 그래프 | 뉴런 | 방향성 간선 | 최대 강연결요소 | 고립 뉴런 |
|---|---:|---:|---:|---:|
| [1,000개](../previous_research/data/subgraphs/malecns_cx_1000.json) | 1,000 | 83,336 | 1,000 | 0 |
| [2,000개](../previous_research/data/subgraphs/malecns_cx_2000.json) | 2,000 | 169,373 | 2,000 | 0 |

생성된 `npz`는 `body_ids`, `pre_index`, `post_index`, `synapse_count`를 담는다. 간선은 `pre_index → post_index` 방향이다.

`verify_subgraphs.py`는 원본 파일의 SHA-256, 뉴런 주석, 선정 순위, 모든 저장 간선의 방향·시냅스 수를 다시 확인한다. 현재 1,000개·2,000개 그래프 모두 검증을 통과했다. **뉴런은 무작위로 뽑지 않았고**, 학습 시작 시 연결 가중치만 무작위로 초기화된다. 기존 500개 그래프와 실험 기록은 초기 진단 이력으로 보존했다.

현재 `CX` 후보는 2,950개이므로 5,000개 확장 시에는 후보 뇌 영역을 넓히는 새 선정 기준이 필요하다.

## 16GB VRAM 측정

RTX 5060 Ti, PyTorch 2.11.0+cu128에서 무작위 32차원 입력, 배치 32, 64스텝 역전파, AdamW, 5회 측정의 중앙값이다. 간선 가중치만 학습하고 빈 연결에는 파라미터를 두지 않았다. 학습 동작 확인용 측정이며 MiniGrid·오목의 실제 사용량은 아니다.

| 뉴런 | 계산 방식 | 학습 스텝 중앙값 | PyTorch 최대 할당 VRAM |
|---:|---|---:|---:|
| 1,000 | 밀집 행렬 계산 | 20.5 ms | 0.043 GiB |
| 1,000 | COO 희소 계산 | 78.0 ms | 0.106 GiB |
| 2,000 | 밀집 행렬 계산 | 22.8 ms | 0.101 GiB |
| 2,000 | COO 희소 계산 | 98.2 ms | 0.345 GiB |

측정 원자료: [1,000개·2,000개 배치 32·64스텝](../previous_research/data/phase0_probe_1000_2000_b32_t64.json). 수치는 PyTorch가 추적한 할당량으로 CUDA 컨텍스트와 다른 프로세스의 VRAM을 포함하지 않는다. 이 규모에서는 **활성 간선만 학습하면서 밀집 행렬로 계산**하는 방식을 기본으로 삼는다.

## 기억 과제 학습 확인

`train_memory_sanity.py`는 길이 16의 시퀀스 첫 시점에만 보이는 0/1 신호를 마지막 시점에 맞히도록 학습한다. 이후 `check_memory_checkpoint.py`가 저장된 모델을 다시 불러와 새로 생성한 2,048개 예시와 **신호를 지운 대조군**에서 평가한다.

| 그래프 | 새 예시 정확도 | 신호 제거 정확도 |
|---|---:|---:|
| 1,000개 | 100.0% | 50.3% |
| 2,000개 | 100.0% | 49.9% |

둘 다 단순 진단 과제의 결과다. 신호를 지우면 우연 수준으로 내려가므로 답이 다른 입력에서 새어 나오지는 않는다. [1,000개 학습 기록](../previous_research/data/sanity/malecns_cx_1000_delay16_seed42.json)과 [2,000개 학습 기록](../previous_research/data/sanity/malecns_cx_2000_delay16_seed42.json)을 저장했다. 이 수치로 MiniGrid 기억 능력이나 RNN·GRU 대비 우수성을 주장할 수는 없다.

## 기존 카드 게임 1단계 파일럿

사용자가 만든 `C:/Users/a/Desktop/자바스크립트 게임/index.html`의 첫 단계를 [화면 없는 Python 게임](../previous_research/code/pilot_card_game.py)으로 옮겼다. 처음 본 카드 4개 중 하나가 새 카드로 바뀌면 그 위치를 고르는 과제다. REINFORCE로 선택 보상만 사용해 학습했고, Fly·RNN·GRU는 같은 관측과 파라미터 수가 비슷한 출력 구조를 사용했다. 세 시드, 각 307,200번 선택, 시드마다 새로운 1,024게임으로 평가했다.

| 뉴런 기준 | 실제 Fly | 차수 보존 재배선 Fly | Vanilla RNN | GRU |
|---:|---:|---:|---:|---:|
| 1,000 | 99.84 ± 0.15% | 99.71 ± 0.17% | 74.06 ± 44.51% | 84.77 ± 3.73% |
| 2,000 | 98.96 ± 0.15% | 99.32 ± 0.85% | 74.74 ± 43.50% | 79.92 ± 2.03% |

표본 표준편차다. RNN은 한 시드에서 실패하고 두 시드에서 거의 100%였다. 실제 연결과 재배선 연결이 모두 거의 포화되어 이 파일럿만으로 실제 연결망의 이점이나 5,000개 확장 필요성을 확인할 수 없다. 상세 조건과 한계는 [카드 게임 파일럿 보고서](../previous_research/reports/card_game/card_game_pilot_report.txt)에 기록했다. 핵심 연구 질문의 현재 답변은 [중간 보고서](../previous_research/reports/foundation/core_research_questions_report.txt)에 있다.

### 기억 지연 및 2단계 확대

1단계에서 공백 관측을 1·4·8스텝으로 바꾸고 **각 조건을 처음부터 다시 학습**했다. 1,000개 실제 Fly의 성공률은 99.8% → 89.6% → 44.7%, 재배선 Fly는 99.7% → 84.1% → 50.8%였다. 2,000개 실제 Fly는 99.0% → 77.7% → 50.4%, 재배선 Fly는 99.3% → 68.8% → 39.3%였다. RNN·GRU까지 포함한 전체 표와 시드 편차는 [지연·2단계 보고서](../previous_research/reports/card_game/card_game_delay_stage2_report.txt)에 있다. 별도로 1스텝 학습 모델을 재학습 없이 더 긴 지연에서 평가해 일반화 결과를 구분해 기록했다.

2단계에서는 원본처럼 색상 또는 이모지 카드가 무작위로 나오게 했다. 공백 1스텝에서 실제 Fly / 재배선 Fly / RNN / GRU의 평균 성공률은 1,000개 기준 95.7 / 94.8 / 98.8 / 73.5%, 2,000개 기준 94.6 / 94.8 / 73.2 / 74.7%였다. 2단계는 독립 라운드로 학습했고 원본의 가짜 흔들림과 실시간 타이머는 포함하지 않았다.

### 4스텝 결과의 독립 반복과 기준 모델 설정 점검

초기 4스텝 실험에서 실제 Fly가 재배선 Fly보다 높았지만, **새 학습 시드 5개와 재배선 그래프 5개**로 반복한 뒤 동일한 새 게임 4,096개에서 평가하니 우위가 재현되지 않았다. 1,000개는 실제 80.0% / 재배선 평균 81.0%, 2,000개는 실제 67.5% / 재배선 평균 72.8%였다. 차이의 불확실성 범위는 모두 0을 포함한다.

RNN의 학습률을 별도 조정 시드에서 0.0003으로 선택하니, 독립 시드의 4스텝 성공률이 1,000개 25.2% → 79.9%, 2,000개 25.1% → 78.1%로 올랐다. 두 크기 모두 5시드 중 1시드는 여전히 실패했다. GRU는 탐색한 학습률 중 기존 0.001이 가장 높았다. 실험 조건, 시드별 값과 한계는 [후속 검증 보고서](../previous_research/reports/card_game/card_followup_results.txt) 및 [사전 조건](../card_followup_protocol.txt)에 있다.

### RNN 실패 시드 진단과 초기화 개선

위의 실패 시드 203을 조사하니 저장된 RNN은 첫 카드 정보를 지워도 출력이 거의 변하지 않았고, 행동 확률은 거의 균등했다. 3,000업데이트로 학습을 늘리거나 엔트로피 보너스를 제거해도 회복하지 않았지만, 정답을 제공하는 지도학습 진단에서는 두 크기 모두 100%에 도달했다. 학습 전에는 첫 카드 입력 차이가 기본 초기화의 은닉 상태에서 선택 시점에 약 6%만 남았다. 은닉→은닉 행렬의 직교 초기화에서는 약 94~96%가 남았다. 이는 실패 원인을 확정하는 단일 증거가 아니며, 초기화와 보상 학습의 상호작용을 뒷받침한다.

`pilot_card_game.py --rnn-init orthogonal` 옵션을 추가했다. 4스텝의 새 학습 시드 206~210에서는 기본 초기화 대비 직교 초기화의 평균 성공률이 1,000개 96.04% → 100.00%, 2,000개 67.22% → 99.97%였다(각 시드의 새 4,096게임 평가). 8스텝 탐색에서는 직교 초기화도 2,000개 3시드 중 1개가 실패했다. 실험 설정, 시드별 결과와 한계는 [RNN 실패 원인 및 개선 보고서](../previous_research/reports/card_game/rnn_recovery_results.txt)에 있다. 이전 표는 당시의 기본 초기화 실험 기록이다.

## MiniGrid Memory 첫 파일럿

[MiniGrid 실행 프로토콜](../minigrid_pilot_protocol.txt)에 따라 MemoryS11의 부분관측 이미지로 실제 Fly·재배선 Fly·RNN·GRU를 1,000개·2,000개 기준, 각 3시드, 실행당 10,240 환경 상호작용으로 학습했다. 출발 물체가 첫 관측에 보이도록 시작 위치를 고정했고, 원래 성공 보상과 이동 3행동을 사용했다. 동일 초기 파라미터의 실제/재배선 Fly 비교, 순환 상태 초기화와 단서 제거를 확인했다.

최종 새 에피소드에서 1,000개는 네 모델 모두 성공률 0%였다. 2,000개는 실제 Fly와 GRU가 0%, 재배선 Fly 평균 16.1%, RNN 평균 19.3%였다. 후자 둘은 각각 한 시드만 약 50%에 도달했고 **출발 단서를 제거해도 점수가 같았다**. 따라서 이 결과로 모델의 기억 능력이나 연결 구조 우열을 판단할 수 없다. S11 학습 모델의 S7·S13 전이 평가에서도 단서 사용은 확인되지 않았다. 상세 수치와 한계는 [MiniGrid 파일럿 결과](../previous_research/reports/minigrid/minigrid_pilot_results.txt)에 기록했다.

이번 실행에서 2,000개 Fly의 PyTorch 최대 할당 VRAM은 약 0.09GiB였다. 자원 한도보다 학습 탐색과 기억 신호가 현재의 우선 문제다. 5,000개 확장은 보류한다.

### MiniGrid S11 기억 사용 후속 검증

[후속 결과 보고서](../previous_research/reports/minigrid/minigrid_memory_followup_results.txt)에는 복도 이동을 고정한 마지막 선택 진단, 단서 제거·교체·선택 직전 상태 초기화, 전체 이동 정책으로의 확장, PPO 추가 학습을 기록했다. 1,000개 뉴런 기준 고정 복도 선택 진단의 새 평가 512개에서 실제 Fly는 3시드 중 2개, 재배선 Fly·RNN·GRU는 각각 3개 모두 100% 정확도였다. 성공한 11개 실행은 단서를 가리거나 선택 직전 상태를 초기화하면 약 48~52%, 단서를 반대로 보여주면 0% 정확도가 되어 **S11 조건에서 첫 물체를 기억해 사용함**을 확인했다.

성공 경로를 훈련 때 제공해 전체 이동까지 학습시키면 GRU·RNN·재배선 Fly는 3시드 모두 자율 이동 성공률 100%였으나, 실제 Fly는 1/3시드만 100%였다. 이어 낮은 학습률로 원래 성공 보상의 PPO를 10,240스텝 추가했을 때, 독립 새 평가 256개에서 GRU 3시드는 모두 100%, RNN·재배선 Fly는 각각 2/3시드가 100%, 실제 Fly는 성공한 1시드만 100%를 유지했다. 정답이나 교사 경로는 평가 입력에 없고, 단서 제거·교체 시 성공률이 떨어졌다. 이 절차에는 정답·경로 지도학습이 포함되므로 앞선 보상 전용 PPO 파일럿과 직접 비교하지 않는다. S7·S13 전이는 불안정했고, 5,000개 확장과 오목은 계속 보류한다.

### 무작위 시작과 복도 길이 후속 연구

[무작위 시작·길이 전이 보고서](../previous_research/reports/minigrid/minigrid_random_start_results.txt)에 원래 MemoryS11 시작 분포, 단서 찾기, PPO 안정성, 새 복도 길이 평가를 정리했다. 원래 무작위 시작 1,000개 중 첫 관측에 출발 물체가 보인 것은 113개였다. 단서를 찾아가는 성공 경로를 **훈련 때만** 제공하니 1,000개 기준 네 모델 모두 무작위 시작에서 단서를 본 뒤 기억 선택을 학습할 수 있었다. 평가에는 교사가 없으며, 성공한 정책은 단서 제거 시 성공률이 떨어지고 단서를 반대로 보여주면 선택이 뒤집혔다. 이후 원래 성공 보상의 PPO를 추가하면 일부 시드에서 탐색·이동·선택 성능이 하락했다. 검증 체크포인트 선택 실험에서 GRU 3시드는 모두 PPO 시작 전 모델이 선택됐다.

S7·S11·S13을 함께 학습한 12개 실행은 세 훈련 길이에서 새 평가 성공률 100%, 단서 교체 시 0%였다. 그러나 학습에 쓰지 않은 S9의 3시드 평균 성공률은 실제 Fly 80.2%, 재배선 Fly 63.8%, RNN 36.2%, GRU 77.9%로 편차가 컸다. 실제 Fly의 S9 실패는 주로 분기점 도달, RNN 일부 실패는 마지막 물체 비교에서 발생했다. 서로 다른 사전 체크포인트와 3시드의 탐색 결과이므로 실제 연결 구조의 우위를 뜻하지 않는다. 오목과 5,000개 확장은 계속 보류한다.

### S9 실패 분해와 동일 단계 5시드 비교

[S9 후속 연구 보고서](../previous_research/reports/minigrid/minigrid_s9_three_step_results.txt)에 요청한 세 단계를 순서대로 기록했다. 새 S9 진단에서 12개 정책 모두 출발 단서는 봤지만, Fly 401·402는 갈림길 도달률 73.4%, RNN 401은 갈림길 도달률 100%에도 교사 경로의 마지막 선택 정확도 17.2%였다. RNN에 단서 기억·물체 비교 보조 손실을 추가한 3시드 대조는 S9 평균 33.9% → 39.6%였으나 시드별 개선·악화가 섞였고 단서 교체 대조도 통과하지 못했다.

실제 Fly와 차수 보존 재배선 Fly를 같은 시드로 생성해 초기 가중치가 같음을 재검사하고, 고정 시작 기억 → 전체 이동 → 무작위 시작 → 혼합 길이를 같은 데이터·예산으로 5시드 진행했다. 새 S9 성공률 평균은 Fly 73.0%, 재배선 70.6%였고 짝별 차이의 방향이 섞였다. 재배선 404는 초기 기억 학습부터 실패했다. 이번 반복에서도 실제 연결 배치의 일관된 우위는 확인되지 않았다. [단계·초기값 감사 원자료](../previous_research/data/minigrid_matched_topology/summary.json)에 시드별 결과가 있다.

### 미사용 길이의 단서·목표 관계 선택

[관계 일반화 후속 보고서](../previous_research/reports/minigrid/minigrid_relation_generalization_results.txt)에 S9·S15 길이 보류, 단서와 목표 물체를 각각 바꾸는 네 조합 평가, 추가 학습 대조, 분리된 시각 판독 정책을 기록했다. RNN 401에 같은 양의 교사 경로를 추가한 네 조건은 S9·S15의 네 조합 동시 성공률이 모두 0%였다. 가변 복도 학습은 정상 성공률은 높였지만 관계 선택을 안정화하지 못했다.

부분관측 화면에서 단서 종류·갈림길·위쪽 물체를 학습해 판독하고, 첫 단서를 별도 메모리에 저장한 뒤 명시적인 일치 규칙으로 회전하는 정책을 만들었다. 독립 새 지형에서 RNN 401·402·403은 S9·S15·S19의 네 조합을 모두 해결했다. RNN 401은 길이마다 256개 기본 지형, 다른 두 시드는 128개씩 평가했다. 같은 시각 모듈을 Fly 405와 재배선 Fly 401에 붙인 S9 평가도 128개 기본 지형의 네 조합을 모두 해결했다. 이는 과제에 맞춘 기억·비교 구조의 성과이며 원래 순환망이나 실제 연결의 우위를 의미하지 않는다. RNN 401의 기존 은닉 상태를 선형 판독하면 S9 단서 종류 정확도는 28.9%였다.

### 학습 가능한 기억·비교 모듈

[학습형 관계 모듈 보고서](../previous_research/reports/minigrid/minigrid_learned_relation_results.txt)에 단서 기억을 Fly·재배선 Fly·RNN·GRU의 순환 상태로, 마지막 선택을 학습 가능한 비교기로 바꾼 5시드 결과를 기록했다. 같은 시각 판독기·이동 정책과 새 평가 지형에서 S9의 네 조합 동시 성공률은 네 구조 모두 평균 97.7~100%였다. S19는 실제 Fly / 재배선 Fly / RNN / GRU가 81.1 / 78.8 / 50.3 / 66.9%였다. 단서 입력을 가리거나 선택 직전 기억을 초기화한 대조는 네 조합 성공률 0%였다. 실제 Fly와 재배선 Fly의 짝 차이는 시드별 방향이 섞였다. 시각 판독기와 비교기는 별도 지도학습을 사용했고 분기 개입도 코드로 정하므로 보상만으로 학습한 단일 정책의 결과가 아니다. 다음 우선순위는 실제 주행 경로에서의 기억 유지와 분기 오검출 개선, 그리고 여러 재배선 그래프 비교다.

### 실제 주행 학습·설정 조정·다중 재배선 비교

[세 단계 후속 보고서](../previous_research/reports/minigrid/minigrid_relation_followup_results.txt)에 실제 주행 경로 학습, 검증 길이의 학습률 조정, 새 5시드와 재배선 그래프 5개 비교를 순서대로 기록했다. S7/S11/S13과 S17Random에서 S9/S15/S19의 갈림길 위치를 제외한 훈련 경로 2,048개를 수집했다. 판독기의 검증 분기 오검출은 56개에서 0개로 줄었고, 훈련 길이 검증으로 Fly·재배선 Fly·GRU의 학습률 0.003, RNN의 0.0003을 선택했다.

독립 새 자율 평가에서 실제 Fly 5개와 재배선 25개는 S9/S15/S19 네 조합을 모두 성공했다. RNN·GRU는 S9/S15에서 100%, S19 평균은 86.25/92.97%였다. 모든 실행에서 조기 분기 개입은 0%였다. 실제 연결과 재배선의 짝 차이는 모두 0이었다. 추가로 **처음부터 간선을 0으로 고정한 누설 기억도 5시드 모두 세 길이에서 100%**였으므로, 현재 이진 기억 과제의 최종 성능에는 학습 가능한 뉴런 간 연결이 필수이지 않았다. 다음 연결 구조 연구는 간선 없는 누설 기억을 기준 모델에 포함하고 방해 단서나 여러 단서의 구분·보관을 요구하는 과제로 확장하는 것이 타당하다. 오목과 5,000개 확장은 계속 보류한다.

## 이전 연구 재실행

아래 명령은 `Set-Location previous_research` 이후 실행한다. 원래 상대 경로는 이 폴더 기준이며, 가상환경은 상위 폴더의 `.venv`를 사용한다.

완료한 다중 단서 연구는 아래 명령으로 저장된 결과와 체크포인트를 재사용하며 이어서 실행한다. `--from-stage load` 또는 `visual`로 해당 단계부터 시작할 수 있다. 결과는 `data/multi_cue_followup/`에 저장한다. 다른 조건의 새 연구는 별도 프로토콜과 출력 폴더를 사용한다.

```powershell
..\.venv\Scripts\python.exe run_multi_cue_followup.py
..\.venv\Scripts\python.exe summarize_multi_cue_followup.py
```

이전 MiniGrid 관계 학습 후속 연구는 다음 순서로 실행한다. `controls`와 `leaky`는 기억·간선 의존성을 해석하기 위한 보조 대조다. `evaluate`는 저장된 실행을 이어서 처리하므로 새 조건의 반복은 별도 출력 폴더를 사용한다.

```powershell
..\.venv\Scripts\python.exe minigrid_relation_followup.py prepare
..\.venv\Scripts\python.exe minigrid_relation_followup.py tune
..\.venv\Scripts\python.exe minigrid_relation_followup.py run
..\.venv\Scripts\python.exe minigrid_relation_followup.py evaluate
..\.venv\Scripts\python.exe minigrid_relation_followup.py controls
..\.venv\Scripts\python.exe minigrid_relation_followup.py leaky
..\.venv\Scripts\python.exe summarize_minigrid_relation_followup.py
```

Python 3.12 가상환경에서:

```powershell
..\.venv\Scripts\python.exe -m pip install -r requirements-phase0.txt
..\.venv\Scripts\python.exe -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
..\.venv\Scripts\python.exe build_subgraphs.py
..\.venv\Scripts\python.exe verify_subgraphs.py
..\.venv\Scripts\python.exe probe_fly_memory.py --batch 32 --steps 64 --output data/phase0_probe_1000_2000_b32_t64.json
..\.venv\Scripts\python.exe train_memory_sanity.py --graph data/subgraphs/malecns_cx_1000.npz
..\.venv\Scripts\python.exe train_memory_sanity.py --graph data/subgraphs/malecns_cx_2000.npz
..\.venv\Scripts\python.exe check_memory_checkpoint.py --graph data/subgraphs/malecns_cx_2000.npz --checkpoint data/sanity/malecns_cx_2000_delay16_seed42.pt
```

원본 Feather 파일 2개는 [공식 다운로드 페이지](https://male-cns.janelia.org/download/)의 `Annotations` 및 `Connectivity` 탭에서 `data/` 폴더에 받을 수 있다. RTX 50 계열의 CUDA용 PyTorch 설치는 [PyTorch 공식 설치 안내](https://pytorch.org/get-started/previous-versions/)를 참조한다.

카드 게임을 재실행하려면 먼저 `rewire_subgraphs.py`로 차수 보존 대조 그래프를 만든 뒤, 예를 들어 `pilot_card_game.py --stage 1 --blank-steps 4 --model fly --neurons 1000 --seed 42`를 실행한다. 지연 일반화는 `evaluate_card_delays.py`로 재평가한다. 카드 후속 검증 순서는 [후속 검증 보고서](../previous_research/reports/card_game/card_followup_results.txt)에 있다. MiniGrid 첫 파일럿은 [기존 결과 보고서](../previous_research/reports/minigrid/minigrid_pilot_results.txt), 고정 시작 기억 진단은 [S11 후속 보고서](../previous_research/reports/minigrid/minigrid_memory_followup_results.txt), 무작위 시작과 길이 실험은 [무작위 시작 보고서](../previous_research/reports/minigrid/minigrid_random_start_results.txt), S9 실패 분해와 동일 단계 비교는 [세 단계 결과 보고서](../previous_research/reports/minigrid/minigrid_s9_three_step_results.txt), 관계 일반화 실험은 [관계 일반화 후속 보고서](../previous_research/reports/minigrid/minigrid_relation_generalization_results.txt), 학습형 기억·비교 탐색은 [관계 모듈 보고서](../previous_research/reports/minigrid/minigrid_learned_relation_results.txt), 최신 실제 주행·다중 재배선 결과는 [세 단계 후속 보고서](../previous_research/reports/minigrid/minigrid_relation_followup_results.txt)에 있다.

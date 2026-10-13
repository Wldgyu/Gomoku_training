# 최신 재개 상태 (2026-10-13)

이번 요청은 후보 3→6→12 비교 후 고유 장기 출발 판 30개 확보까지다. 모델 추가 학습은 아직 시작하지 않았다.

- 비교 완료: 동일 737개 판, 첫 후보만 3/6/12, 이후 후보3, 총2,000노드. 확인 계획54/54/60.
- 최신 수집 수치는 bank_summary.json, 실행 상태는 coverage_progress.json을 읽는다.
- 실행 중에도 경기 단위 JSON을 저장하고 16경기 단위로 학습 자료를 갱신한다.
- 원본 실제 기보의 복구 접두 판은 visits/에 저장했다. 기존 인덱스를 유지했다.
- 프로세스가 이미 실행 중이면 중복 실행하지 않는다.

## 중단과 재개

STOP 파일을 만들면 현재 작업을 저장하고 중단한다. 중단 후 학습 자료와 보고서를 갱신한다:

```powershell
$env:PYTHONPATH='.'
.\.venv\Scripts\python.exe gomoku/runs/candidate_width_nested_20261013/build_bank.py
.\.venv\Scripts\python.exe gomoku/runs/candidate_width_nested_20261013/update_report.py
```

재개할 때 이 디렉터리의 STOP 파일이 있다면 해당 파일만 제거하고 다음을 실행한다:

```powershell
$env:PYTHONPATH='.'
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
.\.venv\Scripts\python.exe gomoku/runs/candidate_width_nested_20261013/run_coverage.py
```

최신 사용자 요청에 따라 잔여 사용량이10% 미만이면 저장하고 마무리한다. usedPercent는 사용률이고 잔여는100-usedPercent다. 과거의 '한도 무시' 지침은 최신10% 지침으로 대체됐다.

완료된 비교를 재실행할 필요는 없다. 비교 당시 교사 코드는 source_comparison/에 보존했고 이후 증명 검증과 기록 변환을 최적화했다. 현재 교사 코드의 해시가 비교 당시 해시와 다른 것은 의도된 것이다.

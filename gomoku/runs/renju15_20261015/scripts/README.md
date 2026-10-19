# 실행 스크립트 (기록용)

v7·v8과 15×15 통제 비교(`ctl_*`)를 만들 때 실제로 쓴 PowerShell 스크립트다. 학습 가중치(.pt)와 Rapfi 실행 파일은 공개 사본에 없어서 그대로는 다시 실행할 수 없다. 어떤 자료·학습률·시드·평가 순서를 썼는지 확인하는 기록으로 쓴다.

| 파일 | 내용 |
|---|---|
| `run_v7_data.ps1` · `run_v7_train.ps1` · `run_v7_eval.ps1` | v7: 깊은 연쇄·금수 자료 생성, v6a에서 이어 학습(시드 7401·7402), 평가 |
| `run_v8_data.ps1` · `run_v8_train.ps1` · `run_v8_eval.ps1` | v8: 사람 기보 재라벨, Rapfi DAgger, v7에서 이어 학습(시드 8401·8402), 평가 |
| `run_ctl_train.ps1` | 통제 비교: 재배열 Fly·실제 Fly·RNN을 같은 자료·12,000회·lr 1e-3·시드 9101/9102로 처음부터 학습 |
| `run_ctl_lane1.ps1` · `run_ctl_lane2.ps1` | 학습 후반부를 GPU 두 줄로 나눠 돌릴 때 쓴 보조 스크립트(총 시간은 줄지 않았다) |
| `run_ctl_eval.ps1` · `eval_suites.py` | 고정 시험, 약한 Rapfi 720국, 리그 평가 |

첫 줄의 `Set-Location`은 공개용으로 저장소 최상위를 가리키게 고쳤다. 로그는 `gomoku\logs\`에 쓰므로 먼저 폴더를 만들어야 한다.

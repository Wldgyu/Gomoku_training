# 이전 연구 자료

카드게임·방/복도·다중 단서 기억 실험의 코드와 보고서입니다.

## 폴더

| 경로 | 내용 |
|---|---|
| [code/](code/) | 기존 Python 코드와 실행 스크립트 |
| [reports/](reports/) | 주제별 결과 보고서 |
| [games/card_memory/](games/card_memory/) | 학습 규칙의 원본 카드 기억 웹게임 |
| [data/](data/) | 원자료·학습 기록·체크포인트·당시 소스 |
| [requirements/](requirements/) | 단계별 의존성 목록 |
| [manifests/](manifests/) | 이동 목록·해시·경로 수정 전 코드 |

## 보고서

### 카드게임

- [card_followup_results.txt](reports/card_game/card_followup_results.txt)
- [card_game_delay_stage2_report.txt](reports/card_game/card_game_delay_stage2_report.txt)
- [card_game_pilot_report.txt](reports/card_game/card_game_pilot_report.txt)
- [rnn_recovery_results.txt](reports/card_game/rnn_recovery_results.txt)

### 방·복도 / MiniGrid

- [minigrid_learned_relation_results.txt](reports/minigrid/minigrid_learned_relation_results.txt)
- [minigrid_memory_followup_results.txt](reports/minigrid/minigrid_memory_followup_results.txt)
- [minigrid_pilot_results.txt](reports/minigrid/minigrid_pilot_results.txt)
- [minigrid_random_start_results.txt](reports/minigrid/minigrid_random_start_results.txt)
- [minigrid_relation_followup_results.txt](reports/minigrid/minigrid_relation_followup_results.txt)
- [minigrid_relation_generalization_results.txt](reports/minigrid/minigrid_relation_generalization_results.txt)
- [minigrid_s9_three_step_results.txt](reports/minigrid/minigrid_s9_three_step_results.txt)

### 다중 단서 기억

- [multi_cue_audit_results.txt](reports/memory/multi_cue_audit_results.txt)
- [multi_cue_aux_generalization_results.txt](reports/memory/multi_cue_aux_generalization_results.txt)
- [multi_cue_binding_results.txt](reports/memory/multi_cue_binding_results.txt)
- [multi_cue_followup_results.txt](reports/memory/multi_cue_followup_results.txt)
- [multi_cue_recovery_results.txt](reports/memory/multi_cue_recovery_results.txt)

## 기존 코드 실행

기초 연구 질문의 종합 기록은 [RQ1~RQ4 중간 결과](reports/foundation/core_research_questions_report.txt)에 있습니다.

작업 폴더는 previous_research로 둡니다.

~~~powershell
Set-Location previous_research
..\.venv\Scripts\python.exe code\pilot_card_game.py --help
..\.venv\Scripts\python.exe -m unittest discover -s code -p 'test_*.py'
~~~

현재 경로는 수정했으며 당시 원본은 데이터의 source 폴더와 manifests/source_before_layout에 보존했습니다.

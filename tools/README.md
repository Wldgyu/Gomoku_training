# 정리·재현 보조 도구

연구의 실제 환경·모델·학습 코드는 ../gomoku/와 ../previous_research/code/에 있습니다.

| 파일 | 역할 |
|---|---|
| organize_archive.py | 이전 연구의 1회 이동·해시 기록·경로 수정 |
| merge_parallel.py | 완료된 독립 학습을 공통 실행 폴더로 합치기 |
| finalize_layout.py | 완료 검증 후 gomoku9→gomoku 이동·보고서 사본 구성 |
| complete_experiment.py | 고정 평가·예산 감사·보고서 생성 |
| full_history_control.py | 원 예산의 기보 단독 추가 대조 |
| write_overview.py | 완료된 수치로 최상위 README 생성 |
| prepare_github.py | 업로드용 사본·포함/제외 파일 목록·해시 생성 |
| validate_publication.py | 사본 해시·큰 파일 제외·문서 링크 검증 |

이동 도구는 이번 정리의 기록을 재현하기 위한 것으로 정리된 폴더에서 다시 실행할 필요는 없습니다.
이미 완료된 이동이나 기존 업로드 사본을 덮어쓰지 않도록 검사합니다.
원 자료와 가중치를 로컬에 남기고 실제 GitHub 업로드는 수행하지 않습니다.

# GitHub 업로드 안내

## 먼저 볼 파일

최상위 README → 오목 최신 결과 → previous_research의 주제별 보고서 순서로 읽으면 됩니다.
연구 기획서와 계획·프로토콜은 최상위에 유지했습니다.

## 업로드할 폴더

최종 정리 후 생성하는 github_upload 폴더의 **내용**을 저장소 최상위에 올리세요.
원 작업 폴더 전체를 브라우저로 끌어 넣으면 대형 원자료와 가상환경도 포함될 수 있습니다.
업로드용 사본은 소스와 해시를 대조하며 PUBLICATION_MANIFEST.json에 포함·제외 목록을 기록합니다.
이 작업에서 원격 저장소 생성이나 push는 수행하지 않습니다.

포함: 오목 코드·보고서·실제 기보·평가 JSON·전술 데이터, 이전 연구 코드·보고서·작은 분석 기록,
Fly 연결 그래프, 원본 카드 웹게임, 실행 계획과 의존성 목록.

로컬 보존: .venv, 대형 연결 원자료(.feather), 학습 가중치(.pt), 캐시·임시 파일·실행 로그.
업로드 사본의 과거 보고서는 당시 경로와 가중치를 언급할 수 있지만 제외 파일 자체는 사본에 없습니다.
기보 HTML과 수치 보고서는 모델 가중치 없이 읽을 수 있습니다.
학습을 재현하면 새로운 가중치가 생성됩니다. 과거 체크포인트 평가에는 별도 가중치가 필요합니다.

## 대형 파일을 함께 공개하려면

일반 GitHub Git은 100MiB를 넘는 파일을 제한하고 브라우저 업로드는 파일당25MiB까지입니다.
큰 체크포인트와 원자료는 Git LFS 또는 Release 첨부를 별도로 검토하세요.
현재 .gitignore는 이런 파일을 제외합니다. LFS를 쓰려면 먼저 추적 설정을 하고 필요한 파일만 추가하세요.
[GitHub 공식 대형 파일 안내](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github),
[Git LFS 안내](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-large-file-storage).

카드게임 원본은 previous_research/games/card_memory/index.html을 브라우저로 열면 됩니다.
원본의 모든 단계가 과거 학습 실험에 포함된 것은 아닙니다. 모델은 보고서에 명시된 단계를 학습했습니다.

## 재현 환경

최상위 requirements.txt는 오목 실행의 핵심 버전입니다.
CUDA용 PyTorch 설치는 사용자의 GPU 환경에 맞는 배포판을 선택하고,
당시 전체 환경 기록은 gomoku/environment_versions.txt를 참고하세요.
이전 과제의 추가 의존성은 previous_research/requirements/에 따로 있습니다.
기록된 설치 환경과 업로드된 코드의 실행 명령은 최상위 README에 있습니다.

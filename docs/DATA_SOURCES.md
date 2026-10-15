# 초파리 연결 데이터 출처와 사용 방식

이 프로젝트의 Fly 모델은 **MaleCNS v1.0의 뉴런 주석과 시냅스 연결 데이터**를 바탕으로 구성했습니다.

- 원자료 제공: FlyEM (HHMI Janelia), University of Cambridge Department of Zoology, MRC Laboratory of Molecular Biology, Google Research의 공동 MaleCNS 프로젝트.
- 공식 프로젝트: https://male-cns.janelia.org/
- 데이터 다운로드: https://male-cns.janelia.org/download/
- 데이터 라이선스: [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/). 이 표기는 MaleCNS 원자료와 이를 가공한 연결 그래프에 관한 것입니다.
- 확인일: 2026-10-15.

사용한 원본 파일은 `body-annotations-male-cns-v1.0-minconf-0.5.feather`와 `connectome-weights-male-cns-v1.0-minconf-0.5.feather`입니다. 원본 파일은 공개 사본에서 제외하고, 가공한 부분 그래프와 출처·해시 메타데이터를 `previous_research/data/subgraphs/`에 포함했습니다.

## 이 프로젝트에서 가공한 내용

뉴런 주석을 이용해 부분 집합을 선택하고, 선택한 뉴런 사이의 방향성 연결을 추출했습니다. 최신 15×15 렌주의 Fly2950은 `cb_intrinsic` / `CX` 후보 2,950개를 사용하며, 자기 연결을 제거하고 최소 시냅스 수 3을 적용한 방향 연결 207,827개로 구성했습니다. 이전 실험에는 더 작은 부분 그래프와 비교용으로 연결을 재배열한 그래프도 있습니다. 가공 조건과 원본 SHA-256은 각 그래프의 JSON 메타데이터에 기록되어 있습니다.

이 연결 구조를 인공 순환 신경망에 적용하고, 오목·렌주 과제로 모델 파라미터를 별도로 학습했습니다. 살아 있는 뉴런이나 초파리의 학습된 오목 능력을 가져온 것이 아니며, 초파리 전체 뇌의 생리적 동작을 복제했다고 주장하지 않습니다.

원자료와 가공 그래프를 재사용할 때에는 위 제공자·출처·CC BY 4.0 링크를 유지하고 변경 내용을 표시해 주세요. 본 프로젝트의 가공 및 오목·렌주 실험은 원자료 제공자의 공식 연구나 보증을 의미하지 않습니다.

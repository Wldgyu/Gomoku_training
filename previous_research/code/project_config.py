"""간단한 게임 파일럿에서 사용할 공통 그래프 규모와 방향 규칙."""

# 이 두 크기로 먼저 같은 게임을 실행하고, 결과를 보고 5,000개 확장 여부를 정한다.
PILOT_NEURON_COUNTS = (1000, 2000)

# 원본의 시냅스 전달 방향이다. 모델의 행렬은 W[post, pre]로 구성한다.
EDGE_SOURCE_FIELD = "pre_index"
EDGE_TARGET_FIELD = "post_index"


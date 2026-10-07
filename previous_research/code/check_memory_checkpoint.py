"""저장된 기억 진단 모델을 다시 불러와 새 문제와 신호 제거 대조군을 평가한다."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from train_memory_sanity import ConnectomeMemoryNet, file_hash, make_batch


def main() -> None:
    """체크포인트의 그래프 해시를 확인한 뒤 재학습 없이 정확도를 측정한다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--delay", type=int, default=16)
    parser.add_argument("--examples", type=int, default=2048)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.delay < 1 or args.examples < 1:
        parser.error("delay and examples must be positive")

    device = torch.device(args.device)
    saved = torch.load(args.checkpoint, map_location=device, weights_only=True)
    # 다른 연결 그래프로 모델을 평가하면 결과를 해석할 수 없으므로 중단한다.
    if saved["graph_sha256"] != file_hash(args.graph):
        raise ValueError("checkpoint was trained on a different graph file")
    model = ConnectomeMemoryNet(args.graph).to(device)
    model.load_state_dict(saved["model_state"])
    model.eval()
    rng = torch.Generator(device=device).manual_seed(99173)
    # 학습에 사용하지 않은 새 난수 예시를 생성한다.
    x, labels = make_batch(args.examples, args.delay, device, rng)
    with torch.no_grad():
        accuracy = (model(x).argmax(dim=1) == labels).float().mean().item()
        x[:, 0, 0] = 0  # 정답 정보를 없애 누출이나 편향만으로 맞히는지 확인한다.
        no_cue_accuracy = (model(x).argmax(dim=1) == labels).float().mean().item()
    result = {
        "checkpoint": str(args.checkpoint),
        "graph_hash_matches": True,
        "training_delay": saved["delay"],
        "evaluation_delay": args.delay,
        "fresh_examples": args.examples,
        "accuracy_with_cue": accuracy,
        "accuracy_without_cue": no_cue_accuracy,
        "chance_accuracy": 0.5,
    }
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()

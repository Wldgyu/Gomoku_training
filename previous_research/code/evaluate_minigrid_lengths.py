"""S11에서 학습한 체크포인트를 S7·S13의 새 에피소드에 재학습 없이 평가한다."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from minigrid_memory_pilot import GRAPH_DIR, MemoryPolicy, evaluate
from run_minigrid_matrix import MODELS, OUTPUT, SEEDS, SIZES

RESULT_PATH = OUTPUT / "length_generalization.json"


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for seed in SEEDS:
        for size in SIZES:
            for kind in MODELS:
                name = f"memory_s11_{kind}_{size}_seed{seed}"
                result_path = OUTPUT / f"{name}.json"
                checkpoint_path = OUTPUT / f"{name}.pt"
                if not result_path.exists() or not checkpoint_path.exists():
                    raise FileNotFoundError(f"학습 체크포인트가 없습니다: {name}")
                result = json.loads(result_path.read_text(encoding="utf-8"))
                graph = GRAPH_DIR / f"malecns_cx_{size}"
                graph_path = Path(f"{graph}{'_rewired_seed0' if kind == 'rewired' else ''}.npz")
                model = MemoryPolicy(
                    "fly" if kind in ("fly", "rewired") else kind,
                    size,
                    graph_path if kind in ("fly", "rewired") else None,
                    result["hidden_size"] if kind in ("rnn", "gru") else None,
                ).to(device)
                saved = torch.load(checkpoint_path, map_location=device, weights_only=False)
                model.load_state_dict(saved["model_state"])
                for environment_size in (7, 13):
                    evaluation_seed = 900_000 + seed * 1_000
                    for mask_cue in (False, True):
                        measured = evaluate(
                            model, size=environment_size, max_steps=100,
                            episodes=64, seed=evaluation_seed, device=device,
                            mask_cue=mask_cue, start_mode="fixed_cue",
                        )
                        row = {
                            "trained_environment": "MiniGrid-MemoryS11-v0",
                            "evaluated_environment": f"MiniGrid-MemoryS{environment_size}-v0",
                            "model": kind, "neurons_reference": size,
                            "seed": seed, "mask_cue": mask_cue, **measured,
                        }
                        rows.append(row)
                        print(f"{kind} n={size} seed={seed} S{environment_size} "
                              f"mask={mask_cue} success={measured['success_rate']:.3f}", flush=True)
    RESULT_PATH.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {RESULT_PATH}", flush=True)


if __name__ == "__main__":
    main()

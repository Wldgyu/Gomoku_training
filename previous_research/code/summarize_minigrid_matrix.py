"""MiniGrid 파일럿의 시드별 원자료와 평균·표본 표준편차를 요약한다."""

from __future__ import annotations

import json
import statistics
from collections import defaultdict

from run_minigrid_matrix import MODELS, OUTPUT, SEEDS, SIZES


def aggregate_rows(rows: list[dict], key: str) -> dict:
    values = [row[key] for row in rows]
    return {"mean": statistics.mean(values),
            "sample_std": statistics.stdev(values), "by_seed": values}


def main() -> None:
    rows, summary = [], []
    for size in SIZES:
        for kind in MODELS:
            subset = []
            for seed in SEEDS:
                path = OUTPUT / f"memory_s11_{kind}_{size}_seed{seed}.json"
                result = json.loads(path.read_text(encoding="utf-8"))
                row = {
                    "model": kind, "neurons_reference": size, "seed": seed,
                    "parameters": result["parameters"],
                    "success": result["final_fresh_evaluation"]["success_rate"],
                    "mean_reward": result["final_fresh_evaluation"]["mean_reward"],
                    "cue_masked_success": result["final_fresh_cue_masked_evaluation"]["success_rate"],
                    "default_start_success": result["final_fresh_default_start_evaluation"]["success_rate"],
                    "peak_vram_gib": result["peak_vram_bytes_pytorch"] / 2**30,
                    "duration_seconds": result["duration_seconds"],
                    "learning_curve": [
                        {"interaction_steps": point["interaction_steps"],
                         "success_rate": point["success_rate"],
                         "mean_reward": point["mean_reward"]}
                        for point in result["history"]
                    ],
                }
                rows.append(row)
                subset.append(row)
            summary.append({
                "model": kind, "neurons_reference": size, "seeds": list(SEEDS),
                "success": aggregate_rows(subset, "success"),
                "mean_reward": aggregate_rows(subset, "mean_reward"),
                "cue_masked_success": aggregate_rows(subset, "cue_masked_success"),
                "default_start_success": aggregate_rows(subset, "default_start_success"),
                "peak_vram_gib": aggregate_rows(subset, "peak_vram_gib"),
                "duration_seconds": aggregate_rows(subset, "duration_seconds"),
            })
    length_rows = json.loads((OUTPUT / "length_generalization.json").read_text(encoding="utf-8"))
    length_groups = defaultdict(list)
    for row in length_rows:
        key = (row["neurons_reference"], row["model"],
               row["evaluated_environment"], row["mask_cue"])
        length_groups[key].append(row["success_rate"])
    length_summary = [
        {"neurons_reference": key[0], "model": key[1],
         "evaluated_environment": key[2], "mask_cue": key[3],
         "success_mean": statistics.mean(values),
         "success_sample_std": statistics.stdev(values), "by_seed": values}
        for key, values in sorted(length_groups.items())
    ]
    output = OUTPUT / "matrix_summary.json"
    output.write_text(json.dumps({"rows": rows, "summary": summary,
                                  "length_generalization": length_summary},
                                 ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    for item in summary:
        print(item["neurons_reference"], item["model"],
              f"success={item['success']['mean']:.3f}±{item['success']['sample_std']:.3f}",
              f"masked={item['cue_masked_success']['mean']:.3f}",
              f"vram={item['peak_vram_gib']['mean']:.3f} GiB", flush=True)
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()

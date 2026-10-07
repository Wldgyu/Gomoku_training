"""Choose independent-model batching by speed, without inspecting task scores."""

import time

import torch

import multi_cue_followup as base
from multi_cue_memory import TaskConfig
from multi_cue_recovery import OUT


def main():
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda")
    results = []
    for seeds in ((2201,), tuple(range(2201, 2206))):
        runs = [
            r
            for s in seeds
            for r in [
                base.Run("fly", s),
                *[base.Run("rewired", s, g) for g in range(5)],
            ]
        ]
        models = [base.model_for(r, device) for r in runs]
        p, b = base.bundle(models)
        del models
        batches = {
            s: base.make_fast(TaskConfig(2, 2, 16, 2), 128, 900000 + s) for s in seeds
        }
        x = torch.stack([batches[r.seed]["inputs"] for r in runs]).to(device)
        y = torch.stack([batches[r.seed]["answers"] for r in runs]).to(device)
        opt = torch.optim.AdamW([v for v in p.values() if v.requires_grad], lr=0.003)
        torch.cuda.reset_peak_memory_stats()
        replay = base.capture_backward(p, b, "fly", x, y)
        for _ in range(3):
            replay(x, y)
            base.clip_independent(p)
            opt.step()
        torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(20):
            replay(x, y)
            base.clip_independent(p)
            opt.step()
        torch.cuda.synchronize()
        seconds = (time.perf_counter() - started) / 20
        results.append(
            {
                "models": len(runs),
                "seconds_per_update": seconds,
                "seconds_per_model_update": seconds / len(runs),
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            }
        )
        del replay, p, b, opt, x, y
        torch.cuda.empty_cache()
    best = min(results, key=lambda r: r["seconds_per_model_update"])
    record = {
        "results": results,
        "selected_models_per_group": best["models"],
        "selection_uses_task_accuracy": False,
        "tf32": False,
    }
    base.atomic_json(OUT / "training_batch_probe.json", record)
    print(record)


if __name__ == "__main__":
    main()

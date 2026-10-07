"""Exploratory read-only inspection of the already completed stage-2 classifiers."""

import json

import torch

from multi_cue_followup import OUT, atomic_json, load_model
from multi_cue_memory import TaskConfig, make_batch


def main():
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda")
    rows = []
    for path in sorted((OUT / "diagnosis").glob("*.json")):
        record = json.loads(path.read_text())
        result = record["run"]
        model, _ = load_model(result["checkpoint"], device)
        active = torch.zeros(128, dtype=torch.long)
        predictions = []

        def hook(_module, _args, output, active=active):
            active.add_((output > 0).sum(0).cpu())

        handle = model.head[0].register_forward_hook(hook)
        data = make_batch(TaskConfig(2, 2, 4, 2), 4096, 3_000_000_000 + result["seed"])
        with torch.no_grad():
            for x in data["inputs"].split(512):
                predictions.append(model(x.to(device)).argmax(-1).cpu())
        handle.remove()
        pred = torch.cat(predictions)
        row = {
            "name": result["name"],
            "kind": result["kind"],
            "seed": result["seed"],
            "frozen": result["frozen"],
            "relu_units_never_active": int(active.eq(0).sum()),
            "relu_units_total": 128,
            "mean_active_fraction": float(active.float().mean() / 4096),
            "predicted_class_counts": pred.bincount(minlength=3).tolist(),
            "accuracy": float(pred.eq(data["answers"]).float().mean()),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
    atomic_json(OUT / "head_inspection.json", rows)


if __name__ == "__main__":
    main()

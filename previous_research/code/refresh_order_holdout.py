"""Break down strict held-out cue-order errors by queried slot and color pair."""

from __future__ import annotations

import json

import torch

from multi_cue_binding import BindMemory
from multi_cue_order_holdout import OUT, constrained_batch


@torch.no_grad()
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for seed in (3201, 3202, 3203):
        data = constrained_batch(4096, 9_000_000_000 + seed, False)
        colors = data["inputs"][:, :2, :4].argmax(-1)
        query = data["inputs"][:, -1, :4].argmax(-1)
        slot = (colors[:, 1] == query).long()
        for path in sorted(OUT.glob(f"*_seed{seed}.pt")):
            saved = torch.load(path, map_location="cpu", weights_only=True)
            model = BindMemory(saved["kind"], saved["hidden"], "aux", seed % 5).to(device)
            model.load_state_dict(saved["state"])
            model.eval()
            predictions = []
            for start in range(0, len(data["answers"]), 512):
                predictions.append(model(data["inputs"][start:start + 512].to(device))[0]
                                   .argmax(-1).cpu())
            correct = torch.cat(predictions) == data["answers"]
            result_path = path.with_suffix(".json")
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if abs(float(correct.float().mean()) - result["results"]["heldout_order"]) > 1e-9:
                raise RuntimeError(f"mismatch: {path.name}")
            result["heldout_by_slot"] = {
                str(i): {"accuracy": float(correct[slot == i].float().mean()),
                         "count": int((slot == i).sum())} for i in (0, 1)}
            result["heldout_by_pair"] = {}
            for pair in sorted({tuple(x) for x in colors.tolist()}):
                mask = (colors[:, 0] == pair[0]) & (colors[:, 1] == pair[1])
                result["heldout_by_pair"][f"{pair[0]}-{pair[1]}"] = {
                    "accuracy": float(correct[mask].float().mean()), "count": int(mask.sum())}
            result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            print(seed, saved["kind"], result["heldout_by_slot"], flush=True)


if __name__ == "__main__":
    main()

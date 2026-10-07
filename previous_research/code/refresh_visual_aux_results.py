"""Add frozen auxiliary cue accuracy to saved room-corridor results."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from multi_cue_visual_aux import OUT, VisualAuxModel, dataset, fixed_path_eval


@torch.no_grad()
def explicit_aux_selection(model, data, device):
    """Diagnostic: read the visible final door color code and route aux logits."""
    correct = 0
    for start in range(0, len(data["answers"]), 128):
        ids = slice(start, start + 128)
        images = data["images"][ids]
        lengths = data["lengths"][ids]
        row = torch.arange(len(lengths))
        front = images[row, lengths - 1, 3, 5]
        if not torch.all(front[:, 0] == 4):
            raise RuntimeError("expected the query door in front at the final frame")
        code = front[:, 1].long()
        lookup = torch.tensor([0, 1, 2, -1, 3, -1])
        query = lookup[code]
        if not torch.all(query >= 0):
            raise RuntimeError("unknown MiniGrid query color")
        _, aux, _ = model(images.to(device), lengths.to(device))
        pred = aux[torch.arange(len(query), device=device), query.to(device)].argmax(-1)
        correct += int((pred.cpu() == data["answers"][ids]).sum())
    return correct / len(data["answers"])


def main():
    torch.set_num_threads(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tests = {"trained_lengths": dataset("test", 2048, (6, 10), 63),
             "unseen_lengths": dataset("unseen", 2048, (8, 14), 64),
             "unseen_wait": dataset("wait", 2048, (8, 14), 65, 0.5)}
    for path in sorted((OUT / "runs").glob("*_aux_seed*.pt")):
        saved = torch.load(path, map_location="cpu", weights_only=True)
        model = VisualAuxModel(saved["kind"], saved["hidden"],
                               saved["seed"], True).to(device)
        model.load_state_dict(saved["state"])
        model.eval()
        json_path = path.with_suffix(".json")
        result = json.loads(json_path.read_text(encoding="utf-8"))
        for name, data in tests.items():
            fresh = fixed_path_eval(model, data, device)
            if abs(fresh["answer_accuracy"] - result["tests"][name]["answer_accuracy"]) > 1e-9:
                raise RuntimeError(f"answer mismatch: {path.name} {name}")
            result["tests"][name]["aux_cue_accuracy"] = fresh["aux_cue_accuracy"]
            result["tests"][name]["aux_visible_door_route_accuracy"] = explicit_aux_selection(
                model, data, device)
        json_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(path.name, {name: result["tests"][name]["aux_cue_accuracy"]
                          for name in tests}, flush=True)


if __name__ == "__main__":
    main()

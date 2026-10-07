"""Visible-attribute probes, fixed-memory readout repair, and observation-summary controls."""

from __future__ import annotations

import argparse
import json

import torch
from minigrid.core.constants import COLOR_TO_IDX, OBJECT_TO_IDX
from torch import nn
from torch.nn import functional as F

from grid_multi_cue import (
    COLOR_NAMES,
    GridMemory,
    GridSpec,
    build_dataset,
    matched_hidden_for,
    one_hot_cells,
)
from multi_cue_followup import atomic_json, atomic_torch
from multi_cue_memory import MemoryModel, parameter_count
from multi_cue_recovery import (
    KINDS,
    OUT,
    final_states,
    fit_head,
    head_predict,
    original_predict,
    prediction_metrics,
)
from multi_cue_visual_followup import data as previous_data

VISUAL = OUT / "visual"
COLOR_IDS = tuple(COLOR_TO_IDX[c] for c in COLOR_NAMES)
OBJECT_IDS = tuple(OBJECT_TO_IDX[k] for k in ("key", "ball", "box"))
AGENT_CELL = 3 * 7 + 6  # MiniGrid's image axes are x,y; agent is at (3,6).


def new_tests():
    tests = {}
    for name, seed, delays, wait in (
        ("trained_delays", 86, (6, 10), 0),
        ("unseen_delays", 87, (8, 14), 0),
        ("unseen_delays_wait", 88, (8, 14), 0.5),
    ):
        path = VISUAL / "cache" / f"{name}_seed{seed}.pt"
        if path.exists():
            data = torch.load(path, weights_only=True)
        else:
            data = build_dataset([GridSpec(2, d, 2, wait) for d in delays], 2048, seed)
            atomic_torch(path, data)
        tests[name] = data
        print(
            json.dumps(
                {
                    "event": "fresh_visual_data",
                    "group": name,
                    "episodes": len(data["answers"]),
                }
            ),
            flush=True,
        )
    return tests


def visible_attributes(images):
    """Only attributes of objects visible in the supplied partial observation."""
    obj, color = images[..., 0], images[..., 1]
    counts = torch.stack(
        [
            (obj.eq(k) & color.eq(c)).sum((-2, -1))
            for c in COLOR_IDS
            for k in OBJECT_IDS
        ],
        -1,
    ).float()
    query = torch.stack(
        [
            (obj.eq(OBJECT_TO_IDX["door"]) & color.eq(c)).any(-1).any(-1)
            for c in COLOR_IDS
        ],
        -1,
    ).float()
    return counts, query


def observation_summary(images):
    counts, query = visible_attributes(images)
    walls = images[..., 0].eq(OBJECT_TO_IDX["wall"]).flatten(-2)
    if walls[..., AGENT_CELL].any():
        raise ValueError("the excluded agent cell unexpectedly contains a wall")
    wall_features = torch.cat(
        (walls[..., :AGENT_CELL], walls[..., AGENT_CELL + 1 :]), -1
    ).float()
    return torch.cat((counts / 3, query, wall_features), -1)


def frame_dataset(dataset, limit, seed):
    valid = torch.arange(dataset["images"].shape[1])[None] < dataset["lengths"][:, None]
    frames = dataset["images"][valid]
    if limit is not None and len(frames) > limit:
        order = torch.randperm(
            len(frames), generator=torch.Generator().manual_seed(seed)
        )[:limit]
        frames = frames[order]
    counts, query = visible_attributes(frames)
    return {"images": frames, "targets": torch.cat((counts.gt(0).float(), query), -1)}


@torch.no_grad()
def encode_frames(model, frames, device):
    return torch.cat(
        [
            model.encoder(one_hot_cells(part.to(device))).cpu()
            for part in frames.split(512)
        ]
    )


def perception_metrics(pred, targets, query_scores=None):
    tp = int((pred & targets).sum())
    fp = int((pred & ~targets).sum())
    fn = int((~pred & targets).sum())
    seen_query = targets[:, 12:].any(-1)
    scores = pred[:, 12:].float() if query_scores is None else query_scores
    return {
        "positive_f1": 2 * tp / max(2 * tp + fp + fn, 1),
        "positive_recall": tp / max(tp + fn, 1),
        "exact_frame_accuracy": float(pred.eq(targets).all(-1).float().mean()),
        "visible_query_color_accuracy": float(
            scores[seen_query]
            .argmax(-1)
            .eq(targets[seen_query, 12:].long().argmax(-1))
            .float()
            .mean()
        ),
        "visible_query_exact_bits": float(
            pred[seen_query, 12:].eq(targets[seen_query, 12:]).all(-1).float().mean()
        ),
        "frames": len(targets),
        "query_visible_frames": int(seen_query.sum()),
        "positive_bits": int(targets.sum()),
    }


def fit_perception(trainx, valx, trainy, valy, seed, path, device):
    if path.exists():
        return torch.load(path, weights_only=True)
    mean, scale = trainx.mean(0), trainx.std(0).clamp(min=1e-4)
    tx, vx = (
        (trainx.to(device) - mean.to(device)) / scale.to(device),
        (valx.to(device) - mean.to(device)) / scale.to(device),
    )
    ty, vy = trainy.to(device), valy.to(device)
    torch.manual_seed(seed + 40000)
    decoder = nn.Linear(tx.shape[1], 16).to(device)
    opt = torch.optim.AdamW(decoder.parameters(), lr=0.003)
    gen = torch.Generator(device=device).manual_seed(seed + 40001)
    best, best_state = float("inf"), None
    for update in range(1, 501):
        ids = torch.randint(len(tx), (256,), generator=gen, device=device)
        loss = F.binary_cross_entropy_with_logits(decoder(tx[ids]), ty[ids])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(decoder.parameters(), 1)
        opt.step()
        if update % 50 == 0:
            with torch.no_grad():
                ce = float(F.binary_cross_entropy_with_logits(decoder(vx), vy))
            if ce < best:
                best = ce
                best_state = {
                    n: p.detach().cpu().clone() for n, p in decoder.state_dict().items()
                }
    saved = {
        "decoder": best_state,
        "mean": mean,
        "scale": scale,
        "validation_bce": best,
        "updates": 500,
        "training_frames": len(tx),
    }
    atomic_torch(path, saved)
    return saved


@torch.no_grad()
def cnn_memory_features(model, dataset, device):
    states = []
    for start in range(0, len(dataset["answers"]), 128):
        images = dataset["images"][start : start + 128].to(device)
        features = model.encoder(one_hot_cells(images.flatten(0, 1))).view(
            len(images), images.shape[1], -1
        )
        states.append(
            final_states(
                model.memory, features, device, dataset["lengths"][start : start + 128]
            )
        )
    final_images = dataset["images"][
        torch.arange(len(dataset["answers"])), dataset["lengths"] - 1
    ]
    _, query = visible_attributes(final_images)
    if not query.sum(-1).eq(1).all():
        raise ValueError("the final observation must contain exactly one query door")
    return {
        "states": torch.cat(states),
        "answers": dataset["answers"],
        "query_color": query.argmax(-1),
        "query_slot": dataset["query_slot"],
    }


def diagnose_cnn(train, val, tests, device):
    choice = json.loads((OUT / "selected_head.json").read_text(encoding="utf-8"))[
        "variant"
    ]
    frames = {
        "train": frame_dataset(train, 8192, 41001),
        "val": frame_dataset(val, 2048, 41002),
        **{
            name: frame_dataset(dataset, None, 41003) for name, dataset in tests.items()
        },
    }
    for kind in KINDS:
        for seed in (1101, 1102, 1103):
            name = f"{kind}_seed{seed}"
            result_path = VISUAL / "diagnosis" / f"{name}.json"
            if result_path.exists():
                continue
            checkpoint = OUT.parent / "multi_cue_followup/visual/main" / f"{name}.pt"
            saved = torch.load(checkpoint, weights_only=True)
            model = GridMemory(
                kind, saved["result"]["hidden"], seed % 5 if kind == "rewired" else 0
            ).to(device)
            model.load_state_dict(saved["model_state"])
            encoded = {
                s: encode_frames(model, d["images"], device) for s, d in frames.items()
            }
            decoder_saved = fit_perception(
                encoded["train"],
                encoded["val"],
                frames["train"]["targets"],
                frames["val"]["targets"],
                seed,
                VISUAL / "perception_heads" / f"{name}.pt",
                device,
            )
            decoder = nn.Linear(64, 16).to(device)
            decoder.load_state_dict(decoder_saved["decoder"])
            perceptions = {}
            with torch.no_grad():
                for s in tests:
                    norm = (
                        encoded[s].to(device) - decoder_saved["mean"].to(device)
                    ) / decoder_saved["scale"].to(device)
                    logits = torch.cat([decoder(x).cpu() for x in norm.split(1024)])
                    perceptions[s] = perception_metrics(
                        logits.ge(0), frames[s]["targets"].bool(), logits[:, 12:]
                    )
            del encoded
            train_features = cnn_memory_features(model, train, device)
            val_features = cnn_memory_features(model, val, device)
            repaired = fit_head(
                train_features,
                val_features,
                choice,
                seed,
                VISUAL / "readout_heads" / f"{name}.pt",
                device,
            )
            evaluations = {}
            for s, dataset in tests.items():
                features = cnn_memory_features(model, dataset, device)
                evaluations[s] = {
                    "original": prediction_metrics(
                        original_predict(model.memory, features, device), features
                    ),
                    "repaired": prediction_metrics(
                        head_predict(repaired, features, device), features
                    ),
                }
            result = {
                "kind": kind,
                "seed": seed,
                "source_checkpoint": str(checkpoint),
                "selected_head": choice,
                "perception": perceptions,
                "perception_metric_version": 2,
                "tests": evaluations,
                "decoder_validation_bce": decoder_saved["validation_bce"],
                "extra_head_training": repaired["result"],
            }
            atomic_json(result_path, result)
            print(
                json.dumps(
                    {
                        "event": "visual_diagnosis",
                        "run": name,
                        "f1": perceptions["trained_delays"]["positive_f1"],
                        "original": evaluations["trained_delays"]["original"][
                            "accuracy"
                        ],
                        "repaired": evaluations["trained_delays"]["repaired"][
                            "accuracy"
                        ],
                    }
                ),
                flush=True,
            )


def summarized_dataset(dataset):
    return {
        **{k: v for k, v in dataset.items() if k != "images"},
        "inputs": observation_summary(dataset["images"]),
    }


@torch.no_grad()
def logits_for(model, dataset, device):
    return torch.cat(
        [
            model(
                dataset["inputs"][i : i + 128].to(device),
                dataset["lengths"][i : i + 128].to(device),
            ).cpu()
            for i in range(0, len(dataset["answers"]), 128)
        ]
    )


def fit_summary(kind, seed, lr, hidden, train, val, epochs, path, device):
    if path.exists():
        saved = torch.load(path, weights_only=True)
        model = MemoryModel(
            kind, hidden, seed % 5 if kind == "rewired" else 0, input_dim=64
        ).to(device)
        model.load_state_dict(saved["model_state"])
        return model, saved["result"]
    torch.manual_seed(seed)
    model = MemoryModel(
        kind, hidden, seed % 5 if kind == "rewired" else 0, input_dim=64
    ).to(device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    gen = torch.Generator().manual_seed(seed + 7)
    best, curve = (-1, float("-inf")), []
    for epoch in range(1, epochs + 1):
        order = torch.randperm(len(train["answers"]), generator=gen)
        for ids in order.split(128):
            loss = F.cross_entropy(
                model(
                    train["inputs"][ids].to(device), train["lengths"][ids].to(device)
                ),
                train["answers"][ids].to(device),
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1)
            opt.step()
        logits = logits_for(model, val, device)
        acc = float(logits.argmax(-1).eq(val["answers"]).float().mean())
        ce = float(F.cross_entropy(logits, val["answers"]))
        curve.append({"epoch": epoch, "validation": acc, "validation_loss": ce})
        if (acc, -ce) > best:
            best = (acc, -ce)
            best_state = {
                n: p.detach().cpu().clone() for n, p in model.state_dict().items()
            }
            best_epoch = epoch
    model.load_state_dict(best_state)
    result = {
        "kind": kind,
        "seed": seed,
        "hidden": hidden,
        "learning_rate": lr,
        "epochs": epochs,
        "updates": epochs * ((len(train["answers"]) + 127) // 128),
        "parameters": parameter_count(model),
        "training_episodes": len(train["answers"]),
        "best_validation": best[0],
        "best_validation_loss": -best[1],
        "best_epoch": best_epoch,
        "curve": curve,
    }
    atomic_torch(path, {"model_state": best_state, "result": result})
    return model, result


def summary_control(train, val, tests, device):
    target = parameter_count(MemoryModel("fly", 1000, input_dim=64))
    hidden = {k: matched_hidden_for(k, target) for k in KINDS}
    rates = (0.0003, 0.003, 0.01)
    tune = {k: v[:2048] for k, v in train.items()}
    rows = []
    for kind in KINDS:
        for rate in rates:
            for seed in (901, 902):
                _, result = fit_summary(
                    kind,
                    seed,
                    rate,
                    hidden[kind],
                    tune,
                    val,
                    8,
                    VISUAL / "summary_tuning" / f"{kind}_seed{seed}_lr{rate}.pt",
                    device,
                )
                rows.append(result)
                print(
                    json.dumps(
                        {
                            "event": "summary_tuning",
                            "kind": kind,
                            "seed": seed,
                            "lr": rate,
                            "validation": result["best_validation"],
                        }
                    ),
                    flush=True,
                )
    atomic_json(VISUAL / "summary_tuning.json", rows)
    selected = {}
    for kind in KINDS:
        related = ("fly", "rewired") if kind in ("fly", "rewired") else (kind,)

        def score(rate, related=related):
            group = [
                r for r in rows if r["kind"] in related and r["learning_rate"] == rate
            ]
            return (
                sum(r["best_validation"] for r in group) / len(group),
                -sum(r["best_validation_loss"] for r in group) / len(group),
                -rate,
            )

        selected[kind] = max(rates, key=score)
    atomic_json(VISUAL / "summary_selected_rates.json", selected)
    for kind in KINDS:
        for seed in (1201, 1202, 1203):
            path = VISUAL / "summary_main" / f"{kind}_seed{seed}.json"
            if path.exists():
                continue
            model, result = fit_summary(
                kind,
                seed,
                selected[kind],
                hidden[kind],
                train,
                val,
                20,
                path.with_suffix(".pt"),
                device,
            )
            result["tests"] = {}
            for s, dataset in tests.items():
                pred = logits_for(model, dataset, device).argmax(-1)
                result["tests"][s] = prediction_metrics(pred, dataset)
            atomic_json(path, result)
            print(
                json.dumps(
                    {
                        "event": "summary_result",
                        "kind": kind,
                        "seed": seed,
                        "tests": {s: r["accuracy"] for s, r in result["tests"].items()},
                    }
                ),
                flush=True,
            )


@torch.no_grad()
def refresh_perception(tests, device):
    """Correct label semantics without changing weights, data or selection."""
    frames = {s: frame_dataset(d, None, 41003) for s, d in tests.items()}
    for path in sorted((VISUAL / "diagnosis").glob("*.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        if result.get("perception_metric_version") == 2:
            continue
        kind, seed = result["kind"], result["seed"]
        name = f"{kind}_seed{seed}"
        saved = torch.load(result["source_checkpoint"], weights_only=True)
        model = GridMemory(
            kind, saved["result"]["hidden"], seed % 5 if kind == "rewired" else 0
        ).to(device)
        model.load_state_dict(saved["model_state"])
        probe = torch.load(
            VISUAL / "perception_heads" / f"{name}.pt", weights_only=True
        )
        decoder = nn.Linear(64, 16).to(device)
        decoder.load_state_dict(probe["decoder"])
        for s, data in frames.items():
            x = encode_frames(model, data["images"], device).to(device)
            x = (x - probe["mean"].to(device)) / probe["scale"].to(device)
            logits = torch.cat([decoder(part).cpu() for part in x.split(1024)])
            result["perception"][s] = perception_metrics(
                logits.ge(0), data["targets"].bool(), logits[:, 12:]
            )
        result["perception_metric_version"] = 2
        atomic_json(path, result)
        print(
            json.dumps({"event": "perception_metric_refresh", "run": name}), flush=True
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh-perception", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda")
    if args.refresh_perception:
        refresh_perception(new_tests(), device)
        return
    train, val = (
        previous_data("train", 8192, (6, 10)),
        previous_data("val", 1024, (6, 10)),
    )
    tests = new_tests()
    diagnose_cnn(train, val, tests, device)
    refresh_perception(tests, device)
    summary_control(
        summarized_dataset(train),
        summarized_dataset(val),
        {s: summarized_dataset(d) for s, d in tests.items()},
        device,
    )


if __name__ == "__main__":
    main()

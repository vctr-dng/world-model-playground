"""Fit state probes on the spatial summary of cached patch tokens.

A probe decodes the (cart position, cos, sin, velocities) observation from the
token grid pooled to ``--summary-side`` x ``--summary-side`` and flattened (see
``wm.patch_predictor.PatchPredictor.pooled``), stacking ``--history`` steps
``--history-stride`` control steps apart, as the patch world model's steps are.
The probe then decodes the model's predicted grids for planning and for rollout
errors in observation units. Same episode split and gate as ``train_probe.py``;
compare its R² with the pooled-feature probes to see what spatial tokens add.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np

from world_model_playground.cli import positive_int
from world_model_playground.wm.data import OBSERVATION_NAMES, split_episode_ids
from world_model_playground.wm.embeddings import load_embeddings_manifest
from world_model_playground.wm.episodes import episode_lengths
from world_model_playground.wm.probes import (
    GATE_OBSERVATIONS,
    GATE_R2,
    fit_mlp_probe,
    passes_gate,
    probe_arrays,
    regression_metrics,
    save_probe,
)
from world_model_playground.wm.runtime import torch_device
from world_model_playground.wm.token_cache import open_token_store, summary_features
from world_model_playground.wm.tracking import flat_probe_metrics, tracked_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings-dir", type=Path, required=True)
    parser.add_argument("--summary-side", type=positive_int, default=4)
    parser.add_argument("--history", type=positive_int, default=2)
    parser.add_argument(
        "--history-stride",
        type=positive_int,
        default=5,
        help="control steps between stacked steps; the world model's "
        "--step-stride (default: 5)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=positive_int, default=100)
    parser.add_argument(
        "--hidden-sizes", type=positive_int, nargs="+", default=[512, 512]
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/cartpole_world_model/probes"),
        help="probes go to <root>/<collection_id>/<embeddings>/patch_mlp_*.pt",
    )
    parser.add_argument(
        "--mlflow-experiment",
        default=os.environ.get("MLFLOW_EXPERIMENT_NAME", "cartpole-wm"),
    )
    parser.add_argument("--device", help="torch device (default: cuda if available)")
    return parser.parse_args()


def log_to_mlflow(experiment: str, run_name: str, record: dict, path: Path) -> None:
    with tracked_run(experiment, run_name, "patch_state_probe") as mlflow:
        mlflow.log_params(record["params"])
        mlflow.log_metrics(
            flat_probe_metrics("validation", record["metrics"]["validation"])
            | flat_probe_metrics("test", record["metrics"]["test"])
        )
        mlflow.log_dict(record, "probe.json")
        mlflow.log_artifact(str(path), artifact_path="probe")


def main() -> None:
    args = parse_args()
    device = torch_device(args.device)
    manifest = load_embeddings_manifest(args.embeddings_dir)
    store = open_token_store(args.embeddings_dir)
    ids = np.fromiter(episode_lengths(manifest), dtype=np.int64)
    splits = dict(
        zip(
            ("train", "validation", "test"),
            split_episode_ids(ids, seed=args.seed),
            strict=True,
        )
    )
    arrays = {}
    for split, episode_ids in splits.items():
        episodes = {
            int(i): {
                "features": summary_features(store, int(i), args.summary_side, device),
                "observations": store.observations[store.episode_slice(int(i))],
            }
            for i in episode_ids
        }
        arrays[split] = probe_arrays(episodes, args.history, args.history_stride)
    print(f"stacked feature dimension: {arrays['train'][0].shape[1]}")

    probe, best_epoch = fit_mlp_probe(
        arrays["train"],
        arrays["validation"],
        history=args.history,
        stride=args.history_stride,
        hidden_sizes=args.hidden_sizes,
        epochs=args.epochs,
        seed=args.seed,
        device=device,
    )
    metrics = {
        split: regression_metrics(probe.predict(features), targets)
        for split, (features, targets) in arrays.items()
        if split != "train"
    }
    name = f"{manifest['encoder']}_ctx{manifest['context_frames']}"
    record = {
        "params": {
            "encoder": manifest["encoder"],
            "checkpoint": manifest["checkpoint"],
            "context_frames": manifest["context_frames"],
            "collection_id": manifest["collection_id"],
            "probe": "patch_mlp",
            "summary_side": args.summary_side,
            "history": args.history,
            "history_stride": args.history_stride,
            "seed": args.seed,
            "hidden_sizes": json.dumps(args.hidden_sizes),
            "best_epoch": best_epoch,
        },
        "embeddings_dir": str(args.embeddings_dir.resolve()),
        "episode_splits": {k: v.tolist() for k, v in splits.items()},
        "passes_gate": passes_gate(metrics["test"]),
        "metrics": metrics,
    }
    path = (
        args.output_root
        / manifest["collection_id"]
        / args.embeddings_dir.name
        / (f"patch_mlp_h{args.history}_k{args.history_stride}_s{args.summary_side}.pt")
    )
    save_probe(path, probe, record)
    if os.environ.get("MLFLOW_TRACKING_URI"):
        log_to_mlflow(args.mlflow_experiment, f"{name}-patch-probe", record, path)
    print(f"Saved {path}")

    test = metrics["test"]
    print(f"\nTest R² (gate: {', '.join(GATE_OBSERVATIONS)} >= {GATE_R2})")
    print(" | ".join(["embeddings", *OBSERVATION_NAMES, "angle_mae", "gate"]))
    cells = [f"{test['r2'][obs]:.4f}" for obs in OBSERVATION_NAMES]
    print(
        " | ".join(
            [
                name,
                *cells,
                f"{test['angle_mae_rad']:.4f}",
                "pass" if record["passes_gate"] else "fail",
            ]
        )
    )


if __name__ == "__main__":
    main()

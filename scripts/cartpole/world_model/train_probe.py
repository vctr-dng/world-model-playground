"""Fit state probes on cached encoder features and compare encoders.

For each embeddings directory written by embed.py, fits linear (ridge) and/or
MLP probes from the stacked features of the last ``--history`` steps to the
CartPole observation, on the same episode split as the other world models, and
prints a test-set comparison. An encoder passes the gate when the probe reaches
R² >= 0.95 on the cart position and both pole-angle components. Runs are also
logged to MLflow when MLFLOW_TRACKING_URI is set.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np

from world_model_playground.cartpole import positive_int
from world_model_playground.wm.data import OBSERVATION_NAMES, split_episode_ids
from world_model_playground.wm.embeddings import (
    load_embedded_episodes,
    load_embeddings_manifest,
)
from world_model_playground.wm.episodes import episode_lengths
from world_model_playground.wm.probes import (
    GATE_OBSERVATIONS,
    GATE_R2,
    fit_linear_probe,
    fit_mlp_probe,
    passes_gate,
    probe_arrays,
    regression_metrics,
    save_probe,
)
from world_model_playground.wm.runtime import torch_device

PROBES = ("linear", "mlp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--embeddings-dir",
        type=Path,
        nargs="+",
        required=True,
        help="one or more directories written by embed.py",
    )
    parser.add_argument("--probes", choices=PROBES, nargs="+", default=list(PROBES))
    parser.add_argument(
        "--history",
        type=positive_int,
        default=2,
        help="steps whose features are stacked; 2+ lets single-frame encoders "
        "see velocities (default: 2)",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="episode split seed (default: 0)"
    )
    parser.add_argument(
        "--ridges",
        type=float,
        nargs="+",
        default=[1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e4],
        help="ridge strengths tried by the linear probe, picked on validation",
    )
    parser.add_argument("--epochs", type=positive_int, default=100)
    parser.add_argument(
        "--hidden-sizes", type=positive_int, nargs="+", default=[512, 512]
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/cartpole_world_model/probes"),
        help="probes go to <root>/<collection_id>/<embeddings>/<probe>_h<history>.pt",
    )
    parser.add_argument(
        "--mlflow-experiment",
        default=os.environ.get("MLFLOW_EXPERIMENT_NAME", "cartpole-wm"),
    )
    parser.add_argument("--device", help="torch device (default: cuda if available)")
    return parser.parse_args()


def flat_metrics(prefix: str, metrics: dict) -> dict[str, float]:
    flat = {f"{prefix}_angle_mae_rad": metrics["angle_mae_rad"]}
    flat[f"{prefix}_rmse"] = metrics["rmse"]["overall"]
    for name, value in metrics["r2"].items():
        flat[f"{prefix}_r2_{name}"] = value
    return flat


def log_to_mlflow(experiment: str, run_name: str, record: dict, path: Path) -> None:
    import mlflow

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name, tags={"stage": "state_probe"}):
        mlflow.log_params(record["params"])
        mlflow.log_metrics(
            flat_metrics("validation", record["metrics"]["validation"])
            | flat_metrics("test", record["metrics"]["test"])
        )
        mlflow.log_dict(record, "probe.json")
        mlflow.log_artifact(str(path), artifact_path="probe")


def main() -> None:
    args = parse_args()
    device = torch_device(args.device)
    rows = []
    for embeddings_dir in args.embeddings_dir:
        manifest = load_embeddings_manifest(embeddings_dir)
        ids = np.fromiter(episode_lengths(manifest), dtype=np.int64)
        splits = dict(
            zip(
                ("train", "validation", "test"),
                split_episode_ids(ids, seed=args.seed),
                strict=True,
            )
        )
        arrays = {
            split: probe_arrays(
                load_embedded_episodes(
                    embeddings_dir, episode_ids, fields=("features", "observations")
                ),
                args.history,
            )
            for split, episode_ids in splits.items()
        }
        for kind in args.probes:
            if kind == "linear":
                probe, ridge = fit_linear_probe(
                    arrays["train"],
                    arrays["validation"],
                    history=args.history,
                    ridges=args.ridges,
                    device=device,
                )
                fit = {"ridge": ridge}
            else:
                probe, best_epoch = fit_mlp_probe(
                    arrays["train"],
                    arrays["validation"],
                    history=args.history,
                    hidden_sizes=args.hidden_sizes,
                    epochs=args.epochs,
                    seed=args.seed,
                    device=device,
                )
                fit = {"hidden_sizes": args.hidden_sizes, "best_epoch": best_epoch}
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
                    "probe": kind,
                    "history": args.history,
                    "seed": args.seed,
                    **{k: json.dumps(v) for k, v in fit.items()},
                },
                "embeddings_dir": str(embeddings_dir.resolve()),
                "episode_splits": {k: v.tolist() for k, v in splits.items()},
                "passes_gate": passes_gate(metrics["test"]),
                "metrics": metrics,
            }
            path = (
                args.output_root
                / manifest["collection_id"]
                / embeddings_dir.name
                / f"{kind}_h{args.history}.pt"
            )
            save_probe(path, probe, record)
            if os.environ.get("MLFLOW_TRACKING_URI"):
                log_to_mlflow(args.mlflow_experiment, f"{name}-{kind}", record, path)
            rows.append((name, kind, metrics["test"], record["passes_gate"]))
            print(f"Saved {path}")

    header = ["embeddings", "probe", *OBSERVATION_NAMES, "angle_mae", "gate"]
    print(f"\nTest R² (gate: {', '.join(GATE_OBSERVATIONS)} >= {GATE_R2})")
    print(" | ".join(header))
    for name, kind, metrics, passed in rows:
        r2 = [f"{metrics['r2'][obs]:.4f}" for obs in OBSERVATION_NAMES]
        angle = f"{metrics['angle_mae_rad']:.4f}"
        print(" | ".join([name, kind, *r2, angle, "pass" if passed else "fail"]))


if __name__ == "__main__":
    main()

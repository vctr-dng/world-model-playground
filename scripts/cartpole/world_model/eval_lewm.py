"""Score a trained LeWM like the Level 1 latent models: decoded open-loop rollouts.

Run after ``train_lewm.py``, ``embed.py --encoder lewm:<artifact>`` and
``train_probe.py --history 2 --history-stride <step stride>``. The model rolls
out from the true embeddings of the test episodes with their true actions,
every ``--start-every`` steps, and the predicted embeddings are decoded by the
state probe. Rollout RMSE is compared at the horizons that are multiples of the
model stride with the linear model, the probe applied to the true embeddings
(the decoder's error floor), and persistence, on the same episode split.

The probe path is recorded in the model artifact, which is how
``latent_mpc.py`` and ``action_response.py`` find it, and the metrics are
written next to the model and logged to MLflow when MLFLOW_TRACKING_URI is set.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from world_model_playground.cli import positive_int
from world_model_playground.wm.data import split_episode_ids
from world_model_playground.wm.embeddings import (
    load_embedded_episodes,
    load_embeddings_manifest,
)
from world_model_playground.wm.episodes import episode_lengths
from world_model_playground.wm.eval import OBSERVATION_GROUPS, group_rmse
from world_model_playground.wm.lewm import ENCODER_PREFIX, lewm_rollouts, load_lewm
from world_model_playground.wm.linear import fit_linear_dynamics
from world_model_playground.wm.probes import check_probe, load_probe
from world_model_playground.wm.rollout_eval import (
    evaluate_rollouts,
    evaluation_horizons,
)
from world_model_playground.wm.runtime import torch_device
from world_model_playground.wm.tracking import flat_rollout_metrics, tracked_run

SPLITS = ("train", "validation", "test")

LATENT_KEY = "latent_rollout_rmse"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--embeddings-dir",
        type=Path,
        required=True,
        help="embeddings of the LeWM encoder written by embed.py",
    )
    parser.add_argument(
        "--model",
        type=Path,
        help="LeWM artifact (default: the one the embeddings were made with)",
    )
    parser.add_argument(
        "--probe",
        type=Path,
        help="state probe stacking steps --step-stride apart (default: "
        "<probe-root>/<collection_id>/<embeddings>/mlp_h2_s<stride>.pt)",
    )
    parser.add_argument(
        "--probe-root",
        type=Path,
        default=Path("artifacts/cartpole_world_model/probes"),
    )
    parser.add_argument("--seed", type=int, default=0, help="episode split seed")
    parser.add_argument(
        "--start-every",
        type=positive_int,
        default=20,
        help="start evaluation rollouts every this many steps of each episode",
    )
    parser.add_argument(
        "--mlflow-experiment",
        default=os.environ.get("MLFLOW_EXPERIMENT_NAME", "cartpole-wm"),
    )
    parser.add_argument("--device", help="torch device (default: cuda if available)")
    return parser.parse_args()


def record_probe(model_path: Path, probe_path: Path) -> None:
    """Stores the probe in the artifact's params, where the planners look for it."""
    artifact = torch.load(model_path, map_location="cpu", weights_only=True)
    artifact["metadata"]["params"]["probe"] = str(probe_path)
    torch.save(artifact, model_path)


def main() -> None:
    args = parse_args()
    device = torch_device(args.device)
    manifest = load_embeddings_manifest(args.embeddings_dir)
    if not manifest["encoder"].startswith(ENCODER_PREFIX):
        raise ValueError("the embeddings were not made with a LeWM encoder")
    model_path = args.model or Path(manifest["encoder"].removeprefix(ENCODER_PREFIX))
    model, metadata = load_lewm(model_path, device)
    if metadata["params"]["collection_id"] != manifest["collection_id"]:
        raise ValueError("the model was trained on another collection")
    stride = model.step_stride

    ids = np.fromiter(episode_lengths(manifest), dtype=np.int64)
    splits = dict(zip(SPLITS, split_episode_ids(ids, seed=args.seed), strict=True))
    if splits["train"].tolist() != metadata["episode_splits"]["train"]:
        raise ValueError("--seed does not reproduce the model's episode split")

    probe_name = f"mlp_h2_s{stride}.pt" if stride > 1 else "mlp_h2.pt"
    probe_path = args.probe or (
        args.probe_root
        / manifest["collection_id"]
        / args.embeddings_dir.name
        / probe_name
    )
    probe, probe_metadata = load_probe(probe_path, device)
    check_probe(probe_metadata, manifest, splits)
    if probe.stride != stride:
        raise ValueError(f"the probe stacks steps {probe.stride} apart, not {stride}")

    train = load_embedded_episodes(
        args.embeddings_dir, splits["train"], fields=("observations", "actions")
    )
    linear_model = fit_linear_dynamics(
        np.concatenate([e["observations"][:-1] for e in train.values()]),
        np.concatenate([e["actions"] for e in train.values()]),
        np.concatenate([e["observations"][1:] for e in train.values()]),
        ridge=1e-6,
    )
    horizons = evaluation_horizons(stride)
    metrics = {}
    for split in ("validation", "test"):
        episodes = load_embedded_episodes(
            args.embeddings_dir,
            splits[split],
            fields=("features", "observations", "actions"),
        )
        metrics[split] = evaluate_rollouts(
            episodes,
            label="lewm",
            rollouts=lewm_rollouts(model, probe),
            probe=probe,
            linear_model=linear_model,
            start_every=args.start_every,
            horizons=horizons,
            latent_key=LATENT_KEY,
        )

    record_probe(model_path, probe_path)
    record = {
        "model": str(model_path),
        "embeddings_dir": str(args.embeddings_dir.resolve()),
        "probe": str(probe_path),
        "probe_kind": probe.kind,
        "probe_history": probe.history,
        "start_every": args.start_every,
        "metrics": metrics,
    }
    eval_path = model_path.with_suffix(".eval.json")
    eval_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    if os.environ.get("MLFLOW_TRACKING_URI"):
        with tracked_run(
            args.mlflow_experiment, f"{model_path.stem}_eval", "lewm_eval"
        ) as mlflow:
            mlflow.log_params(
                {
                    "model": model_path.name,
                    "collection_id": manifest["collection_id"],
                    "probe_kind": probe.kind,
                    "probe_history": probe.history,
                    "start_every": args.start_every,
                    **{
                        f"model_{k}": v
                        for k, v in metadata["params"].items()
                        if k
                        in (
                            "sigreg_weight",
                            "history",
                            "step_stride",
                            "resolution",
                            "iterations",
                        )
                    },
                }
            )
            mlflow.log_metrics(
                flat_rollout_metrics(
                    "validation",
                    metrics["validation"],
                    latent_key=LATENT_KEY,
                    latent_label="latent",
                )
                | flat_rollout_metrics(
                    "test",
                    metrics["test"],
                    latent_key=LATENT_KEY,
                    latent_label="latent",
                )
            )
            mlflow.log_dict(record, "lewm_eval.json")
    print(f"Saved {eval_path}; probe recorded in {model_path}")

    observations = metrics["test"]["observation_rollout_rmse"]
    for group in OBSERVATION_GROUPS:
        print(
            f"\nTest decoded {group} rollout RMSE ({model_path.stem}, probe {probe.kind})"
        )
        print(" | ".join(["horizon", *observations]))
        for horizon in map(str, horizons):
            if horizon in observations["lewm"]:
                cells = [
                    f"{group_rmse(errors[horizon], group):.4f}"
                    for errors in observations.values()
                ]
                print(" | ".join([horizon, *cells]))


if __name__ == "__main__":
    main()

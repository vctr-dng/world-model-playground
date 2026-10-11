"""Train an action-conditioned latent world model on cached encoder features.

A causal transformer predicts the next frozen-encoder feature from the features
and actions of the last ``--history`` steps (see ``wm.predictors``). It trains
on the episode split shared with the probes and the linear baseline, then is
rolled out open loop on validation and test episodes. Predicted latents are
decoded by the Phase 2 state probe, so the rollout RMSE table compares the
latent model with the linear state-space model fitted on the same episodes,
with the probe applied to true features (the decoder's error floor), and with
persistence. Runs are also logged to MLflow when MLFLOW_TRACKING_URI is set.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np

from world_model_playground.cartpole import positive_int
from world_model_playground.wm.data import split_episode_ids
from world_model_playground.wm.embeddings import (
    load_embedded_episodes,
    load_embeddings_manifest,
)
from world_model_playground.wm.episodes import episode_lengths
from world_model_playground.wm.eval import (
    ROLLOUT_HORIZONS,
    episode_rollout_rmse,
    observation_rollout,
    persistence_rollout,
)
from world_model_playground.wm.linear import fit_linear_dynamics
from world_model_playground.wm.predictors import (
    PredictorConfig,
    decode_steps,
    fit_predictor,
    latent_rollout,
    normalized_latent_rollout,
    save_predictor,
)
from world_model_playground.wm.probes import load_probe
from world_model_playground.wm.runtime import torch_device

SPLITS = ("train", "validation", "test")
OBSERVATION_GROUPS = {"pose": slice(0, 3), "velocity": slice(3, 5)}


def non_negative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0.0:
        raise argparse.ArgumentTypeError("must be at least 0")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--embeddings-dir",
        type=Path,
        required=True,
        help="directory written by embed.py",
    )
    parser.add_argument(
        "--probe",
        type=Path,
        help="state probe decoding latents, fitted by train_probe.py on the same "
        "embeddings and split (default: <probe-root>/<collection_id>/"
        "<embeddings>/mlp_h2.pt)",
    )
    parser.add_argument(
        "--probe-root",
        type=Path,
        default=Path("artifacts/cartpole_world_model/probes"),
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="episode split seed (default: 0)"
    )
    parser.add_argument(
        "--history",
        type=positive_int,
        default=4,
        help="steps of latents and actions the predictor reads (default: 4)",
    )
    parser.add_argument(
        "--rollout-steps",
        type=positive_int,
        default=8,
        help="open-loop steps in the training loss (default: 8)",
    )
    parser.add_argument(
        "--rollout-weight",
        type=non_negative_float,
        default=1.0,
        help="weight of the open-loop loss against the one-step loss (default: 1)",
    )
    parser.add_argument("--width", type=positive_int, default=256)
    parser.add_argument("--depth", type=positive_int, default=4)
    parser.add_argument("--heads", type=positive_int, default=4)
    parser.add_argument("--dropout", type=non_negative_float, default=0.0)
    parser.add_argument("--epochs", type=positive_int, default=100)
    parser.add_argument("--batch-size", type=positive_int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument(
        "--start-every",
        type=positive_int,
        default=20,
        help="start evaluation rollouts every this many steps of each episode, "
        "for more samples per horizon (default: 20)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/cartpole_world_model/latent_wm"),
        help="predictors go to <root>/<collection_id>/<embeddings>/"
        "predictor_h<history>_r<rollout_steps>.pt",
    )
    parser.add_argument(
        "--mlflow-experiment",
        default=os.environ.get("MLFLOW_EXPERIMENT_NAME", "cartpole-wm"),
    )
    parser.add_argument("--device", help="torch device (default: cuda if available)")
    return parser.parse_args()


def check_probe(probe_metadata: dict, manifest: dict, splits: dict) -> None:
    """Rejects a probe of another encoder or one that saw the test episodes."""
    params = probe_metadata["params"]
    for name in ("encoder", "context_frames", "collection_id"):
        if params[name] != manifest[name]:
            raise ValueError(
                f"probe {name} {params[name]!r} does not match the embeddings "
                f"({manifest[name]!r})"
            )
    probe_splits = probe_metadata["episode_splits"]
    if any(probe_splits[name] != ids.tolist() for name, ids in splits.items()):
        raise ValueError(
            "the probe was fitted on another episode split; refit it with the same "
            "--seed so it never saw the validation and test episodes"
        )


def evaluate(episodes, predictor, probe, linear_model, start_every):
    """Decoded-observation and latent rollout RMSE of the model and baselines."""
    for episode in episodes.values():
        episode["latents"] = (
            episode["features"] - predictor.feature_mean.cpu().numpy()
        ) / predictor.feature_std.cpu().numpy()

    def true_features(episode, start, num_steps):
        steps = slice(start + 1, start + 1 + num_steps)
        return decode_steps(probe, episode["features"], steps)

    observation_models = {
        "latent_model": latent_rollout(predictor, probe),
        "linear_model": observation_rollout(linear_model.predict_next),
        "probe_on_true_features": true_features,
        "persistence": persistence_rollout(),
    }
    latent_models = {
        "latent_model": normalized_latent_rollout(predictor),
        "persistence": persistence_rollout("latents"),
    }
    return {
        "observation_rollout_rmse": {
            name: episode_rollout_rmse(rollout, episodes, start_every=start_every)
            for name, rollout in observation_models.items()
        },
        "normalized_latent_rollout_rmse": {
            name: {
                horizon: errors["overall"]
                for horizon, errors in episode_rollout_rmse(
                    rollout, episodes, target="latents", start_every=start_every
                ).items()
            }
            for name, rollout in latent_models.items()
        },
    }


def group_rmse(errors: dict, group: str) -> float:
    """RMSE over the pose (position and angle) or velocity observations.

    Probes decode the pose far better than the velocities, so the overall RMSE
    of decoded rollouts is dominated by the probe's velocity error.
    """
    per_observation = np.asarray(errors["per_observation"])
    return float(
        np.sqrt(np.mean(np.square(per_observation[OBSERVATION_GROUPS[group]])))
    )


def flat_metrics(split: str, metrics: dict) -> dict[str, float]:
    flat = {}
    for model, horizons in metrics["observation_rollout_rmse"].items():
        for horizon, errors in horizons.items():
            flat[f"{split}_{model}_rmse_h{horizon}"] = errors["overall"]
            for group in OBSERVATION_GROUPS:
                flat[f"{split}_{model}_{group}_rmse_h{horizon}"] = group_rmse(
                    errors, group
                )
    for model, horizons in metrics["normalized_latent_rollout_rmse"].items():
        for horizon, error in horizons.items():
            flat[f"{split}_{model}_latent_rmse_h{horizon}"] = error
    return flat


def log_to_mlflow(experiment: str, run_name: str, record: dict, path: Path) -> None:
    import mlflow

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name, tags={"stage": "latent_dynamics"}):
        mlflow.log_params(record["params"])
        for epoch, losses in enumerate(record["training"]["losses"], start=1):
            mlflow.log_metrics(losses, step=epoch)
        mlflow.log_metrics(
            flat_metrics("validation", record["metrics"]["validation"])
            | flat_metrics("test", record["metrics"]["test"])
        )
        mlflow.log_dict(record, "latent_wm.json")
        mlflow.log_artifact(str(path), artifact_path="latent_wm")


def main() -> None:
    args = parse_args()
    device = torch_device(args.device)
    manifest = load_embeddings_manifest(args.embeddings_dir)
    ids = np.fromiter(episode_lengths(manifest), dtype=np.int64)
    splits = dict(zip(SPLITS, split_episode_ids(ids, seed=args.seed), strict=True))
    episodes = {
        split: load_embedded_episodes(args.embeddings_dir, episode_ids)
        for split, episode_ids in splits.items()
    }

    probe_path = args.probe or (
        args.probe_root
        / manifest["collection_id"]
        / args.embeddings_dir.name
        / "mlp_h2.pt"
    )
    probe, probe_metadata = load_probe(probe_path, device)
    check_probe(probe_metadata, manifest, splits)

    config = PredictorConfig(
        feature_dim=manifest["feature_dim"],
        history=args.history,
        width=args.width,
        depth=args.depth,
        heads=args.heads,
        dropout=args.dropout,
    )
    predictor, best_epoch, losses = fit_predictor(
        episodes["train"],
        episodes["validation"],
        config=config,
        rollout_steps=args.rollout_steps,
        rollout_weight=args.rollout_weight,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device=device,
    )

    train_steps = [
        (e["observations"][:-1], e["actions"], e["observations"][1:])
        for e in episodes["train"].values()
    ]
    linear_model = fit_linear_dynamics(
        *(np.concatenate(arrays) for arrays in zip(*train_steps, strict=True)),
        ridge=1e-6,
    )
    metrics = {
        split: evaluate(
            episodes[split], predictor, probe, linear_model, args.start_every
        )
        for split in ("validation", "test")
    }

    name = f"{manifest['encoder']}_ctx{manifest['context_frames']}"
    record = {
        "params": {
            "encoder": manifest["encoder"],
            "checkpoint": manifest["checkpoint"],
            "context_frames": manifest["context_frames"],
            "collection_id": manifest["collection_id"],
            "probe": str(probe_path),
            "probe_kind": probe.kind,
            "probe_history": probe.history,
            "seed": args.seed,
            "rollout_steps": args.rollout_steps,
            "rollout_weight": args.rollout_weight,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "start_every": args.start_every,
            **{f"predictor_{k}": v for k, v in vars(config).items()},
        },
        "embeddings_dir": str(args.embeddings_dir.resolve()),
        "episode_splits": {k: v.tolist() for k, v in splits.items()},
        "training": {"best_epoch": best_epoch, "losses": losses},
        "metrics": metrics,
    }
    path = (
        args.output_root
        / manifest["collection_id"]
        / args.embeddings_dir.name
        / f"predictor_h{args.history}_r{args.rollout_steps}.pt"
    )
    save_predictor(path, predictor, record)
    if os.environ.get("MLFLOW_TRACKING_URI"):
        log_to_mlflow(args.mlflow_experiment, f"{name}-latent-wm", record, path)
    print(f"Saved {path} (best epoch {best_epoch}/{args.epochs})")

    observations = metrics["test"]["observation_rollout_rmse"]
    for group in OBSERVATION_GROUPS:
        print(f"\nTest decoded {group} rollout RMSE ({name}, probe {probe.kind})")
        print(" | ".join(["horizon", *observations]))
        for horizon in map(str, ROLLOUT_HORIZONS):
            if horizon in observations["latent_model"]:
                cells = [
                    f"{group_rmse(errors[horizon], group):.4f}"
                    for errors in observations.values()
                ]
                print(" | ".join([horizon, *cells]))
    test = metrics["test"]
    print("\nTest normalized latent rollout RMSE")
    latents = test["normalized_latent_rollout_rmse"]
    print(" | ".join(["horizon", *latents]))
    for horizon in latents["latent_model"]:
        print(" | ".join([horizon, *(f"{latents[m][horizon]:.4f}" for m in latents)]))
    print(json.dumps({"best_epoch": best_epoch, **losses[best_epoch - 1]}, indent=2))


if __name__ == "__main__":
    main()

"""MLflow run setup and metric flattening shared by the training scripts.

MLflow takes flat ``name -> number`` metrics, while the evaluation records are
nested by model and horizon. The key names built here are part of the logged
history, so changing them breaks comparisons with earlier runs.
"""

import os
from collections.abc import Iterator
from contextlib import contextmanager

from world_model_playground.wm.eval import OBSERVATION_GROUPS, group_rmse


@contextmanager
def tracked_run(experiment: str, run_name: str, stage: str) -> Iterator:
    """Starts an MLflow run tagged with its pipeline ``stage`` and yields ``mlflow``.

    The tracking server comes from ``MLFLOW_TRACKING_URI``, which callers check
    first so that scripts still run without a server.
    """
    import mlflow

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name, tags={"stage": stage}):
        yield mlflow


def flat_rollout_metrics(
    split: str, metrics: dict, *, latent_key: str, latent_label: str
) -> dict[str, float]:
    """Flattens an ``evaluate`` record: observation RMSE (overall and per group)
    per model and horizon, then the latent RMSE under ``metrics[latent_key]``.

    Latent keys read ``{split}_{model}_{latent_label}_rmse_h{horizon}``.
    """
    flat = {}
    for model, horizons in metrics["observation_rollout_rmse"].items():
        for horizon, errors in horizons.items():
            flat[f"{split}_{model}_rmse_h{horizon}"] = errors["overall"]
            for group in OBSERVATION_GROUPS:
                flat[f"{split}_{model}_{group}_rmse_h{horizon}"] = group_rmse(
                    errors, group
                )
    for model, horizons in metrics[latent_key].items():
        for horizon, error in horizons.items():
            flat[f"{split}_{model}_{latent_label}_rmse_h{horizon}"] = error
    return flat


def flat_probe_metrics(prefix: str, metrics: dict) -> dict[str, float]:
    """Flattens the metrics of a state probe: angle error, RMSE and per-state R²."""
    flat = {f"{prefix}_angle_mae_rad": metrics["angle_mae_rad"]}
    flat[f"{prefix}_rmse"] = metrics["rmse"]["overall"]
    for name, value in metrics["r2"].items():
        flat[f"{prefix}_r2_{name}"] = value
    return flat

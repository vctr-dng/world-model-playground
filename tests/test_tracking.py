"""Tests for metric flattening and the shared rollout evaluation.

The flattened key names are part of the history logged in MLflow, so the
reference functions below are the per-script implementations these helpers
replaced, and the tests demand identical output.
"""

import numpy as np
import pytest

from world_model_playground.wm import tracking
from world_model_playground.wm.eval import OBSERVATION_GROUPS, group_rmse
from world_model_playground.wm.rollout_eval import (
    evaluate_rollouts,
    evaluation_horizons,
)


def reference_rollout_metrics(split, metrics, latent_key, latent_label):
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


def reference_probe_metrics(prefix, metrics):
    flat = {f"{prefix}_angle_mae_rad": metrics["angle_mae_rad"]}
    flat[f"{prefix}_rmse"] = metrics["rmse"]["overall"]
    for name, value in metrics["r2"].items():
        flat[f"{prefix}_r2_{name}"] = value
    return flat


def rollout_record(latent_key):
    errors = {"overall": 0.3, "per_observation": [0.1, 0.2, 0.3, 0.4, 0.5]}
    return {
        "observation_rollout_rmse": {
            "model": {"5": errors, "25": errors},
            "persistence": {"5": errors},
        },
        latent_key: {"model": {"5": 0.7, "25": 0.9}, "persistence": {"5": 1.1}},
    }


@pytest.mark.parametrize(
    ("latent_key", "latent_label"),
    [
        ("normalized_latent_rollout_rmse", "latent"),
        ("latent_rollout_rmse", "latent"),
        ("pooled_latent_rollout_rmse", "pooled_latent"),
    ],
)
def test_flat_rollout_metrics_keeps_the_logged_key_names(latent_key, latent_label):
    record = rollout_record(latent_key)
    flat = tracking.flat_rollout_metrics(
        "test", record, latent_key=latent_key, latent_label=latent_label
    )
    assert flat == reference_rollout_metrics("test", record, latent_key, latent_label)
    assert f"test_model_{latent_label}_rmse_h25" in flat
    assert "test_model_pose_rmse_h5" in flat


def test_flat_probe_metrics_keeps_the_logged_key_names():
    metrics = {
        "angle_mae_rad": 0.02,
        "rmse": {"overall": 0.1},
        "r2": {"cart_position": 0.99, "pole_angle": 0.98},
    }
    flat = tracking.flat_probe_metrics("validation", metrics)
    assert flat == reference_probe_metrics("validation", metrics)
    assert set(flat) == {
        "validation_angle_mae_rad",
        "validation_rmse",
        "validation_r2_cart_position",
        "validation_r2_pole_angle",
    }


def test_evaluation_horizons_keeps_multiples_of_the_stride():
    assert evaluation_horizons(5) == (5, 10, 25, 50, 100, 200)
    assert evaluation_horizons(10) == (10, 10, 50, 100, 200)


class StubProbe:
    """Decodes a feature to the observations it was built from."""

    history = 1
    stride = 1

    @staticmethod
    def predict(features):
        return features


class StubLinear:
    @staticmethod
    def predict_next(observations, actions):
        return observations


def test_evaluate_rollouts_scores_the_model_against_the_baselines():
    steps = 6
    observations = np.arange((steps + 1) * 5, dtype=float).reshape(steps + 1, 5)
    episodes = {
        0: {
            "observations": observations,
            "features": observations.copy(),
            "actions": np.zeros((steps, 1)),
        }
    }

    def perfect(episode, start, num_steps):
        return episode["observations"][start + 1 : start + 1 + num_steps]

    result = evaluate_rollouts(
        episodes,
        label="model",
        rollouts=(perfect, perfect),
        probe=StubProbe(),
        linear_model=StubLinear(),
        start_every=1,
        horizons=(1, 2),
        latent_key="latent_rollout_rmse",
    )

    assert list(result) == ["observation_rollout_rmse", "latent_rollout_rmse"]
    assert list(result["observation_rollout_rmse"]) == [
        "model",
        "linear_model",
        "probe_on_true_features",
        "persistence",
    ]
    assert list(result["latent_rollout_rmse"]) == ["model", "persistence"]
    for horizon in result["observation_rollout_rmse"]["model"]:
        assert result["observation_rollout_rmse"]["model"][horizon]["overall"] == 0
        assert result["latent_rollout_rmse"]["model"][horizon] == 0
        assert result["observation_rollout_rmse"]["persistence"][horizon]["overall"] > 0

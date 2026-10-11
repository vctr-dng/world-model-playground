"""Linear residual ridge world model on CartPole observations."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from world_model_playground.wm.data import project_angle_pair


@dataclass(frozen=True)
class Normalization:
    """Affine feature normalization fitted on training data only."""

    mean: np.ndarray
    std: np.ndarray

    def normalize(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.std

    def denormalize(self, values: np.ndarray) -> np.ndarray:
        return values * self.std + self.mean


@dataclass(frozen=True)
class LinearResidualDynamics:
    """Action-conditioned ridge model that predicts the next observation delta."""

    feature_normalization: Normalization
    delta_normalization: Normalization
    weights: np.ndarray
    bias: np.ndarray

    def predict_next(self, observations: np.ndarray, actions: np.ndarray) -> np.ndarray:
        observations = np.asarray(observations, dtype=np.float64)
        actions = np.asarray(actions, dtype=np.float64)
        features = np.concatenate((observations, actions), axis=-1)
        normalized_features = self.feature_normalization.normalize(features)
        normalized_delta = normalized_features @ self.weights.T + self.bias
        next_observations = observations + self.delta_normalization.denormalize(
            normalized_delta
        )
        return project_angle_pair(next_observations)


def fit_normalization(values: np.ndarray) -> Normalization:
    """Return stable per-feature statistics for a non-empty training matrix."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or len(values) == 0:
        raise ValueError("normalization requires a non-empty two-dimensional array")
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    return Normalization(mean=mean, std=np.where(std > 1e-12, std, 1.0))


def fit_linear_dynamics(
    observations: np.ndarray,
    actions: np.ndarray,
    next_observations: np.ndarray,
    *,
    ridge: float,
) -> LinearResidualDynamics:
    """Fit normalized ridge regression to next-observation deltas."""
    observations = np.asarray(observations, dtype=np.float64)
    actions = np.asarray(actions, dtype=np.float64)
    next_observations = np.asarray(next_observations, dtype=np.float64)
    if not (len(observations) == len(actions) == len(next_observations)):
        raise ValueError(
            "observation, action, and target arrays must have equal length"
        )

    features = np.concatenate((observations, actions), axis=1)
    deltas = next_observations - observations
    feature_normalization = fit_normalization(features)
    delta_normalization = fit_normalization(deltas)
    normalized_features = feature_normalization.normalize(features)
    normalized_deltas = delta_normalization.normalize(deltas)
    design = np.column_stack((normalized_features, np.ones(len(normalized_features))))
    regularizer = np.eye(design.shape[1]) * ridge
    regularizer[-1, -1] = 0.0  # Do not regularize the intercept.
    coefficients = np.linalg.solve(
        design.T @ design + regularizer, design.T @ normalized_deltas
    ).T
    return LinearResidualDynamics(
        feature_normalization=feature_normalization,
        delta_normalization=delta_normalization,
        weights=coefficients[:, :-1],
        bias=coefficients[:, -1],
    )


def save_linear_model(
    path: Path, model: LinearResidualDynamics, metadata: dict[str, object]
) -> None:
    """Write the model parameters and JSON metadata to a compressed artifact."""
    np.savez_compressed(
        path,
        weights=model.weights,
        bias=model.bias,
        feature_mean=model.feature_normalization.mean,
        feature_std=model.feature_normalization.std,
        delta_mean=model.delta_normalization.mean,
        delta_std=model.delta_normalization.std,
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
    )


def load_linear_model(path: Path) -> LinearResidualDynamics:
    """Restore a model written by :func:`save_linear_model`."""
    with np.load(path, allow_pickle=False) as artifact:
        return LinearResidualDynamics(
            feature_normalization=Normalization(
                artifact["feature_mean"], artifact["feature_std"]
            ),
            delta_normalization=Normalization(
                artifact["delta_mean"], artifact["delta_std"]
            ),
            weights=artifact["weights"],
            bias=artifact["bias"],
        )

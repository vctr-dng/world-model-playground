"""Train a linear residual world model from CartPole observation transitions."""

import argparse
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_OBSERVATION_SIZE = 5
_ACTION_SIZE = 1
_ROLLOUT_HORIZONS = (1, 10, 25, 50, 100, 200)


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


def positive_float(value: str) -> float:
    """Parse a strictly positive floating-point value."""
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def fit_normalization(values: np.ndarray) -> Normalization:
    """Return stable per-feature statistics for a non-empty training matrix."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or len(values) == 0:
        raise ValueError("normalization requires a non-empty two-dimensional array")
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    return Normalization(mean=mean, std=np.where(std > 1e-12, std, 1.0))


def project_angle_pair(observations: np.ndarray) -> np.ndarray:
    """Project the cosine/sine observation components back onto the unit circle."""
    projected = np.array(observations, dtype=np.float64, copy=True)
    pair = projected[..., 1:3]
    norm = np.linalg.norm(pair, axis=-1, keepdims=True)
    valid = norm > 1e-12
    projected[..., 1:3] = np.divide(pair, norm, out=np.zeros_like(pair), where=valid)
    if np.any(~valid):
        projected[..., 1] = np.where(valid[..., 0], projected[..., 1], 1.0)
        projected[..., 2] = np.where(valid[..., 0], projected[..., 2], 0.0)
    return projected


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


def split_episode_ids(
    episode_ids: np.ndarray, *, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split complete episodes into reproducible 80/10/10 partitions."""
    unique_ids = np.unique(episode_ids)
    if len(unique_ids) < 3:
        raise ValueError(
            "at least three episodes are required for train/validation/test"
        )
    shuffled = np.random.default_rng(seed).permutation(unique_ids)
    train_count = max(1, int(len(shuffled) * 0.8))
    validation_count = max(1, int(len(shuffled) * 0.1))
    test_count = len(shuffled) - train_count - validation_count
    if test_count < 1:
        train_count -= 1
        test_count = 1
    return (
        shuffled[:train_count],
        shuffled[train_count : train_count + validation_count],
        shuffled[train_count + validation_count :],
    )


def validate_transitions(transitions: Mapping[str, np.ndarray]) -> int:
    """Validate the observation-only transition fields required by this trainer."""
    required = (
        "observations",
        "actions",
        "next_observations",
        "episode_ids",
        "timesteps",
    )
    missing = [name for name in required if name not in transitions]
    if missing:
        raise ValueError(f"transitions.npz is missing fields: {', '.join(missing)}")

    observations = transitions["observations"]
    actions = transitions["actions"]
    next_observations = transitions["next_observations"]
    count = len(observations)
    if observations.shape != (count, _OBSERVATION_SIZE):
        raise ValueError(f"observations must have shape (N, {_OBSERVATION_SIZE})")
    if next_observations.shape != observations.shape:
        raise ValueError("next_observations must match observations shape")
    if actions.shape != (count, _ACTION_SIZE):
        raise ValueError(f"actions must have shape (N, {_ACTION_SIZE})")
    for name in required[3:]:
        if transitions[name].shape != (count,):
            raise ValueError(f"{name} must have shape (N,)")
    if not all(
        np.isfinite(transitions[name]).all()
        for name in ("observations", "actions", "next_observations")
    ):
        raise ValueError("observations and actions must be finite")
    return count


def partition_mask(episode_ids: np.ndarray, partition: np.ndarray) -> np.ndarray:
    """Return the transition rows belonging to a set of complete episodes."""
    return np.isin(episode_ids, partition)


def rmse(predictions: np.ndarray, targets: np.ndarray) -> dict[str, object]:
    """Summarize total and per-observation-root-mean-squared error."""
    errors = np.asarray(predictions) - np.asarray(targets)
    return {
        "overall": float(np.sqrt(np.mean(np.square(errors)))),
        "per_observation": np.sqrt(np.mean(np.square(errors), axis=0)).tolist(),
    }


def rollout_rmse(
    model: LinearResidualDynamics,
    transitions: Mapping[str, np.ndarray],
    episode_ids: np.ndarray,
) -> dict[str, dict[str, object]]:
    """Evaluate open-loop predictions from each selected episode's initial state."""
    horizon_errors: dict[int, list[np.ndarray]] = {
        horizon: [] for horizon in _ROLLOUT_HORIZONS
    }
    all_episode_ids = transitions["episode_ids"]
    timesteps = transitions["timesteps"]
    for episode_id in episode_ids:
        rows = np.flatnonzero(all_episode_ids == episode_id)
        rows = rows[np.argsort(timesteps[rows])]
        prediction = transitions["observations"][rows[0]][None, :]
        for horizon, row in enumerate(rows, start=1):
            action = transitions["actions"][row][None, :]
            prediction = model.predict_next(prediction, action)
            if horizon in horizon_errors:
                horizon_errors[horizon].append(
                    prediction[0] - transitions["next_observations"][row]
                )
    return {
        str(horizon): {
            "overall": float(np.sqrt(np.mean(np.square(errors)))),
            "per_observation": np.sqrt(np.mean(np.square(errors), axis=0)).tolist(),
        }
        for horizon, errors in horizon_errors.items()
        if errors
    }


def evaluate_partition(
    model: LinearResidualDynamics,
    transitions: Mapping[str, np.ndarray],
    episode_ids: np.ndarray,
) -> dict[str, object]:
    """Return one-step and rollout errors for an episode-disjoint partition."""
    mask = partition_mask(transitions["episode_ids"], episode_ids)
    predictions = model.predict_next(
        transitions["observations"][mask], transitions["actions"][mask]
    )
    persistence = transitions["observations"][mask]
    targets = transitions["next_observations"][mask]
    return {
        "num_episodes": len(episode_ids),
        "num_transitions": int(mask.sum()),
        "one_step_rmse": rmse(predictions, targets),
        "persistence_one_step_rmse": rmse(persistence, targets),
        "rollout_rmse": rollout_rmse(model, transitions, episode_ids),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--collection-dir",
        type=Path,
        required=True,
        help="directory containing transitions.npz",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/cartpole_world_model/linear_model.npz"),
        help="output model artifact (default: artifacts/cartpole_world_model/linear_model.npz)",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="episode split seed (default: 0)"
    )
    parser.add_argument(
        "--ridge",
        type=positive_float,
        default=1e-6,
        help="ridge regularization strength (default: 1e-6)",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="replace an existing output artifact"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    transitions_path = args.collection_dir / "transitions.npz"
    with np.load(transitions_path, allow_pickle=False) as archive:
        transitions = {name: archive[name] for name in archive.files}
    validate_transitions(transitions)
    train_ids, validation_ids, test_ids = split_episode_ids(
        transitions["episode_ids"], seed=args.seed
    )
    train_mask = partition_mask(transitions["episode_ids"], train_ids)
    model = fit_linear_dynamics(
        transitions["observations"][train_mask],
        transitions["actions"][train_mask],
        transitions["next_observations"][train_mask],
        ridge=args.ridge,
    )
    metrics = {
        "validation": evaluate_partition(model, transitions, validation_ids),
        "test": evaluate_partition(model, transitions, test_ids),
    }
    collection_metadata = json.loads(str(transitions.get("metadata", "{}")))
    artifact_metadata = {
        "architecture": "linear_residual_ridge",
        "collection_id": collection_metadata.get("collection_id"),
        "collection_dir": str(args.collection_dir.resolve()),
        "feature_order": [
            "cart_position",
            "cos_theta",
            "sin_theta",
            "cart_velocity",
            "angular_velocity",
            "action",
        ],
        "target_order": [
            "cart_position_delta",
            "cos_theta_delta",
            "sin_theta_delta",
            "cart_velocity_delta",
            "angular_velocity_delta",
        ],
        "ridge": args.ridge,
        "seed": args.seed,
        "episode_splits": {
            "train": train_ids.tolist(),
            "validation": validation_ids.tolist(),
            "test": test_ids.tolist(),
        },
        "metrics": metrics,
    }
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {args.output}; pass --overwrite to replace it"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        weights=model.weights,
        bias=model.bias,
        feature_mean=model.feature_normalization.mean,
        feature_std=model.feature_normalization.std,
        delta_mean=model.delta_normalization.mean,
        delta_std=model.delta_normalization.std,
        metadata=np.asarray(json.dumps(artifact_metadata, sort_keys=True)),
    )
    print(f"Saved linear world model: {args.output}")
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

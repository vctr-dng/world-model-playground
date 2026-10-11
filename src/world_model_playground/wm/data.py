"""Loading, validating, and splitting collected CartPole transitions."""

from collections.abc import Mapping
from pathlib import Path

import numpy as np

OBSERVATION_SIZE = 5
ACTION_SIZE = 1
OBSERVATION_NAMES = (
    "cart_position",
    "cos_theta",
    "sin_theta",
    "cart_velocity",
    "angular_velocity",
)


def load_transitions(collection_dir: Path) -> dict[str, np.ndarray]:
    """Read every array of a collection's ``transitions.npz`` into memory."""
    with np.load(collection_dir / "transitions.npz", allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def validate_transitions(transitions: Mapping[str, np.ndarray]) -> int:
    """Validate the observation-only transition fields required by the trainers."""
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
    if observations.shape != (count, OBSERVATION_SIZE):
        raise ValueError(f"observations must have shape (N, {OBSERVATION_SIZE})")
    if next_observations.shape != observations.shape:
        raise ValueError("next_observations must match observations shape")
    if actions.shape != (count, ACTION_SIZE):
        raise ValueError(f"actions must have shape (N, {ACTION_SIZE})")
    for name in required[3:]:
        if transitions[name].shape != (count,):
            raise ValueError(f"{name} must have shape (N,)")
    if not all(
        np.isfinite(transitions[name]).all()
        for name in ("observations", "actions", "next_observations")
    ):
        raise ValueError("observations and actions must be finite")
    return count


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


def partition_mask(episode_ids: np.ndarray, partition: np.ndarray) -> np.ndarray:
    """Return the transition rows belonging to a set of complete episodes."""
    return np.isin(episode_ids, partition)


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

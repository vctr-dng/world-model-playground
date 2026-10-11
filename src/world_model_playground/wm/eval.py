"""Model-agnostic one-step and open-loop rollout metrics on observation space.

Any world model is evaluated through a ``predict_next(observations, actions)``
callable on batched NumPy arrays, so learned latent models can be compared with
the linear baseline once they decode back to observations.
"""

from collections.abc import Callable, Mapping

import numpy as np

from world_model_playground.wm.data import partition_mask

ROLLOUT_HORIZONS = (1, 10, 25, 50, 100, 200)

type PredictNext = Callable[[np.ndarray, np.ndarray], np.ndarray]
# (episode, start step, number of steps) -> predictions for the following steps.
type EpisodeRollout = Callable[[Mapping[str, np.ndarray], int, int], np.ndarray]


def rmse(predictions: np.ndarray, targets: np.ndarray) -> dict[str, object]:
    """Summarize total and per-observation-root-mean-squared error."""
    errors = np.asarray(predictions) - np.asarray(targets)
    return {
        "overall": float(np.sqrt(np.mean(np.square(errors)))),
        "per_observation": np.sqrt(np.mean(np.square(errors), axis=0)).tolist(),
    }


def rollout_rmse(
    predict_next: PredictNext,
    transitions: Mapping[str, np.ndarray],
    episode_ids: np.ndarray,
    horizons: tuple[int, ...] = ROLLOUT_HORIZONS,
) -> dict[str, dict[str, object]]:
    """Evaluate open-loop predictions from each selected episode's initial state."""
    horizon_errors: dict[int, list[np.ndarray]] = {horizon: [] for horizon in horizons}
    all_episode_ids = transitions["episode_ids"]
    timesteps = transitions["timesteps"]
    for episode_id in episode_ids:
        rows = np.flatnonzero(all_episode_ids == episode_id)
        rows = rows[np.argsort(timesteps[rows])]
        prediction = transitions["observations"][rows[0]][None, :]
        for horizon, row in enumerate(rows, start=1):
            action = transitions["actions"][row][None, :]
            prediction = predict_next(prediction, action)
            if horizon in horizon_errors:
                horizon_errors[horizon].append(
                    prediction[0] - transitions["next_observations"][row]
                )
    return {
        str(horizon): rmse(np.asarray(errors), 0.0)
        for horizon, errors in horizon_errors.items()
        if errors
    }


def evaluate_partition(
    predict_next: PredictNext,
    transitions: Mapping[str, np.ndarray],
    episode_ids: np.ndarray,
) -> dict[str, object]:
    """Return one-step and rollout errors for an episode-disjoint partition."""
    mask = partition_mask(transitions["episode_ids"], episode_ids)
    predictions = predict_next(
        transitions["observations"][mask], transitions["actions"][mask]
    )
    persistence = transitions["observations"][mask]
    targets = transitions["next_observations"][mask]
    return {
        "num_episodes": len(episode_ids),
        "num_transitions": int(mask.sum()),
        "one_step_rmse": rmse(predictions, targets),
        "persistence_one_step_rmse": rmse(persistence, targets),
        "rollout_rmse": rollout_rmse(predict_next, transitions, episode_ids),
    }


def episode_rollout_rmse(
    rollout: EpisodeRollout,
    episodes: Mapping[int, Mapping[str, np.ndarray]],
    *,
    target: str = "observations",
    horizons: tuple[int, ...] = ROLLOUT_HORIZONS,
    start_every: int | None = None,
) -> dict[str, dict[str, object]]:
    """Open-loop error of any model rolled out over episode shards.

    Episodes hold per-step arrays: ``actions`` ``(T, A)`` and the ``target``
    ``(T+1, ...)`` the rollout predicts. Rollouts start from step 0, matching
    :func:`rollout_rmse`, or from every ``start_every`` steps for more samples
    per horizon; a horizon is scored only from starts that reach it.
    """
    horizon_errors: dict[int, list[np.ndarray]] = {horizon: [] for horizon in horizons}
    for episode in episodes.values():
        length = len(episode["actions"])
        starts = range(0, length, start_every) if start_every else (0,)
        for start in starts:
            num_steps = min(max(horizons), length - start)
            predictions = rollout(episode, start, num_steps)
            for horizon in horizons:
                if horizon <= num_steps:
                    horizon_errors[horizon].append(
                        predictions[horizon - 1] - episode[target][start + horizon]
                    )
    return {
        str(horizon): rmse(np.asarray(errors), 0.0)
        for horizon, errors in horizon_errors.items()
        if errors
    }


def observation_rollout(predict_next: PredictNext) -> EpisodeRollout:
    """Rolls an observation-space model out from an episode's true observation."""

    def rollout(episode, start, num_steps):
        prediction = episode["observations"][start][None, :]
        predictions = []
        for action in episode["actions"][start : start + num_steps]:
            prediction = predict_next(prediction, action[None, :])
            predictions.append(prediction[0])
        return np.asarray(predictions)

    return rollout


def persistence_rollout(target: str = "observations") -> EpisodeRollout:
    """Predicts that the ``target`` stays at its value at the start step."""

    def rollout(episode, start, num_steps):
        return np.repeat(episode[target][start][None], num_steps, axis=0)

    return rollout

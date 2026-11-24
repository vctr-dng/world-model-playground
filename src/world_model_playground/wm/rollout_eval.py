"""Open-loop rollout scoring of a latent world model against the same baselines.

Latent models are scored the same way: decoded to observations by a state probe
and compared with the linear model, the probe applied to the true features (the
decoder's error floor), and persistence, on the same episodes.
"""

from world_model_playground.wm.eval import (
    ROLLOUT_HORIZONS,
    EpisodeRollout,
    episode_rollout_rmse,
    observation_rollout,
    persistence_rollout,
)
from world_model_playground.wm.predictors import decode_steps


def evaluation_horizons(stride: int) -> tuple[int, ...]:
    """Rollout horizons where a model stepping ``stride`` control steps is exact."""
    return tuple(h for h in (stride, *ROLLOUT_HORIZONS) if h % stride == 0)


def evaluate_rollouts(
    episodes,
    *,
    label: str,
    rollouts: tuple[EpisodeRollout, EpisodeRollout],
    probe,
    linear_model,
    start_every: int,
    horizons: tuple[int, ...],
    latent_key: str,
) -> dict[str, dict]:
    """Decoded-observation and latent rollout RMSE of a model and the baselines.

    ``rollouts`` is the model's ``(observation, latent)`` pair, named ``label``
    in the result. Episodes carry ``features``, ``observations`` and ``actions``.
    The latent errors are stored under ``latent_key``.
    """
    observations, latents = rollouts

    def true_features(episode, start, num_steps):
        steps = slice(start + 1, start + 1 + num_steps)
        return decode_steps(probe, episode["features"], steps)

    observation_models = {
        label: observations,
        "linear_model": observation_rollout(linear_model.predict_next),
        "probe_on_true_features": true_features,
        "persistence": persistence_rollout(),
    }
    latent_models = {label: latents, "persistence": persistence_rollout("features")}
    return {
        "observation_rollout_rmse": {
            name: episode_rollout_rmse(
                rollout, episodes, horizons=horizons, start_every=start_every
            )
            for name, rollout in observation_models.items()
        },
        latent_key: {
            name: {
                horizon: errors["overall"]
                for horizon, errors in episode_rollout_rmse(
                    rollout,
                    episodes,
                    target="features",
                    horizons=horizons,
                    start_every=start_every,
                ).items()
            }
            for name, rollout in latent_models.items()
        },
    }

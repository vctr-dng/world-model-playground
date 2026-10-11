"""Tests for the model-agnostic world-model evaluation metrics."""

import numpy as np

from world_model_playground.wm import eval as wm_eval


def make_episodes(num_episodes: int, length: int) -> dict[str, np.ndarray]:
    """Builds episodes whose next observation is the current one plus the action."""
    rng = np.random.default_rng(1)
    rows = {name: [] for name in ("observations", "actions", "next_observations")}
    episode_ids, timesteps = [], []
    for episode in range(num_episodes):
        observation = rng.normal(size=5)
        for timestep in range(length):
            action = rng.uniform(-1.0, 1.0, size=1)
            next_observation = observation + action[0]
            rows["observations"].append(observation)
            rows["actions"].append(action)
            rows["next_observations"].append(next_observation)
            episode_ids.append(episode)
            timesteps.append(timestep)
            observation = next_observation
    # Shuffle rows so the evaluator must reorder each episode by timestep.
    order = rng.permutation(len(episode_ids))
    transitions = {name: np.asarray(values)[order] for name, values in rows.items()}
    transitions["episode_ids"] = np.asarray(episode_ids)[order]
    transitions["timesteps"] = np.asarray(timesteps)[order]
    return transitions


def exact_dynamics(observations, actions):
    return observations + actions


def test_rollout_rmse_is_zero_for_exact_dynamics_and_skips_unreached_horizons():
    transitions = make_episodes(num_episodes=3, length=30)

    errors = wm_eval.rollout_rmse(exact_dynamics, transitions, np.array([0, 2]))

    assert sorted(errors, key=int) == ["1", "10", "25"]
    for horizon_error in errors.values():
        assert horizon_error["overall"] < 1e-12
        assert len(horizon_error["per_observation"]) == 5


def test_rollout_rmse_accumulates_open_loop_error():
    transitions = make_episodes(num_episodes=2, length=25)

    def biased_dynamics(observations, actions):
        return observations + actions + 0.01

    errors = wm_eval.rollout_rmse(biased_dynamics, transitions, np.array([0, 1]))

    np.testing.assert_allclose(errors["1"]["overall"], 0.01)
    np.testing.assert_allclose(errors["25"]["overall"], 0.25)


def test_evaluate_partition_only_uses_selected_episodes():
    transitions = make_episodes(num_episodes=4, length=10)

    metrics = wm_eval.evaluate_partition(exact_dynamics, transitions, np.array([1, 3]))

    assert metrics["num_episodes"] == 2
    assert metrics["num_transitions"] == 20
    assert metrics["one_step_rmse"]["overall"] < 1e-12
    assert metrics["persistence_one_step_rmse"]["overall"] > 0.1

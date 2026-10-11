"""Tests for the CartPole linear residual world model."""

import numpy as np

from world_model_playground.wm import data, linear


def test_fit_linear_dynamics_recovers_linear_residuals():
    rng = np.random.default_rng(3)
    observations = np.column_stack(
        (
            rng.normal(size=80),
            np.ones(80),
            np.zeros(80),
            rng.normal(size=80),
            rng.normal(size=80),
        )
    )
    actions = rng.uniform(-1.0, 1.0, size=(80, 1))
    deltas = np.zeros_like(observations)
    deltas[:, 0] = 0.15 * observations[:, 0] + 0.35 * actions[:, 0]
    deltas[:, 3] = -0.2 * observations[:, 3] + 0.1 * actions[:, 0]
    deltas[:, 4] = 0.4 * observations[:, 4] - 0.3 * actions[:, 0]
    targets = observations + deltas

    model = linear.fit_linear_dynamics(observations, actions, targets, ridge=1e-8)

    np.testing.assert_allclose(
        model.predict_next(observations, actions), targets, atol=1e-6
    )


def test_normalization_uses_only_values_supplied_for_training():
    normalization = linear.fit_normalization(np.array([[0.0], [2.0]]))

    np.testing.assert_allclose(normalization.mean, [1.0])
    np.testing.assert_allclose(normalization.std, [1.0])
    np.testing.assert_allclose(normalization.normalize(np.array([[100.0]])), [[99.0]])


def test_project_angle_pair_returns_unit_norm_for_zero_and_scaled_pairs():
    observations = np.array([[0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 3.0, 4.0, 0.0, 0.0]])

    projected = data.project_angle_pair(observations)

    np.testing.assert_allclose(np.linalg.norm(projected[:, 1:3], axis=1), [1.0, 1.0])
    np.testing.assert_allclose(projected[0, 1:3], [1.0, 0.0])
    np.testing.assert_allclose(projected[1, 1:3], [0.6, 0.8])


def test_split_episode_ids_is_reproducible_and_disjoint():
    train, validation, test = data.split_episode_ids(
        np.repeat(np.arange(10), 2), seed=7
    )

    assert len(train) == 8
    assert len(validation) == 1
    assert len(test) == 1
    assert not np.intersect1d(train, validation).size
    assert not np.intersect1d(train, test).size
    assert not np.intersect1d(validation, test).size
    second_train, second_validation, second_test = data.split_episode_ids(
        np.repeat(np.arange(10), 2), seed=7
    )
    np.testing.assert_array_equal(train, second_train)
    np.testing.assert_array_equal(validation, second_validation)
    np.testing.assert_array_equal(test, second_test)


def test_save_and_load_linear_model_round_trip(tmp_path):
    rng = np.random.default_rng(0)
    observations = data.project_angle_pair(rng.normal(size=(40, 5)))
    actions = rng.uniform(-1.0, 1.0, size=(40, 1))
    model = linear.fit_linear_dynamics(
        observations, actions, observations + 0.1 * actions, ridge=1e-6
    )
    path = tmp_path / "model.npz"

    linear.save_linear_model(path, model, {"architecture": "test"})
    restored = linear.load_linear_model(path)

    np.testing.assert_array_equal(
        restored.predict_next(observations, actions),
        model.predict_next(observations, actions),
    )

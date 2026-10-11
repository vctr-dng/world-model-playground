"""Tests for CEM planning, the CartPole reward, and the model evaluators."""

import jax
import numpy as np
import pytest
import torch

from world_model_playground.cartpole import load_cartpole
from world_model_playground.wm import planning
from world_model_playground.wm.linear import fit_linear_dynamics
from world_model_playground.wm.predictors import LatentPredictor, PredictorConfig
from world_model_playground.wm.probes import StateProbe, stack_history


def test_cartpole_reward_matches_the_environment():
    env = load_cartpole(full_circle_reset=True)
    keys = jax.random.split(jax.random.key(0), 64)
    states = jax.vmap(env.reset)(keys)
    actions = jax.random.uniform(jax.random.key(1), (64, 1), minval=-1.3, maxval=1.3)
    # Spin some poles up so the velocity and position terms are exercised.
    states = states.replace(
        data=states.data.replace(qvel=states.data.qvel * 400.0 + 1.0)
    )
    following = jax.vmap(env.step)(states, actions)

    reward = planning.cartpole_reward(
        torch.as_tensor(np.array(following.obs)),
        torch.as_tensor(np.array(actions)),
    )

    np.testing.assert_allclose(reward.numpy(), np.asarray(following.reward), atol=1e-5)
    assert (
        float(planning.cartpole_reward(torch.tensor([0.0, 1, 0, 0, 0]), torch.zeros(1)))
        == 1.0
    )


def test_cem_finds_the_best_constant_action_of_a_toy_problem():
    config = planning.CEMConfig(
        horizon=6, candidates=64, iterations=5, elites=8, action_repeat=3
    )
    planner = planning.CEMPlanner(config, batch_size=2, seed=0)
    targets = torch.tensor([[0.4], [-0.7]])

    def evaluate(actions):
        returns = -torch.square(actions - targets[:, None, None]).sum((2, 3))
        return planning.Evaluation(returns, {"mean_action": actions.mean((2, 3))})

    plan = planner.plan(evaluate)

    assert plan.actions.shape == (2, 6, 1)
    np.testing.assert_allclose(plan.first_action.numpy(), targets.numpy(), atol=0.05)
    np.testing.assert_allclose(
        plan.extras["mean_action"].numpy(), targets[:, 0].numpy(), atol=0.05
    )
    # Knots are held for action_repeat steps.
    assert torch.equal(plan.actions[:, 0], plan.actions[:, 2])


def test_warm_start_shifts_the_plan_by_one_step():
    config = planning.CEMConfig(horizon=6, candidates=2, elites=1, action_repeat=2)
    planner = planning.CEMPlanner(config, batch_size=1)
    knots = torch.tensor([[[0.1], [0.2], [0.3]]])

    shifted = planner._shifted(knots)

    # Steps 0.1 0.1 0.2 0.2 0.3 0.3 advance to 0.1 0.2 0.2 0.3 0.3 0.3.
    np.testing.assert_allclose(shifted[0, :, 0].numpy(), [0.1, 0.2, 0.3])


def make_probe(feature_dim: int, history: int) -> StateProbe:
    torch.manual_seed(0)
    return StateProbe(feature_dim * history, history=history, hidden_sizes=(16,))


def test_decode_latents_stacks_history_like_the_probe_features():
    probe = make_probe(feature_dim=3, history=2)
    features = torch.randn(1, 7, 3)

    decoded = planning.decode_latents(probe, features[:, :3], features[:, 3:])

    stacked = stack_history(features[0].numpy(), 2)[3:]
    expected = probe(torch.as_tensor(stacked))
    torch.testing.assert_close(decoded[0], expected)


def test_decode_latents_pads_the_episode_start():
    probe = make_probe(feature_dim=3, history=3)
    features = torch.randn(1, 4, 3)

    decoded = planning.decode_latents(probe, features[:, :1], features[:, 1:])

    stacked = stack_history(features[0].numpy(), 3)[1:]
    torch.testing.assert_close(decoded[0], probe(torch.as_tensor(stacked)))


def test_latent_evaluator_scores_reward_and_goal_objectives():
    torch.manual_seed(0)
    config = PredictorConfig(feature_dim=8, history=3, width=16, depth=1, heads=2)
    predictor = LatentPredictor(config).eval()
    probe = make_probe(feature_dim=8, history=2)
    context = torch.randn(2, 2, 8)
    actions = torch.rand(2, 5, 4, 1) * 2 - 1

    reward = planning.latent_evaluator(predictor, probe, context, torch.zeros(2, 1, 1))(
        actions
    )
    goal = planning.latent_evaluator(
        predictor,
        probe,
        context,
        torch.zeros(2, 1, 1),
        objective="goal",
        goal=context[0, -1],
    )(actions)

    assert reward.returns.shape == goal.returns.shape == (2, 5)
    assert reward.extras["observations"].shape == (2, 5, 4, 5)
    assert reward.extras["first_latent"].shape == (2, 5, 8)
    # The zero-initialized residual head predicts persistence, so episode 0
    # stays at its goal latent and the goal objective is exactly zero there.
    torch.testing.assert_close(goal.returns[0], torch.zeros(5))
    assert (goal.returns[1] < 0).all()
    assert ((reward.returns >= 0) & (reward.returns <= 4)).all()


def test_linear_dynamics_matches_the_numpy_model():
    rng = np.random.default_rng(0)
    observations = rng.normal(size=(200, 5))
    observations[:, 1:3] /= np.linalg.norm(observations[:, 1:3], axis=1, keepdims=True)
    actions = rng.uniform(-1, 1, size=(200, 1))
    following = observations + 0.1 * actions + 0.01 * rng.normal(size=(200, 5))
    model = fit_linear_dynamics(observations, actions, following, ridge=1e-6)

    predicted = planning.linear_dynamics(model)(
        torch.as_tensor(observations, dtype=torch.float32),
        torch.as_tensor(actions, dtype=torch.float32),
    )

    np.testing.assert_allclose(
        predicted.numpy(), model.predict_next(observations, actions), atol=1e-4
    )


def test_state_evaluator_prefers_the_action_toward_upright():
    def tilt(observations, actions):
        # A toy pole whose angle follows the action directly.
        theta = (
            torch.atan2(observations[:, 2], observations[:, 1]) + 0.1 * actions[:, 0]
        )
        return torch.stack(
            (
                observations[:, 0],
                torch.cos(theta),
                torch.sin(theta),
                observations[:, 3],
                observations[:, 4],
            ),
            dim=1,
        )

    start = torch.tensor(
        [[0.0, np.cos(0.5), np.sin(0.5), 0.0, 0.0]], dtype=torch.float32
    )
    actions = torch.tensor([[[[-1.0]] * 3, [[1.0]] * 3]])

    evaluation = planning.state_evaluator(tilt, start)(actions)

    assert evaluation.returns.shape == (1, 2)
    assert evaluation.returns[0, 0] > evaluation.returns[0, 1]
    assert evaluation.extras["observations"].shape == (1, 2, 3, 5)


@pytest.mark.parametrize(
    "bad", [{"elites": 0}, {"candidates": 1}, {"action_repeat": 0}]
)
def test_cem_config_rejects_invalid_settings(bad):
    with pytest.raises(ValueError):
        planning.CEMConfig(**bad)

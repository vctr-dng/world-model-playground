"""Tests for latent dynamics predictors on synthetic linear latent systems."""

import numpy as np
import pytest
import torch

from world_model_playground.wm import predictors
from world_model_playground.wm.eval import episode_rollout_rmse, persistence_rollout
from world_model_playground.wm.probes import StateProbe

DEVICE = torch.device("cpu")


def make_episodes(num_episodes: int, length: int, feature_dim: int = 8, seed: int = 0):
    """Features linearly mixing a damped, action-driven 5-dimensional state.

    The system is fixed; ``seed`` only draws the initial states and actions.
    """
    system = np.random.default_rng(1234)
    dynamics = 0.95 * np.linalg.qr(system.normal(size=(5, 5)))[0]
    control = system.normal(size=(1, 5))
    mixing = system.normal(size=(5, feature_dim))
    rng = np.random.default_rng(seed)
    episodes = {}
    for episode_id in range(num_episodes):
        state = rng.normal(size=5)
        actions = rng.uniform(-1.0, 1.0, size=(length, 1))
        states = [state]
        for action in actions:
            state = state @ dynamics + action @ control
            states.append(state)
        states = np.asarray(states)
        episodes[episode_id] = {
            "features": (states @ mixing).astype(np.float32),
            "observations": states.astype(np.float32),
            "actions": actions.astype(np.float32),
        }
    return episodes


def small_config(feature_dim: int = 8, history: int = 3) -> predictors.PredictorConfig:
    return predictors.PredictorConfig(
        feature_dim=feature_dim, history=history, width=32, depth=2, heads=2
    )


def test_untrained_predictor_is_persistence():
    predictor = predictors.LatentPredictor(small_config())
    context = torch.randn(2, 3, 8)
    predicted = predictor.rollout(context, torch.randn(2, 2, 1), torch.randn(2, 5, 1))
    torch.testing.assert_close(predicted, context[:, -1:].expand(2, 5, 8))


def test_predictor_is_causal():
    torch.manual_seed(0)
    predictor = predictors.LatentPredictor(small_config())
    torch.nn.init.normal_(predictor.latent_out.weight)
    latents, actions = torch.randn(1, 3, 8), torch.randn(1, 3, 1)
    changed = latents.clone()
    changed[:, 2] += 1.0
    first = predictor(latents, actions)[:, :2]
    torch.testing.assert_close(predictor(changed, actions)[:, :2], first)
    torch.testing.assert_close(predictor(latents[:, :2], actions[:, :2]), first)
    with pytest.raises(ValueError, match="windows must hold"):
        predictor(torch.randn(1, 4, 8), torch.randn(1, 4, 1))


@pytest.mark.parametrize("history", [1, 3])
def test_rollout_slides_the_window_over_its_own_predictions(history):
    torch.manual_seed(0)
    predictor = predictors.LatentPredictor(small_config(history=history))
    torch.nn.init.normal_(predictor.latent_out.weight, std=0.1)
    context, context_actions = torch.randn(1, 5, 8), torch.randn(1, 4, 1)
    actions = torch.randn(1, 4, 1)

    with torch.no_grad():
        predicted = predictor.rollout_normalized(context, context_actions, actions)
        latents = list(context[0])
        step_actions = list(context_actions[0])
        for step in range(4):
            step_actions.append(actions[0, step])
            window = torch.stack(latents[-history:])[None]
            window_actions = torch.stack(step_actions[-history:])[None]
            latents.append(predictor(window, window_actions)[0, -1])
    torch.testing.assert_close(predicted[0], torch.stack(latents[5:]))
    with pytest.raises(ValueError, match="one action fewer"):
        predictor.rollout_normalized(context, context_actions[:, :3], actions)


def test_latent_windows_stay_within_episodes():
    episodes = make_episodes(num_episodes=2, length=4, feature_dim=2)
    windows = predictors.LatentWindows(episodes, length=3, device=DEVICE)
    # Each episode has 5 steps and therefore 3 windows of 3 steps.
    assert len(windows) == 6
    latents, actions = windows[torch.arange(6)]
    assert latents.shape == (6, 3, 2) and actions.shape == (6, 2, 1)
    np.testing.assert_array_equal(latents[3].numpy(), episodes[1]["features"][:3])
    np.testing.assert_array_equal(actions[5].numpy(), episodes[1]["actions"][2:4])
    with pytest.raises(ValueError, match="full window"):
        predictors.LatentWindows(episodes, length=6, device=DEVICE)


def test_fit_predictor_beats_persistence_and_round_trips(tmp_path):
    train = make_episodes(num_episodes=24, length=40, seed=0)
    validation = make_episodes(num_episodes=4, length=40, seed=1)
    predictor, best_epoch, losses = predictors.fit_predictor(
        train,
        validation,
        config=small_config(),
        rollout_steps=4,
        epochs=30,
        batch_size=64,
        learning_rate=3e-3,
        weight_decay=0.0,
        device=DEVICE,
    )
    assert 1 <= best_epoch <= 30 and len(losses) == 30
    assert losses[-1]["train_one_step"] < losses[0]["train_one_step"]

    for episode in validation.values():
        episode["latents"] = (
            predictor.normalize(torch.from_numpy(episode["features"])).detach().numpy()
        )
    horizons = (1, 4, 10)
    model = episode_rollout_rmse(
        predictors.normalized_latent_rollout(predictor),
        validation,
        target="latents",
        horizons=horizons,
    )
    persistence = episode_rollout_rmse(
        persistence_rollout("latents"), validation, target="latents", horizons=horizons
    )
    for horizon in map(str, horizons):
        assert model[horizon]["overall"] < 0.5 * persistence[horizon]["overall"]

    path = tmp_path / "predictor.pt"
    predictors.save_predictor(path, predictor, {"encoder": "synthetic"})
    restored, metadata = predictors.load_predictor(path)
    assert metadata == {"encoder": "synthetic"}
    assert restored.config == predictor.config
    context = torch.from_numpy(validation[0]["features"][None, :3])
    actions = torch.from_numpy(validation[0]["actions"][None])
    with torch.no_grad():
        torch.testing.assert_close(
            restored.rollout(context, actions[:, :2], actions[:, 2:10]),
            predictor.rollout(context, actions[:, :2], actions[:, 2:10]),
        )


def test_latent_rollout_decodes_predictions_with_the_probe():
    episodes = make_episodes(num_episodes=1, length=12)
    predictor = predictors.LatentPredictor(small_config()).eval()
    probe = StateProbe(16, history=2).eval()
    rollout = predictors.latent_rollout(predictor, probe)

    decoded = rollout(episodes[0], 5, 4)

    # An untrained predictor repeats the start latent, so the probe sees the
    # stacked pair (features[5], features[5]) and then that latent twice.
    features = episodes[0]["features"]
    expected = probe.predict(
        np.stack([np.concatenate((features[5], features[5]))] * 4).astype(np.float32)
    )
    assert decoded.shape == (4, 5)
    np.testing.assert_allclose(decoded, expected, atol=1e-6)
    np.testing.assert_allclose(
        predictors.decode_steps(probe, features, slice(1, 3)),
        probe.predict(np.stack([features[0:2].ravel(), features[1:3].ravel()])),
        atol=1e-6,
    )

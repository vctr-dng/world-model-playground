"""Tests for state probes on synthetic features with a known observation decoding."""

import numpy as np
import pytest
import torch

from world_model_playground.wm import probes

DEVICE = torch.device("cpu")


def synthetic_split(rng: np.random.Generator, count: int, mixing: np.ndarray):
    """Observations with a valid (cos, sin) pair, linearly mixed into features."""
    theta = rng.uniform(-np.pi, np.pi, count)
    observations = np.column_stack(
        (
            rng.normal(size=count),
            np.cos(theta),
            np.sin(theta),
            rng.normal(size=count),
            rng.normal(size=count),
        )
    ).astype(np.float32)
    noise = 1e-3 * rng.normal(size=(count, mixing.shape[1]))
    return (observations @ mixing + noise).astype(np.float32), observations


@pytest.fixture
def splits():
    rng = np.random.default_rng(0)
    mixing = rng.normal(size=(5, 12))
    return [synthetic_split(rng, count, mixing) for count in (600, 200, 200)]


def test_stack_history_concatenates_previous_steps():
    features = np.arange(6, dtype=np.float32).reshape(3, 2)
    np.testing.assert_array_equal(
        probes.stack_history(features, 2), [[0, 1, 0, 1], [0, 1, 2, 3], [2, 3, 4, 5]]
    )


def test_probe_arrays_concatenate_episodes():
    episodes = {
        0: {"features": np.zeros((3, 4)), "observations": np.zeros((3, 5))},
        1: {"features": np.ones((2, 4)), "observations": np.ones((2, 5))},
    }
    features, observations = probes.probe_arrays(episodes, history=3)
    assert features.shape == (5, 12) and observations.shape == (5, 5)
    with pytest.raises(ValueError, match="no episodes"):
        probes.probe_arrays({}, history=1)


def test_regression_metrics_on_perfect_predictions(splits):
    _, observations = splits[2]
    metrics = probes.regression_metrics(observations, observations)
    assert all(value == pytest.approx(1.0) for value in metrics["r2"].values())
    assert metrics["angle_mae_rad"] == pytest.approx(0.0, abs=1e-12)
    assert probes.passes_gate(metrics)


def test_linear_probe_recovers_a_linear_decoding(splits):
    train, validation, test = splits
    probe, ridge = probes.fit_linear_probe(
        train, validation, history=1, ridges=[1e-3, 1e6], device=DEVICE
    )
    assert ridge == 1e-3
    metrics = probes.regression_metrics(probe.predict(test[0]), test[1])
    assert min(metrics["r2"].values()) > 0.999
    assert probe.kind == "linear"


def test_mlp_probe_learns_and_round_trips(tmp_path, splits):
    train, validation, test = splits
    probe, best_epoch = probes.fit_mlp_probe(
        train,
        validation,
        history=1,
        hidden_sizes=(32,),
        epochs=60,
        batch_size=64,
        device=DEVICE,
    )
    assert 1 <= best_epoch <= 60
    predictions = probe.predict(test[0])
    assert min(probes.regression_metrics(predictions, test[1])["r2"].values()) > 0.6

    path = tmp_path / "probe.pt"
    probes.save_probe(path, probe, {"encoder": "synthetic"})
    restored, metadata = probes.load_probe(path)
    assert metadata == {"encoder": "synthetic"}
    assert restored.kind == "mlp" and restored.history == 1
    np.testing.assert_allclose(restored.predict(test[0]), predictions, atol=1e-6)

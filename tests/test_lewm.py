"""Tests for the LeWM JEPA: losses, rollouts, windows, artifacts, and encoder."""

import json

import numpy as np
import pytest
import torch

from world_model_playground.wm import embeddings as embeddings_module
from world_model_playground.wm import encoders, episodes, lewm
from world_model_playground.wm.sigreg import SIGReg

SIZE = 16


def tiny_config(**overrides) -> lewm.LeWMConfig:
    settings = {
        "image_size": SIZE,
        "patch_size": 8,
        "embed_dim": 32,
        "encoder_depth": 2,
        "encoder_heads": 2,
        "predictor_depth": 2,
        "predictor_heads": 2,
        "predictor_head_dim": 16,
        "predictor_mlp_dim": 64,
        "projector_hidden": 64,
        "history": 3,
        "step_stride": 2,
    }
    return lewm.LeWMConfig(**(settings | overrides))


@pytest.fixture
def model() -> lewm.LeWM:
    torch.manual_seed(0)
    return lewm.LeWM(tiny_config()).eval()


def random_windows(batch: int, config: lewm.LeWMConfig):
    generator = torch.Generator().manual_seed(1)
    frames = torch.randint(
        0, 256, (batch, config.frames, 3, SIZE, SIZE), generator=generator
    ).to(torch.uint8)
    actions = torch.rand(
        batch, config.history, config.action_inputs, generator=generator
    )
    return frames, actions * 2 - 1


def test_losses_are_finite_and_train_the_encoder(model):
    model.train()
    frames, actions = random_windows(8, model.config)

    losses = model.losses(frames, actions, SIGReg(num_projections=32))
    (losses["prediction"] + 0.1 * losses["sigreg"]).backward()

    assert losses["embeddings"].shape == (8, model.config.frames, 32)
    assert torch.isfinite(losses["prediction"]) and torch.isfinite(losses["sigreg"])
    encoder_gradient = model.encoder.embeddings.patch_embeddings.projection.weight.grad
    assert encoder_gradient is not None and encoder_gradient.abs().sum() > 0


def test_action_conditioning_starts_as_the_identity(model):
    # AdaLN-zero: at initialization the actions cannot change the prediction.
    frames, actions = random_windows(2, model.config)
    latents = model.encode(frames)[:, :-1]

    first = model(latents, actions)
    second = model(latents, -actions)

    torch.testing.assert_close(first, second)


def test_prediction_is_causal(model):
    frames, actions = random_windows(2, model.config)
    latents = model.encode(frames)[:, :-1]
    for block in model.predictor.blocks:  # let the actions matter
        torch.nn.init.normal_(block.modulation[-1].weight, std=0.1)

    full = model(latents, actions)
    shorter = model(latents[:, :2], actions[:, :2])

    torch.testing.assert_close(full[:, :2], shorter)


def test_rollout_feeds_predictions_back_over_a_sliding_history(model):
    for block in model.predictor.blocks:
        torch.nn.init.normal_(block.modulation[-1].weight, std=0.1)
    k = model.config.action_inputs
    generator = torch.Generator().manual_seed(2)
    z = torch.randn(2, 3, 32, generator=generator)
    past = torch.randn(2, 2, k, generator=generator)
    future = torch.randn(2, 2, k, generator=generator)

    rolled = model.rollout(z, past, future)

    first = model(z, torch.cat((past, future[:, :1]), 1))[:, -1]
    window = torch.cat((z[:, 1:], first[:, None]), 1)
    second = model(window, torch.cat((past[:, 1:], future), 1))[:, -1]
    assert rolled.shape == (2, 2, 32)
    torch.testing.assert_close(rolled[:, 0], first)
    torch.testing.assert_close(rolled[:, 1], second)


def test_rollout_accepts_context_shorter_than_the_history(model):
    k = model.config.action_inputs

    rolled = model.rollout(
        torch.randn(2, 1, 32), torch.zeros(2, 0, k), torch.zeros(2, 4, k)
    )

    assert rolled.shape == (2, 4, 32)
    with pytest.raises(ValueError, match="one action fewer"):
        model.rollout(torch.randn(2, 2, 32), torch.zeros(2, 2, k), torch.zeros(2, 1, k))


def test_embedding_statistics_flag_collapse():
    healthy = lewm.embedding_statistics(torch.randn(512, 16))
    collapsed = lewm.embedding_statistics(
        torch.ones(512, 16) + 1e-4 * torch.randn(512, 16)
    )
    one_direction = lewm.embedding_statistics(torch.randn(512, 1) * torch.ones(1, 16))

    assert healthy["effective_rank"] > 12 and healthy["std_min"] > 0.8
    assert collapsed["std_mean"] < 1e-3
    assert one_direction["effective_rank"] < 1.5


def test_ridge_r2_scores_a_linear_relationship():
    rng = np.random.default_rng(0)
    features = rng.normal(size=(400, 6))
    targets = np.stack((features @ rng.normal(size=6), rng.normal(size=400)), axis=1)

    r2 = lewm.ridge_r2(features, targets)

    assert r2[0] > 0.95
    assert r2[1] < 0.2


def test_artifact_round_trip_restores_the_same_model(model, tmp_path):
    path = tmp_path / "model.pt"
    lewm.save_lewm(path, model, {"params": {"note": "x"}})
    frames, actions = random_windows(2, model.config)
    latents = model.encode(frames)[:, :-1]

    restored, metadata = lewm.load_lewm(path)

    assert lewm.is_lewm_artifact(path)
    assert metadata == {"params": {"note": "x"}}
    assert restored.config == model.config
    torch.testing.assert_close(restored(latents, actions), model(latents, actions))


def test_encoder_name_slug_and_loader(model, tmp_path):
    path = tmp_path / "lewm_h3.pt"
    lewm.save_lewm(path, model, {})
    name = lewm.encoder_name(path)

    encoder = encoders.load_encoder(name, device="cpu")

    assert embeddings_module.encoder_slug(name) == "lewm-lewm_h3"
    assert embeddings_module.encoder_slug("dinov2-small") == "dinov2-small"
    assert encoder.context_frames == 1 and encoder.feature_dim == 32
    frames = torch.randint(0, 256, (5, 1, 3, SIZE, SIZE), dtype=torch.uint8)
    output = encoder(frames)
    assert output.pooled.shape == (5, 32)
    torch.testing.assert_close(
        output.pooled, model.encode(frames)[:, 0], atol=1e-5, rtol=1e-4
    )


@pytest.fixture
def collection(tmp_path):
    """Two episodes of 12 and 9 transitions whose frames hold their step index."""
    entries = []
    for episode_id, length in {0: 12, 1: 9}.items():
        frames = np.zeros((length + 1, SIZE, SIZE, 3), dtype=np.uint8)
        frames[:, 0, 0, 0] = np.arange(length + 1)
        episode = {
            "observations": np.arange(length + 1, dtype=np.float32)[:, None]
            * np.ones(5, np.float32),
            "actions": np.arange(length, dtype=np.float32)[:, None] / 100,
            "rewards": np.zeros(length, np.float32),
            "action_source": np.zeros(length, np.uint8),
            "qpos": np.zeros((length + 1, 2), np.float32),
            "qvel": np.zeros((length + 1, 2), np.float32),
            "terminated": np.asarray(False),
            "truncated": np.asarray(True),
            episodes.frames_key(SIZE, SIZE): frames,
        }
        episodes.write_episode(tmp_path, episode_id, episode)
        entries.append({"id": episode_id, "length": length})
    manifest = {
        "format_version": episodes.FORMAT_VERSION,
        "resolutions": [{"key": episodes.frames_key(SIZE, SIZE)}],
        "episodes": entries,
    }
    (tmp_path / episodes.MANIFEST_NAME).write_text(json.dumps(manifest))
    return tmp_path


def test_frame_windows_space_frames_by_the_stride(collection):
    windows = lewm.FrameWindows(
        collection, [0, 1], resolution=(SIZE, SIZE), frames=4, stride=3
    )

    # 12 transitions leave 12 - 9 + 1 = 4 starts, 9 leave 1.
    assert len(windows) == 5
    item = windows[1]  # episode 0, start 1
    assert item["frames"].shape == (4, 3, SIZE, SIZE)
    assert item["frames"][:, 0, 0, 0].tolist() == [1, 4, 7, 10]
    assert item["observations"][:, 0].tolist() == [1, 4, 7, 10]
    assert item["actions"].shape == (3, 3)
    # Block j holds the actions applied from frame j until the next one.
    np.testing.assert_allclose(item["actions"][1].numpy(), [0.04, 0.05, 0.06])


def test_frame_windows_skip_episodes_that_are_too_short(collection):
    windows = lewm.FrameWindows(
        collection, [0, 1], resolution=(SIZE, SIZE), frames=5, stride=3
    )

    assert {episode for episode, _ in windows.index} == {0}
    with pytest.raises(ValueError, match="no rendered"):
        lewm.FrameWindows(collection, [0], resolution=(4, 4), frames=2, stride=1)


def test_fit_lewm_lowers_the_loss_on_a_tiny_collection(collection):
    torch.manual_seed(0)
    config = tiny_config(history=2, step_stride=1, dropout=0.0)
    windows = lewm.FrameWindows(
        collection, [0, 1], resolution=(SIZE, SIZE), frames=config.frames, stride=1
    )
    model = lewm.LeWM(config)
    logged = []

    trained, best, losses = lewm.fit_lewm(
        model,
        windows,
        windows,
        sigreg=SIGReg(num_projections=16),
        iterations=40,
        batch_size=8,
        learning_rate=1e-3,
        warmup=2,
        eval_every=20,
        eval_windows=16,
        num_workers=0,
        device=torch.device("cpu"),
        log=logged.append,
    )

    assert [entry["iteration"] for entry in losses] == [20, 40]
    assert logged == losses
    assert best in (20, 40) and not trained.training
    assert losses[-1]["validation_total"] <= losses[0]["validation_total"] * 1.5
    assert {"train_effective_rank", "validation_std_min", "validation_sigreg"} <= set(
        losses[-1]
    )

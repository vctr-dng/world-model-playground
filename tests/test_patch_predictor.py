"""Tests for the patch-token predictor and its window loader."""

import numpy as np
import pytest
import torch

from world_model_playground.wm import patch_predictor as pp
from world_model_playground.wm import token_cache, world_models
from world_model_playground.wm.probes import StateProbe
from world_model_playground.wm.token_cache import TokenStore


def tiny_config(**overrides) -> pp.PatchPredictorConfig:
    values = {
        "token_dim": 6,
        "pool_side": 4,
        "step_stride": 2,
        "history": 3,
        "width": 16,
        "depth": 2,
        "heads": 2,
        "summary_side": 2,
    }
    return pp.PatchPredictorConfig(**{**values, **overrides})


def trained_like(config: pp.PatchPredictorConfig) -> pp.PatchPredictor:
    """A predictor whose residual head is non-zero, so attention matters."""
    torch.manual_seed(0)
    model = pp.PatchPredictor(config)
    torch.nn.init.normal_(model.latent_out.weight, std=0.1)
    return model.eval()


def test_zero_initialized_predictor_is_persistence():
    model = pp.PatchPredictor(tiny_config()).eval()
    latents = torch.randn(2, 3, 16, 6)
    actions = torch.randn(2, 3, 2)
    torch.testing.assert_close(model(latents, actions), latents)


def test_attention_is_causal_across_frames():
    model = trained_like(tiny_config())
    latents = torch.randn(2, 3, 16, 6)
    actions = torch.randn(2, 3, 2)
    changed = latents.clone()
    changed[:, 2] += 1.0  # only the last frame changes
    before, after = model(latents, actions), model(changed, actions)
    torch.testing.assert_close(before[:, :2], after[:, :2])
    assert not torch.allclose(before[:, 2], after[:, 2])
    # Within a frame every token sees every other token of that frame.
    first = latents.clone()
    first[:, 0, 5] += 1.0
    moved = model(first, actions)
    assert not torch.allclose(moved[:, 0, 0], before[:, 0, 0])


def test_actions_condition_every_token_of_their_frame_and_later_frames():
    model = trained_like(tiny_config())
    latents = torch.randn(1, 3, 16, 6)
    actions = torch.randn(1, 3, 2)
    changed = actions.clone()
    changed[:, 1] += 1.0
    before, after = model(latents, actions), model(latents, changed)
    torch.testing.assert_close(before[:, 0], after[:, 0])
    assert not torch.allclose(before[:, 1], after[:, 1])


def test_rollout_matches_forward_and_keeps_shapes():
    config = tiny_config()
    model = trained_like(config)
    latents = torch.randn(2, 3, 16, 6)
    actions = torch.randn(2, 5, 2)
    one = model.rollout_normalized(latents, actions[:, :2], actions[:, 2:3])
    expected = model(latents, torch.cat((actions[:, :2], actions[:, 2:3]), 1))
    torch.testing.assert_close(one[:, 0], expected[:, -1])
    long = model.rollout_normalized(latents, actions[:, :2], actions[:, 2:])
    assert long.shape == (2, 3, 16, 6)
    short = model.rollout_normalized(latents[:, -1:], actions[:, :0], actions)
    assert short.shape == (2, 5, 16, 6)
    with pytest.raises(ValueError, match="one action fewer"):
        model.rollout_normalized(latents, actions[:, :3], actions)


def test_normalize_pools_and_summary_is_grid_independent():
    model = pp.PatchPredictor(tiny_config(pool_side=2, summary_side=2))
    model.set_normalization(np.full(6, 2.0, np.float32), np.full(6, 4.0, np.float32))
    raw = torch.randn(3, 16, 6)
    normalized = model.normalize(raw)
    assert normalized.shape == (3, 4, 6)
    torch.testing.assert_close(
        model.denormalize(normalized),
        token_cache.pool_tokens(raw, 2),
        atol=1e-5,
        rtol=1e-5,
    )
    # Raw and pooled grids give the same summary, which probes rely on.
    torch.testing.assert_close(
        model.pooled(raw),
        model.pooled(token_cache.pool_tokens(raw, 2)),
        atol=1e-6,
        rtol=1e-5,
    )
    assert model.pooled(raw).shape == (3, 4 * 6)


def test_config_validation():
    with pytest.raises(ValueError, match="multiple of heads"):
        tiny_config(width=15)
    with pytest.raises(ValueError, match="multiple of summary_side"):
        tiny_config(pool_side=3, summary_side=2)


def fake_store(lengths, tokens_per_step=16, dim=6) -> TokenStore:
    steps = [length + 1 for length in lengths]
    offsets = np.concatenate(([0], np.cumsum(steps)[:-1]))
    tokens, actions = [], []
    for episode, length in enumerate(lengths):
        block = np.zeros((length + 1, tokens_per_step, dim), np.float16)
        block[:, :, 0] = np.arange(length + 1)[:, None]  # step marker
        block[:, :, 1] = episode
        tokens.append(block)
        # The action taken from step t equals 100 * episode + t.
        taken = 100.0 * episode + np.arange(length + 1, dtype=np.float32)
        taken[-1] = 0.0
        actions.append(taken[:, None])
    return TokenStore(
        tokens=np.concatenate(tokens),
        episode_ids=np.arange(len(lengths)),
        offsets=offsets,
        lengths=np.asarray(steps),
        observations=np.zeros((sum(steps), 5), np.float32),
        actions=np.concatenate(actions),
    )


def test_token_windows_align_frames_and_grouped_actions():
    store = fake_store([9, 7])
    windows = pp.TokenWindows(
        store,
        [1, 0],  # selection order must not matter
        length=3,
        stride=2,
        pool_side=2,
        device=torch.device("cpu"),
    )
    # 8 steps (episode 1) -> 8 - 5 + 1 = 4 windows; 10 steps -> 6 windows.
    assert len(windows) == 10
    tokens, actions = windows[torch.arange(len(windows))]
    assert tokens.shape == (10, 3, 4, 6) and actions.shape == (10, 2, 2)
    # Frames of a window are stride 2 apart within one episode.
    for window in range(10):
        marker = tokens[window, :, 0, 0]
        assert torch.equal(marker, marker[0] + 2 * torch.arange(3))
        assert (tokens[window, :, 0, 1] == tokens[window, 0, 0, 1]).all()
        episode, first = int(tokens[window, 0, 0, 1]), int(marker[0])
        expected = 100 * episode + first + torch.arange(4).float()
        assert torch.equal(actions[window].flatten(), expected)


def test_token_windows_statistics_and_batch_dtype():
    store = fake_store([9])
    windows = pp.TokenWindows(
        store, [0], length=3, stride=1, pool_side=4, device=torch.device("cpu")
    )
    mean, std = windows.channel_statistics()
    assert mean.shape == std.shape == (6,)
    assert mean[0] == pytest.approx(4.5) and mean[1] == pytest.approx(0.0)
    assert windows[torch.arange(2)][0].dtype == torch.float32


def test_patch_losses_and_artifact_roundtrip(tmp_path):
    config = tiny_config()
    model = trained_like(config)
    tokens = torch.randn(4, 3 + 2, 16, 6)
    actions = torch.randn(4, 4, 2)
    one_step, open_loop = pp.patch_losses(model, tokens, actions, 2)
    assert one_step.ndim == open_loop.ndim == 0 and open_loop > 0
    with pytest.raises(ValueError, match="history \\+ rollout_steps"):
        pp.patch_losses(model, tokens, actions, 3)

    path = tmp_path / "model.pt"
    pp.save_patch_predictor(path, model, {"note": "x"})
    assert pp.is_patch_artifact(path)
    dispatched, _ = world_models.load_world_model(path)
    assert isinstance(dispatched, pp.PatchPredictor)
    restored, metadata = pp.load_patch_predictor(path)
    assert metadata == {"note": "x"} and restored.config == config
    latents, step_actions = torch.randn(1, 3, 16, 6), torch.randn(1, 3, 2)
    torch.testing.assert_close(
        restored(latents, step_actions), model(latents, step_actions)
    )


def test_tiny_model_can_overfit_one_batch():
    torch.manual_seed(0)
    model = pp.PatchPredictor(tiny_config(width=32))
    # Latents move by a fixed shift per unit action, so actions must be used.
    base = torch.randn(8, 1, 16, 6)
    actions = torch.randn(8, 4, 2)
    steps = torch.cumsum(actions.sum(-1, keepdim=True)[..., None] * 0.5, dim=1)
    tokens = torch.cat((base, base + steps), dim=1)  # (8, 5, 16, 6)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    first = last = None
    for step in range(150):
        one_step, open_loop = pp.patch_losses(model, tokens, actions, 2)
        loss = one_step + open_loop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        first = float(loss.detach()) if step == 0 else first
        last = float(loss.detach())
    assert last < 0.5 * first


def rollout_fixture(probe_stride: int):
    model = trained_like(tiny_config())  # stride 2, history 3, summary 2x2
    probe = StateProbe(2 * 4 * 6, history=2, stride=probe_stride)
    rng = np.random.default_rng(0)
    tokens = rng.normal(size=(21, 16, 6)).astype(np.float32)
    episode = {
        "tokens": tokens,
        "features": model.pooled(torch.from_numpy(tokens)).numpy(),
        "actions": rng.uniform(-1, 1, size=(20, 1)).astype(np.float32),
    }
    return model, probe, episode


def test_patch_rollouts_hold_each_model_step_over_its_control_steps():
    model, probe, episode = rollout_fixture(probe_stride=2)
    rollout = pp.patch_rollouts(model, probe)
    predictions = rollout(episode, 3, 7)
    assert predictions.shape == (7, 5)
    # Four model steps of two control steps cover seven requested steps.
    np.testing.assert_array_equal(predictions[0], predictions[1])
    np.testing.assert_array_equal(predictions[2], predictions[3])
    assert not np.allclose(predictions[1], predictions[2])
    # Starting at step 0 repeats the first step as context.
    assert rollout(episode, 0, 4).shape == (4, 5)


def test_patch_rollouts_require_a_probe_stacking_model_steps():
    model, probe, _ = rollout_fixture(probe_stride=1)
    with pytest.raises(ValueError, match="step_stride"):
        pp.patch_rollouts(model, probe)

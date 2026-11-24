"""Tests for the V-JEPA 2-AC adapter, on tiny random-weight predictors."""

import zipfile

import numpy as np
import pytest
import torch
from transformers import VJEPA2Config, VJEPA2Model

from world_model_playground.wm import ac_checkpoint, ac_world_model, encoders
from world_model_playground.wm.embeddings import write_embedded_episode
from world_model_playground.wm.episodes import episode_path, step_windows
from world_model_playground.wm.planning import (
    CEMConfig,
    CEMPlanner,
    group_step_actions,
    latent_evaluator,
)
from world_model_playground.wm.probes import StateProbe
from world_model_playground.wm.vendor.vjepa2 import vit_ac_predictor

# 32x32 frames of 16x16 patches: 4 tokens of 16 channels per step, at most 4 steps.
TINY_PREDICTOR = {
    "img_size": (32, 32),
    "patch_size": 16,
    "num_frames": 8,
    "tubelet_size": 2,
    "embed_dim": 16,
    # Head dimension 24: upstream RoPE needs at least 12 (the released model has 64).
    "predictor_embed_dim": 48,
    "depth": 2,
    "num_heads": 2,
}
TOKENS, CHANNELS = 4, 16


def tiny_config(**overrides) -> ac_world_model.ACConfig:
    values = {
        "step_stride": 2,
        "history": 3,
        "pretrained": False,
        "predictor": TINY_PREDICTOR,
    } | overrides
    return ac_world_model.ACConfig(**values)


def tiny_model(**overrides) -> ac_world_model.ACWorldModel:
    torch.manual_seed(0)
    return ac_world_model.build_ac_model(tiny_config(**overrides)).eval()


def random_latents(batch: int, steps: int, seed: int = 1) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(batch, steps, TOKENS, CHANNELS, generator=generator)


def test_action_encoder_reads_grouped_forces_from_pretrained_columns():
    torch.manual_seed(0)
    predictor = vit_ac_predictor(**TINY_PREDICTOR)
    pretrained = predictor.action_encoder.weight.detach().clone()
    model = ac_world_model.ACWorldModel(tiny_config(step_stride=3), predictor)
    assert model.predictor.action_encoder.in_features == 3
    torch.testing.assert_close(model.predictor.action_encoder.weight, pretrained[:, :3])
    # More forces than the 7-D pretrained input start from a fresh layer.
    wide = ac_world_model.ACWorldModel(
        tiny_config(step_stride=8), vit_ac_predictor(**TINY_PREDICTOR)
    )
    assert wide.predictor.action_encoder.in_features == 8


def test_forward_predicts_layer_normalized_tokens_for_every_step():
    model = tiny_model()
    latents = model.normalize(random_latents(2, 3))
    predicted = model(latents, torch.randn(2, 3, 2))
    assert predicted.shape == (2, 3, TOKENS, CHANNELS)
    torch.testing.assert_close(
        predicted.mean(-1), torch.zeros(2, 3, TOKENS), atol=1e-5, rtol=0
    )
    with pytest.raises(ValueError, match="1 to 4 steps"):
        model(model.normalize(random_latents(1, 5)), torch.zeros(1, 5, 2))


def test_predictions_are_causal_over_steps():
    model = tiny_model()
    latents = model.normalize(random_latents(1, 3))
    actions = torch.randn(1, 3, 2)
    reference = model(latents, actions)
    changed_latents = latents.clone()
    changed_latents[:, 2] = model.normalize(random_latents(1, 1, seed=7))[:, 0]
    changed_actions = actions.clone()
    changed_actions[:, 2] = 5.0
    changed = model(changed_latents, changed_actions)
    torch.testing.assert_close(changed[:, :2], reference[:, :2])
    assert not torch.allclose(changed[:, 2], reference[:, 2])


def test_actions_change_the_prediction():
    model = tiny_model()
    latents = model.normalize(random_latents(1, 2))
    pushed = model(latents, torch.ones(1, 2, 2))
    pulled = model(latents, -torch.ones(1, 2, 2))
    assert not torch.allclose(pushed, pulled)


def test_rollout_feeds_predictions_back_with_a_sliding_history():
    model = tiny_model(history=2)
    context = model.normalize(random_latents(2, 3))
    context_actions = torch.randn(2, 2, 2)
    actions = torch.randn(2, 3, 2)
    predicted = model.rollout_normalized(context, context_actions, actions)
    assert predicted.shape == (2, 3, TOKENS, CHANNELS)
    # Step 1 sees the last two context steps; step 2 the last one and step 1.
    first = model(
        context[:, -2:], torch.cat((context_actions[:, -1:], actions[:, :1]), 1)
    )
    torch.testing.assert_close(predicted[:, 0], first[:, -1])
    window = torch.stack((context[:, -1], predicted[:, 0]), 1)
    second = model(window, actions[:, :2])
    torch.testing.assert_close(predicted[:, 1], second[:, -1])


def test_losses_backpropagate_to_the_new_action_layer():
    model = tiny_model().train()
    latents = random_latents(2, 5)
    actions = torch.randn(2, 4, 2)
    one_step, open_loop = ac_world_model.ac_losses(model, latents, actions, 2)
    (one_step + open_loop).backward()
    assert model.predictor.action_encoder.weight.grad.abs().sum() > 0
    with pytest.raises(ValueError, match="history \\+ rollout_steps"):
        ac_world_model.ac_losses(model, latents[:, :4], actions[:, :3], 2)


def test_freeze_blocks_freezes_the_input_side():
    model = tiny_model(freeze_blocks=1)
    blocks = model.predictor.predictor_blocks
    assert not any(p.requires_grad for p in blocks[0].parameters())
    assert all(p.requires_grad for p in blocks[1].parameters())
    assert model.predictor.action_encoder.weight.requires_grad


def test_group_actions_pads_with_the_last_action():
    actions = np.arange(5, dtype=np.float32)[:, None]
    grouped = ac_world_model.group_actions(actions, [0, 3], 3)
    np.testing.assert_array_equal(grouped, [[0, 1, 2], [3, 4, 4]])
    assert ac_world_model.group_actions(actions, [], 3).shape == (0, 3)


def write_token_episodes(directory, lengths, seed=0):
    rng = np.random.default_rng(seed)
    episodes = {}
    for episode_id, length in enumerate(lengths):
        tokens = rng.normal(size=(length + 1, TOKENS, CHANNELS)).astype(np.float16)
        episode = {
            "observations": rng.normal(size=(length + 1, 5)).astype(np.float32),
            "actions": rng.uniform(-1, 1, size=(length, 1)).astype(np.float32),
            "rewards": np.zeros(length, dtype=np.float32),
            "terminated": np.asarray(False),
            "truncated": np.asarray(True),
        }
        features = tokens.astype(np.float32).mean(1)
        write_embedded_episode(directory, episode_id, episode, features, tokens)
        episodes[episode_id] = episode | {"tokens": tokens, "features": features}
    return episodes


def test_npz_member_memmap_reads_uncompressed_members(tmp_path):
    episodes = write_token_episodes(tmp_path, [6])
    tokens = ac_world_model.npz_member_memmap(episode_path(tmp_path, 0), "tokens")
    np.testing.assert_array_equal(tokens, episodes[0]["tokens"])
    compressed = tmp_path / "compressed.npz"
    np.savez_compressed(compressed, tokens=episodes[0]["tokens"])
    with pytest.raises(ValueError, match="compressed"):
        ac_world_model.npz_member_memmap(compressed, "tokens")
    with zipfile.ZipFile(episode_path(tmp_path, 0)) as archive:
        assert archive.getinfo("tokens.npy").compress_type == zipfile.ZIP_STORED


def test_token_windows_space_steps_by_the_stride(tmp_path):
    episodes = write_token_episodes(tmp_path, [9, 4])
    windows = ac_world_model.TokenWindows(tmp_path, [0, 1], length=3, stride=2)
    # Episode 0 has 10 steps: starts 0..5 cover 3 steps 2 apart; episode 1 has 5.
    assert len(windows) == 6 + 1
    tokens, actions = windows[4]
    np.testing.assert_array_equal(tokens.numpy(), episodes[0]["tokens"][[4, 6, 8]])
    np.testing.assert_array_equal(
        actions.numpy(), episodes[0]["actions"][4:8, 0].reshape(2, 2)
    )
    sparse = ac_world_model.TokenWindows(
        tmp_path, [0], length=3, stride=2, start_every=2
    )
    assert sparse.index[:, 1].tolist() == [0, 2, 4]


def test_fit_runs_and_keeps_the_best_validation_weights(tmp_path):
    write_token_episodes(tmp_path, [12, 12, 12])
    model = tiny_model(history=2)
    train = ac_world_model.TokenWindows(tmp_path, [0, 1], length=3, stride=2)
    validation = ac_world_model.TokenWindows(tmp_path, [2], length=3, stride=2)
    fitted, best, losses = ac_world_model.fit_ac_model(
        model,
        train,
        validation,
        rollout_steps=1,
        iterations=6,
        batch_size=4,
        warmup=2,
        eval_every=3,
        num_workers=0,
        device=torch.device("cpu"),
        log=lambda message: None,
    )
    assert best in (3, 6) and [r["iteration"] for r in losses] == [3, 6]
    assert not fitted.training


def test_save_and_load_round_trip(tmp_path):
    model = tiny_model()
    path = tmp_path / "ac.pt"
    ac_world_model.save_ac_model(path, model, {"params": {"probe": "p"}})
    assert ac_world_model.is_ac_artifact(path)
    loaded, metadata = ac_world_model.load_ac_model(path)
    assert metadata["params"]["probe"] == "p"
    assert loaded.config == model.config
    latents = model.normalize(random_latents(1, 2))
    actions = torch.randn(1, 2, 2)
    # Saved in bfloat16: outputs agree to bfloat16 precision.
    torch.testing.assert_close(
        loaded(latents, actions), model(latents, actions), atol=0.1, rtol=0.05
    )


def test_hf_encoder_state_dict_inverts_the_fused_attention_layout():
    config = VJEPA2Config(
        patch_size=16,
        crop_size=32,
        frames_per_clip=2,
        hidden_size=16,
        num_attention_heads=2,
        num_hidden_layers=2,
        pred_hidden_size=16,
        pred_num_attention_heads=2,
        pred_num_hidden_layers=1,
    )
    hf_state = VJEPA2Model(config).encoder.state_dict()
    upstream = {}
    for key, value in hf_state.items():
        if ".attention.query." in key:
            kind = key.rsplit(".", 1)[1]
            prefix = key.split(".attention.")[0]
            index = prefix.split(".")[1]
            fused = torch.cat(
                [
                    hf_state[f"{prefix}.attention.{n}.{kind}"]
                    for n in ("query", "key", "value")
                ]
            )
            upstream[f"module.backbone.blocks.{index}.attn.qkv.{kind}"] = fused
        elif ".attention.key." in key or ".attention.value." in key:
            continue
        else:
            renamed = (
                key.replace("embeddings.patch_embeddings.proj.", "patch_embed.proj.")
                .replace("layernorm.", "norm.")
                .replace("layer.", "blocks.")
                .replace("attention.proj.", "attn.proj.")
            )
            upstream[f"module.backbone.{renamed}"] = value
    mapped = ac_checkpoint.hf_encoder_state_dict(upstream)
    assert mapped.keys() == hf_state.keys()
    for key, value in hf_state.items():
        torch.testing.assert_close(mapped[key], value)


def test_ac_encoder_spec_repeats_frames_and_normalizes_tokens():
    spec = encoders.ENCODERS["vjepa2-ac-vitg"]
    tiny = encoders.EncoderSpec(
        "tiny-ac",
        "vjepa2",
        "none",
        32,
        1,
        repeat_frames=2,
        layer_norm_tokens=True,
    )
    assert spec.repeat_frames == 2 and spec.layer_norm_tokens
    config = VJEPA2Config(
        patch_size=16,
        crop_size=32,
        frames_per_clip=2,
        hidden_size=16,
        num_attention_heads=2,
        num_hidden_layers=1,
        pred_hidden_size=16,
        pred_num_attention_heads=2,
        pred_num_hidden_layers=1,
    )
    encoder = encoders.VJepa2Encoder(tiny, VJEPA2Model(config).encoder, 1)
    frames = torch.randint(0, 255, (2, 1, 3, 32, 32), dtype=torch.uint8)
    output = encoder(frames)
    assert output.tokens.shape == (2, 4, 16)  # one tubelet of the repeated frame
    torch.testing.assert_close(
        output.tokens.mean(-1), torch.zeros(2, 4), atol=1e-5, rtol=0
    )
    torch.testing.assert_close(output.pooled, output.tokens.mean(1))


def test_strided_step_windows():
    windows = step_windows(6, 3, stride=2)
    assert windows[5].tolist() == [1, 3, 5]
    assert windows[2].tolist() == [0, 0, 2]
    np.testing.assert_array_equal(step_windows(4, 2), step_windows(4, 2, stride=1))


def test_group_step_actions_requires_whole_model_steps():
    actions = torch.arange(6.0).view(1, 6, 1)
    assert group_step_actions(actions, 3).tolist() == [[[0, 1, 2], [3, 4, 5]]]
    with pytest.raises(ValueError, match="multiple"):
        group_step_actions(actions, 4)


def tiny_probe(stride: int) -> StateProbe:
    torch.manual_seed(3)
    return StateProbe(2 * CHANNELS, history=2, stride=stride).eval()


@pytest.mark.parametrize("objective", ["reward", "goal"])
def test_latent_evaluator_scores_token_models_per_model_step(objective):
    model = tiny_model(step_stride=2, history=2)
    context = random_latents(2, 2)
    goal = random_latents(1, 1, seed=9)[0, 0]
    evaluate = latent_evaluator(
        model,
        tiny_probe(stride=2),
        context,
        torch.randn(2, 1, 2),
        objective=objective,
        goal=goal,
    )
    actions = torch.rand(2, 3, 6, 1) * 2 - 1
    evaluation = evaluate(actions)
    assert evaluation.returns.shape == (2, 3)
    assert evaluation.extras["observations"].shape == (2, 3, 3, 5)
    assert evaluation.extras["first_latent"].shape == (2, 3, TOKENS, CHANNELS)
    if objective == "goal":
        assert (evaluation.returns <= 0).all()
    with pytest.raises(ValueError, match="stride"):
        latent_evaluator(model, tiny_probe(stride=1), context, torch.randn(2, 1, 2))


@pytest.mark.parametrize("objective", ["reward", "goal"])
@pytest.mark.parametrize("chunk", [1, 4, 5, 100])
def test_latent_evaluator_chunks_rollouts_without_changing_results(objective, chunk):
    model = tiny_model(step_stride=2, history=2)
    context = random_latents(3, 2)
    context_actions = torch.randn(3, 1, 2)
    goal = random_latents(1, 1, seed=9)[0, 0]
    actions = torch.rand(3, 4, 6, 1) * 2 - 1

    def evaluate(rollout_chunk):
        return latent_evaluator(
            model,
            tiny_probe(stride=2),
            context,
            context_actions,
            objective=objective,
            goal=goal,
            rollout_chunk=rollout_chunk,
        )(actions)

    whole, chunked = evaluate(None), evaluate(chunk)
    torch.testing.assert_close(chunked.returns, whole.returns)
    for name in whole.extras:
        torch.testing.assert_close(chunked.extras[name], whole.extras[name])


def test_cem_warm_start_advances_by_the_replanning_shift():
    config = CEMConfig(horizon=6, candidates=8, iterations=1, elites=2, action_repeat=2)
    planner = CEMPlanner(config, batch_size=1, shift=2, seed=0)
    knots = torch.tensor([[[0.1], [0.2], [0.3]]])
    shifted = planner._shifted(knots)
    torch.testing.assert_close(shifted, torch.tensor([[[0.2], [0.3], [0.3]]]))


def test_ac_rollouts_hold_each_model_step_over_its_control_steps(tmp_path):
    episodes = write_token_episodes(tmp_path, [12])
    episode = {
        "tokens": ac_world_model.npz_member_memmap(episode_path(tmp_path, 0), "tokens"),
        "features": episodes[0]["features"],
        "actions": episodes[0]["actions"],
    }
    model = tiny_model(step_stride=2, history=2)
    observations, latents = ac_world_model.ac_rollouts(model, tiny_probe(stride=2))
    predicted = observations(episode, 4, 5)
    assert predicted.shape == (5, 5)
    np.testing.assert_array_equal(predicted[0], predicted[1])
    np.testing.assert_array_equal(predicted[2], predicted[3])
    assert latents(episode, 4, 5).shape == (5, CHANNELS)
    with pytest.raises(ValueError, match="stride"):
        ac_world_model.ac_rollouts(model, tiny_probe(stride=1))

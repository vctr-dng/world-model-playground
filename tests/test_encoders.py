"""Tests for frozen encoder wrappers and the embedding cache, on tiny random models."""

import numpy as np
import pytest
import torch
from transformers import Dinov2Config, Dinov2Model, VJEPA2Config, VJEPA2Model

from world_model_playground.wm import embeddings, encoders, episodes


def tiny_dinov2() -> encoders.Dinov2Encoder:
    spec = encoders.EncoderSpec("tiny-dinov2", "dinov2", "none", 28, 1)
    config = Dinov2Config(
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        image_size=28,
        patch_size=14,
    )
    return encoders.Dinov2Encoder(spec, Dinov2Model(config))


def tiny_vjepa2(context_frames: int = 2) -> encoders.VJepa2Encoder:
    spec = encoders.EncoderSpec("tiny-vjepa2", "vjepa2", "none", 32, 2)
    config = VJEPA2Config(
        patch_size=16,
        crop_size=32,
        frames_per_clip=4,
        hidden_size=16,
        num_attention_heads=2,
        num_hidden_layers=1,
        pred_hidden_size=16,
        pred_num_attention_heads=2,
        pred_num_hidden_layers=1,
    )
    return encoders.VJepa2Encoder(spec, VJEPA2Model(config).encoder, context_frames)


def test_preprocess_frames_resizes_and_normalizes():
    frames = torch.full((2, 3, 3, 40, 40), 255, dtype=torch.uint8)
    pixels = encoders.preprocess_frames(frames, 28)
    assert pixels.shape == (2, 3, 3, 28, 28)
    expected = (1.0 - torch.tensor(encoders.IMAGENET_MEAN)) / torch.tensor(
        encoders.IMAGENET_STD
    )
    torch.testing.assert_close(pixels[0, 0, :, 0, 0], expected)


def test_dinov2_encoder_shapes():
    encoder = tiny_dinov2()
    output = encoder(torch.zeros((3, 1, 3, 64, 64), dtype=torch.uint8))
    assert output.pooled.shape == (3, encoder.feature_dim) == (3, 32)
    assert output.tokens.shape == (3, 4, 16)
    assert not any(p.requires_grad for p in encoder.parameters())


def test_dinov2_rejects_multi_frame_windows():
    with pytest.raises(ValueError, match="single frames"):
        encoders.Dinov2Encoder(tiny_dinov2().spec, tiny_dinov2().backbone, 2)


def test_vjepa2_encoder_shapes_and_tubelet_check():
    encoder = tiny_vjepa2(context_frames=4)
    output = encoder(torch.zeros((2, 4, 3, 32, 32), dtype=torch.uint8))
    assert output.pooled.shape == (2, 16)
    assert output.tokens.shape == (2, 2 * 4, 16)  # two tubelets of 2x2 patches
    with pytest.raises(ValueError, match="multiple of 2"):
        tiny_vjepa2(context_frames=3)
    with pytest.raises(ValueError, match="shape"):
        encoder(torch.zeros((2, 2, 3, 32, 32), dtype=torch.uint8))


def test_load_encoder_rejects_unknown_names():
    with pytest.raises(ValueError, match="unknown encoder"):
        encoders.load_encoder("resnet")


def test_step_windows_pad_with_the_first_step():
    windows = episodes.step_windows(4, 3)
    np.testing.assert_array_equal(windows, [[0, 0, 0], [0, 0, 1], [0, 1, 2], [1, 2, 3]])


class WindowSpy(torch.nn.Module):
    """Stands in for an encoder: features are the step ids encoded in the frames."""

    context_frames = 2

    def forward(self, frames):
        steps = frames[:, :, 0, 0, 0].float()
        return encoders.EncoderOutput(steps, steps[:, :, None].repeat(1, 1, 3))


def test_embed_frames_encodes_the_window_ending_at_each_step():
    frames = np.zeros((5, 4, 4, 3), dtype=np.uint8)
    frames[:, 0, 0, 0] = np.arange(5)
    features, tokens = embeddings.embed_frames(
        WindowSpy(), frames, batch_size=2, keep_tokens=True
    )
    np.testing.assert_array_equal(features, [[0, 0], [0, 1], [1, 2], [2, 3], [3, 4]])
    assert features.dtype == np.float32
    assert tokens.shape == (5, 2, 3) and tokens.dtype == np.float16


def test_embedded_episodes_round_trip(tmp_path):
    episode = {
        "observations": np.ones((4, 5), np.float32),
        "actions": np.zeros((3, 1), np.float32),
        "rewards": np.zeros(3, np.float32),
        "terminated": np.asarray(False),
        "truncated": np.asarray(True),
    }
    features = np.arange(8, dtype=np.float32).reshape(4, 2)
    embeddings.write_embedded_episode(tmp_path, 7, episode, features)
    embeddings.write_embeddings_manifest(
        tmp_path, {"encoder": "spy", "episodes": [{"id": 7, "length": 3}]}
    )
    loaded = embeddings.load_embedded_episodes(tmp_path)
    assert list(loaded) == [7]
    np.testing.assert_array_equal(loaded[7]["features"], features)
    assert embeddings.load_embeddings_manifest(tmp_path)["format_version"] == 1
    with pytest.raises(ValueError, match="one row per step"):
        embeddings.write_embedded_episode(tmp_path, 8, episode, features[:3])

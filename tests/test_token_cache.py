"""Tests for the consolidated patch-token store."""

import numpy as np
import pytest
import torch

from world_model_playground.wm import embeddings, token_cache


def write_fake_embeddings(directory, lengths, tokens_per_step=16, dim=4):
    rng = np.random.default_rng(0)
    entries = []
    for episode_id, length in enumerate(lengths):
        episode = {
            "observations": rng.normal(size=(length + 1, 5)).astype(np.float32),
            "actions": rng.uniform(-1, 1, size=(length, 1)).astype(np.float32),
            "rewards": np.zeros(length, np.float32),
            "terminated": np.asarray(False),
            "truncated": np.asarray(True),
        }
        tokens = np.full((length + 1, tokens_per_step, dim), episode_id, np.float16)
        tokens[:, 0, 0] = np.arange(length + 1)  # a step marker
        embeddings.write_embedded_episode(
            directory,
            episode_id,
            episode,
            np.zeros((length + 1, dim), np.float32),
            tokens,
        )
        entries.append({"id": episode_id, "length": length})
    embeddings.write_embeddings_manifest(
        directory, {"stores_tokens": True, "episodes": entries}
    )


def test_consolidate_aligns_tokens_observations_and_actions(tmp_path):
    write_fake_embeddings(tmp_path, [5, 3])
    token_cache.consolidate_tokens(tmp_path)
    store = token_cache.open_token_store(tmp_path)

    assert store.tokens.shape == (6 + 4, 16, 4)
    second = store.episode_slice(1)
    assert second == slice(6, 10)
    assert np.all(store.tokens[second][:, 1, 1] == 1)
    np.testing.assert_array_equal(store.tokens[second][:, 0, 0], np.arange(4))
    # The last step of each episode has no action and a zero pad.
    assert store.actions[5, 0] == 0 and store.actions[9, 0] == 0
    first = embeddings.load_embedded_episodes(
        tmp_path, [0], fields=("actions", "observations")
    )[0]
    np.testing.assert_array_equal(store.actions[:5], first["actions"])
    np.testing.assert_array_equal(store.observations[:6], first["observations"])


def test_window_starts_stay_inside_episodes(tmp_path):
    write_fake_embeddings(tmp_path, [5, 3])
    token_cache.consolidate_tokens(tmp_path)
    store = token_cache.open_token_store(tmp_path)

    np.testing.assert_array_equal(store.window_starts(3), [0, 1, 2, 3, 6, 7])
    np.testing.assert_array_equal(store.window_starts(3, episode_ids=[1]), [6, 7])
    assert len(store.window_starts(5, episode_ids=[1])) == 0
    # Stride 2 windows of 3 model steps span 5 steps: 2 starts in a 6-step episode.
    np.testing.assert_array_equal(store.window_starts(3, 2), [0, 1])


def test_consolidate_refuses_existing_cache_and_missing_tokens(tmp_path):
    write_fake_embeddings(tmp_path, [3])
    token_cache.consolidate_tokens(tmp_path)
    with pytest.raises(FileExistsError):
        token_cache.consolidate_tokens(tmp_path)
    token_cache.consolidate_tokens(tmp_path, overwrite=True)

    embeddings.write_embeddings_manifest(
        tmp_path, {"stores_tokens": False, "episodes": [{"id": 0, "length": 3}]}
    )
    with pytest.raises(ValueError, match="store-tokens"):
        token_cache.consolidate_tokens(tmp_path, overwrite=True)


def test_pool_tokens_averages_grid_cells():
    grid = torch.arange(16.0).reshape(1, 1, 16, 1).expand(2, 3, 16, 2)
    pooled = token_cache.pool_tokens(grid, 2)
    assert pooled.shape == (2, 3, 4, 2)
    # The top-left 2x2 cell of a row-major 4x4 grid holds 0, 1, 4, 5.
    assert pooled[0, 0, 0, 0] == pytest.approx(2.5)
    assert token_cache.pool_tokens(grid, 4) is grid
    with pytest.raises(ValueError, match="cannot pool"):
        token_cache.pool_tokens(grid, 3)
    with pytest.raises(ValueError, match="square grid"):
        token_cache.pool_tokens(torch.zeros(1, 15, 2), 3)

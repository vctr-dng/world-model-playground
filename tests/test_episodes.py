"""Tests for episodic collections, their DreamerV3 view, and the clip dataset."""

import json

import numpy as np
import pytest

from world_model_playground.wm import data, datasets, dreamer, episodes


def make_episode(length: int, *, seed: int, terminated: bool = False) -> dict:
    """Builds a valid episode whose frames encode their step index."""
    rng = np.random.default_rng(seed)
    frames = np.zeros((length + 1, 4, 6, 3), dtype=np.uint8)
    frames[:, 0, 0, 0] = np.arange(length + 1)
    frames[:, 0, 0, 1] = seed
    return {
        "observations": rng.normal(size=(length + 1, 5)).astype(np.float32),
        "actions": rng.uniform(-1, 1, size=(length, 1)).astype(np.float32),
        "rewards": rng.uniform(size=length).astype(np.float32),
        "action_source": rng.integers(0, 3, size=length).astype(np.uint8),
        "qpos": rng.normal(size=(length + 1, 2)).astype(np.float32),
        "qvel": rng.normal(size=(length + 1, 2)).astype(np.float32),
        "terminated": np.asarray(terminated),
        "truncated": np.asarray(not terminated),
        episodes.frames_key(4, 6): frames,
    }


@pytest.fixture
def collection(tmp_path):
    """Writes a three-episode collection with a format-2 manifest."""
    lengths = {0: 5, 1: 3, 2: 7}
    entries = []
    for episode_id, length in lengths.items():
        episode = make_episode(length, seed=episode_id, terminated=episode_id == 1)
        assert episodes.write_episode(tmp_path, episode_id, episode) == length
        entries.append({"id": episode_id, "length": length})
    manifest = {
        "format_version": episodes.FORMAT_VERSION,
        "resolutions": [{"key": "frames_4x6", "height": 4, "width": 6}],
        "episodes": entries,
    }
    (tmp_path / episodes.MANIFEST_NAME).write_text(json.dumps(manifest))
    return tmp_path


def test_write_and_load_round_trip_selected_fields(tmp_path):
    episode = make_episode(4, seed=0)
    episodes.write_episode(tmp_path, 3, episode)

    loaded = episodes.load_episode(tmp_path, 3, fields=("actions", "frames_4x6"))

    assert set(loaded) == {"actions", "frames_4x6"}
    np.testing.assert_array_equal(loaded["actions"], episode["actions"])
    np.testing.assert_array_equal(loaded["frames_4x6"], episode["frames_4x6"])


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("observations", np.zeros((4, 5)), "observations must have shape"),
        ("qpos", np.zeros((3, 2)), "qpos must have shape"),
        ("frames_4x6", np.zeros((5, 4, 6, 3), dtype=np.float32), "uint8"),
        ("frames_4x6", np.zeros((5, 8, 6, 3), dtype=np.uint8), "does not match"),
        ("rewards", np.full(4, np.nan, dtype=np.float32), "finite"),
    ],
)
def test_validate_episode_rejects_malformed_fields(field, value, message):
    episode = make_episode(4, seed=0) | {field: value}

    with pytest.raises(ValueError, match=message):
        episodes.validate_episode(episode)


def test_split_batched_episodes_cuts_at_first_done():
    batch, horizon = 2, 4
    initial = {
        "observations": np.full((batch, 5), -1.0),
        "qpos": np.full((batch, 2), -1.0),
        "qvel": np.full((batch, 2), -1.0),
    }
    steps = {
        "observations": np.arange(batch * horizon * 5, dtype=float).reshape(
            batch, horizon, 5
        ),
        "qpos": np.zeros((batch, horizon, 2)),
        "qvel": np.zeros((batch, horizon, 2)),
        "actions": np.zeros((batch, horizon, 1)),
        "rewards": np.ones((batch, horizon)),
        "dones": np.array([[0, 0, 0, 0], [0, 1, 1, 0]], dtype=float),
    }
    sources = np.array([[0, 1, 2, 0], [2, 2, 1, 1]], dtype=np.uint8)

    full, cut = episodes.split_batched_episodes(initial, steps, sources)

    assert episodes.validate_episode(full) == 4
    assert bool(full["truncated"]) and not bool(full["terminated"])
    assert episodes.validate_episode(cut) == 2
    assert bool(cut["terminated"]) and not bool(cut["truncated"])
    np.testing.assert_array_equal(cut["observations"][0], initial["observations"][1])
    np.testing.assert_array_equal(cut["observations"][1:], steps["observations"][1, :2])
    np.testing.assert_array_equal(cut["action_source"], [2, 2])


def test_episodes_to_transitions_matches_the_flat_format(collection):
    loaded = [(i, episodes.load_episode(collection, i)) for i in (0, 1, 2)]

    transitions = episodes.episodes_to_transitions(loaded)

    assert data.validate_transitions(transitions) == 15
    np.testing.assert_array_equal(
        transitions["episode_ids"], np.repeat([0, 1, 2], [5, 3, 7])
    )
    np.testing.assert_array_equal(transitions["timesteps"][5:8], [0, 1, 2])
    np.testing.assert_array_equal(
        transitions["next_observations"][5:8], loaded[1][1]["observations"][1:]
    )
    assert np.flatnonzero(transitions["dones"]).tolist() == [4, 7, 14]
    assert np.flatnonzero(transitions["terminated"]).tolist() == [7]
    assert np.flatnonzero(transitions["truncated"]).tolist() == [4, 14]


def test_transition_frames_index_like_the_flat_image_arrays(collection):
    loaded = [episodes.load_episode(collection, i)["frames_4x6"] for i in (0, 1, 2)]
    images = np.concatenate([frames[:-1] for frames in loaded])
    next_images = np.concatenate([frames[1:] for frames in loaded])

    view = episodes.TransitionFrames(collection, "frames_4x6", cache_size=1)
    next_view = episodes.TransitionFrames(collection, "frames_4x6", following=True)

    assert view.shape == images.shape and len(view) == 15
    np.testing.assert_array_equal(view[6], images[6])
    np.testing.assert_array_equal(view[np.array([14, 0, 7])], images[[14, 0, 7]])
    np.testing.assert_array_equal(next_view[3:9], next_images[3:9])
    assert view[np.array([], dtype=int)].shape == (0, 4, 6, 3)


@pytest.mark.parametrize("alignment", dreamer.ACTION_ALIGNMENTS)
def test_to_dreamer_episode_aligns_steps(alignment):
    episode = make_episode(3, seed=4, terminated=True)

    steps = dreamer.to_dreamer_episode(
        episode, image_key="frames_4x6", action_alignment=alignment
    )

    assert {len(values) for values in steps.values()} == {4}
    np.testing.assert_array_equal(steps["image"], episode["frames_4x6"])
    np.testing.assert_array_equal(steps["reward"][1:], episode["rewards"])
    assert steps["reward"][0] == 0.0
    assert steps["is_first"].tolist() == [True, False, False, False]
    assert steps["is_last"].tolist() == [False, False, False, True]
    assert steps["is_terminal"].tolist() == [False, False, False, True]
    assert steps["discount"].tolist() == [1.0, 1.0, 1.0, 0.0]
    if alignment == "preceding":
        np.testing.assert_array_equal(steps["action"][1:], episode["actions"])
        assert steps["action"][0].tolist() == [0.0]
    else:
        np.testing.assert_array_equal(steps["action"][:-1], episode["actions"])
        assert steps["action"][-1].tolist() == [0.0]


def test_to_dreamer_episode_marks_truncation_as_non_terminal():
    steps = dreamer.to_dreamer_episode(
        make_episode(2, seed=0, terminated=False), image_key="frames_4x6"
    )

    assert not steps["is_terminal"].any()
    assert steps["is_last"][-1]


def test_clip_dataset_cuts_aligned_clips(collection):
    dataset = datasets.EpisodeClipDataset(
        collection, resolution=(4, 6), clip_length=4, episode_ids=[0, 1]
    )

    # Episode 0 has 6 frames (3 clip starts), episode 1 has 4 frames (1 start).
    assert len(dataset) == 4
    clip = dataset[2]
    episode = episodes.load_episode(collection, 0)
    assert clip["frames"].shape == (4, 3, 4, 6)
    assert clip["frames"][:, 0, 0, 0].tolist() == [2, 3, 4, 5]
    np.testing.assert_array_equal(clip["actions"].numpy(), episode["actions"][2:5])
    np.testing.assert_array_equal(
        clip["observations"].numpy(), episode["observations"][2:6]
    )
    assert clip["rewards"].shape == (3,)


def test_clip_dataset_rejects_missing_resolution(collection):
    with pytest.raises(ValueError, match="frames_64x64"):
        datasets.EpisodeClipDataset(collection, resolution=(64, 64), clip_length=2)

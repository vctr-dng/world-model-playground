"""Episode-sharded CartPole collections (format version 2).

A collection directory holds ``collection.json`` (the manifest), one compressed
``episodes/episode_<id>.npz`` per episode, and a flat ``transitions.npz`` for
the state-only trainers. Each episode of ``T`` transitions stores:

- ``observations`` ``(T+1, 5)`` float32, including the final observation
- ``actions`` ``(T, 1)``, ``rewards`` ``(T,)``, ``action_source`` ``(T,)``
- ``qpos`` ``(T+1, nq)`` and ``qvel`` ``(T+1, nv)``, enough to re-render frames
- ``frames_<H>x<W>`` ``(T+1, H, W, 3)`` uint8, one array per rendered resolution
- ``terminated`` and ``truncated`` boolean scalars for the episode end

``actions[t]`` is applied to the state shown in ``frames[t]`` and leads to
``frames[t + 1]``; ``rewards[t]`` is received on that transition.
"""

import json
from collections import OrderedDict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import numpy as np

from world_model_playground.wm.data import ACTION_SIZE, OBSERVATION_SIZE

FORMAT_VERSION = 2
EPISODES_DIRNAME = "episodes"
MANIFEST_NAME = "collection.json"


def frames_key(height: int, width: int) -> str:
    """Names the episode array holding frames rendered at ``height`` x ``width``."""
    return f"frames_{height}x{width}"


def episode_path(collection_dir: Path, episode_id: int) -> Path:
    return collection_dir / EPISODES_DIRNAME / f"episode_{episode_id:06d}.npz"


def validate_episode(episode: Mapping[str, np.ndarray]) -> int:
    """Checks an episode's fields against the format and returns its length ``T``."""
    required = (
        "observations",
        "actions",
        "rewards",
        "action_source",
        "qpos",
        "qvel",
        "terminated",
        "truncated",
    )
    missing = [name for name in required if name not in episode]
    if missing:
        raise ValueError(f"episode is missing fields: {', '.join(missing)}")

    length = len(episode["actions"])
    if length < 1:
        raise ValueError("an episode needs at least one transition")
    expected_shapes = {
        "observations": (length + 1, OBSERVATION_SIZE),
        "actions": (length, ACTION_SIZE),
        "rewards": (length,),
        "action_source": (length,),
        "terminated": (),
        "truncated": (),
    }
    for name, shape in expected_shapes.items():
        if np.shape(episode[name]) != shape:
            raise ValueError(f"{name} must have shape {shape}")
    for name in ("qpos", "qvel"):
        if np.ndim(episode[name]) != 2 or len(episode[name]) != length + 1:
            raise ValueError(f"{name} must have shape (T+1, n)")
    for name, frames in episode.items():
        if name.startswith("frames_"):
            if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[3] != 3:
                raise ValueError(f"{name} must be uint8 with shape (T+1, H, W, 3)")
            height, width = frames.shape[1:3]
            if name != frames_key(height, width) or len(frames) != length + 1:
                raise ValueError(f"{name} does not match its (T+1, H, W, 3) shape")
    if not all(
        np.isfinite(episode[name]).all()
        for name in ("observations", "actions", "rewards", "qpos", "qvel")
    ):
        raise ValueError("episode values must be finite")
    return length


def write_episode(
    collection_dir: Path, episode_id: int, episode: Mapping[str, np.ndarray]
) -> int:
    """Validates and writes one episode shard, returning its length."""
    length = validate_episode(episode)
    path = episode_path(collection_dir, episode_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **episode)
    return length


def load_manifest(collection_dir: Path) -> dict:
    return json.loads((collection_dir / MANIFEST_NAME).read_text(encoding="utf-8"))


def is_episodic(manifest: Mapping) -> bool:
    """Whether a manifest describes an episode-sharded (version 2+) collection."""
    return manifest.get("format_version", 1) >= FORMAT_VERSION


def episode_lengths(manifest: Mapping) -> dict[int, int]:
    """Maps each episode id of an episodic collection to its transition count."""
    return {int(entry["id"]): int(entry["length"]) for entry in manifest["episodes"]}


def load_episode(
    collection_dir: Path, episode_id: int, *, fields: Sequence[str] | None = None
) -> dict[str, np.ndarray]:
    """Reads an episode shard, decompressing only ``fields`` when given."""
    with np.load(episode_path(collection_dir, episode_id), allow_pickle=False) as npz:
        names = npz.files if fields is None else fields
        return {name: npz[name] for name in names}


def episodes_to_transitions(
    episodes: Iterable[tuple[int, Mapping[str, np.ndarray]]],
) -> dict[str, np.ndarray]:
    """Flattens episodes, in the given order, into the state-only transition arrays.

    The result has the fields read by ``wm.data.validate_transitions`` plus the
    reward, episode-boundary, and action-source columns of a collection.
    """
    columns = {
        name: []
        for name in (
            "observations",
            "actions",
            "next_observations",
            "rewards",
            "dones",
            "terminated",
            "truncated",
            "episode_ids",
            "timesteps",
            "action_source",
        )
    }
    for episode_id, episode in episodes:
        length = validate_episode(episode)
        last = np.arange(length) == length - 1
        columns["observations"].append(episode["observations"][:-1])
        columns["actions"].append(episode["actions"])
        columns["next_observations"].append(episode["observations"][1:])
        columns["rewards"].append(episode["rewards"])
        columns["terminated"].append(last & bool(episode["terminated"]))
        columns["truncated"].append(last & bool(episode["truncated"]))
        columns["dones"].append(last)
        columns["episode_ids"].append(np.full(length, episode_id))
        columns["timesteps"].append(np.arange(length))
        columns["action_source"].append(episode["action_source"])
    if not columns["observations"]:
        raise ValueError("cannot flatten an empty list of episodes")

    dtypes = {
        "observations": np.float32,
        "actions": np.float32,
        "next_observations": np.float32,
        "rewards": np.float32,
        "dones": np.bool_,
        "terminated": np.bool_,
        "truncated": np.bool_,
        "episode_ids": np.int32,
        "timesteps": np.int32,
        "action_source": np.uint8,
    }
    return {
        name: np.concatenate(values).astype(dtypes[name])
        for name, values in columns.items()
    }


def step_windows(num_steps: int, context: int) -> np.ndarray:
    """Indices ``(num_steps, context)`` of the steps ending at each step.

    Row ``t`` is ``t-context+1 .. t``; steps before the episode start repeat
    step 0, so every step of an episode, including the first, gets a window.
    """
    if num_steps < 1 or context < 1:
        raise ValueError("num_steps and context must be at least 1")
    offsets = np.arange(1 - context, 1)
    return np.maximum(np.arange(num_steps)[:, None] + offsets, 0)


def split_batched_episodes(
    initial: dict[str, np.ndarray],
    steps: dict[str, np.ndarray],
    sources: np.ndarray,
) -> list[dict[str, np.ndarray]]:
    """Cuts each row of a batched rollout at its first ``done`` into an episode.

    ``initial`` holds ``(B, ...)`` arrays for the reset states and ``steps``
    ``(B, T, ...)`` arrays for the states after each action (``observations``,
    ``qpos``, ``qvel``) plus ``actions``, ``rewards``, and ``dones``. Batched
    environments keep stepping after termination, so everything past the first
    terminal transition is discarded.
    """
    episodes = []
    horizon = sources.shape[1]
    for index in range(len(sources)):
        dones = np.asarray(steps["dones"][index], dtype=bool)
        terminated = bool(dones.any())
        length = int(np.argmax(dones)) + 1 if terminated else horizon
        episode = {
            name: np.concatenate(
                (initial[name][index][None], steps[name][index][:length])
            ).astype(np.float32)
            for name in ("observations", "qpos", "qvel")
        }
        episode |= {
            "actions": steps["actions"][index][:length].astype(np.float32),
            "rewards": steps["rewards"][index][:length].astype(np.float32),
            "action_source": sources[index][:length].astype(np.uint8),
            "terminated": np.asarray(terminated),
            "truncated": np.asarray(not terminated),
        }
        episodes.append(episode)
    return episodes


class EpisodeCache:
    """Keeps the most recently used episode shards' requested fields in memory."""

    def __init__(self, collection_dir: Path, fields: Sequence[str], size: int = 8):
        self.collection_dir = collection_dir
        self.fields = tuple(fields)
        self.size = size
        self._episodes: OrderedDict[int, dict[str, np.ndarray]] = OrderedDict()

    def __call__(self, episode_id: int) -> dict[str, np.ndarray]:
        if episode_id in self._episodes:
            self._episodes.move_to_end(episode_id)
        else:
            self._episodes[episode_id] = load_episode(
                self.collection_dir, episode_id, fields=self.fields
            )
            if len(self._episodes) > self.size:
                self._episodes.popitem(last=False)
        return self._episodes[episode_id]


class TransitionFrames:
    """Array-like ``(N, H, W, 3)`` view of an episodic collection's frames.

    Row ``i`` is the frame before (or, with ``following=True``, after) the
    ``i``-th row of ``transitions.npz``, which lists episodes in manifest order,
    so it stands in for the ``images`` and ``next_images`` arrays of format-1
    collections without loading every frame.
    """

    def __init__(
        self,
        collection_dir: Path,
        key: str,
        *,
        following: bool = False,
        cache_size: int = 8,
    ):
        manifest = load_manifest(collection_dir)
        if not is_episodic(manifest):
            raise ValueError(f"{collection_dir} is not an episodic collection")
        resolution = next(
            (entry for entry in manifest["resolutions"] if entry["key"] == key), None
        )
        if resolution is None:
            raise ValueError(f"{collection_dir} has no rendered {key}")
        lengths = episode_lengths(manifest)
        self._episode_ids = np.repeat(
            np.fromiter(lengths, dtype=np.int64), list(lengths.values())
        )
        self._timesteps = np.concatenate([np.arange(n) for n in lengths.values()])
        self._offset = int(following)
        self._key = key
        self._cache = EpisodeCache(collection_dir, (key,), cache_size)
        self.shape = (
            len(self._episode_ids),
            resolution["height"],
            resolution["width"],
            3,
        )
        self.dtype = np.dtype(np.uint8)

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, index):
        rows = np.arange(len(self))[index]
        if np.ndim(rows) == 0:
            return self._frame(int(rows))
        frames = np.empty((len(rows), *self.shape[1:]), dtype=self.dtype)
        for position, row in enumerate(rows):
            frames[position] = self._frame(int(row))
        return frames

    def _frame(self, row: int) -> np.ndarray:
        episode = self._cache(int(self._episode_ids[row]))
        return episode[self._key][self._timesteps[row] + self._offset]

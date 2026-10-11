"""PyTorch datasets over episodic CartPole collections."""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from world_model_playground.wm.episodes import (
    EpisodeCache,
    episode_lengths,
    frames_key,
    is_episodic,
    load_manifest,
)


class EpisodeClipDataset(Dataset):
    """Fixed-length clips of frames, observations, and actions from episode shards.

    A clip of ``clip_length`` steps starting at ``t`` holds ``frames`` and
    ``observations`` for steps ``t .. t+L-1`` and the ``L-1`` ``actions`` and
    ``rewards`` between them. Frames are uint8 ``(L, 3, H, W)``; scaling and
    model-specific normalization are left to the consumer.
    """

    def __init__(
        self,
        collection_dir: Path,
        *,
        resolution: tuple[int, int],
        clip_length: int,
        episode_ids: Sequence[int] | np.ndarray | None = None,
        stride: int = 1,
        cache_size: int = 8,
    ):
        if clip_length < 2:
            raise ValueError("clip_length must be at least 2 to contain an action")
        if stride < 1:
            raise ValueError("stride must be at least 1")
        manifest = load_manifest(collection_dir)
        if not is_episodic(manifest):
            raise ValueError(f"{collection_dir} is not an episodic collection")
        self.frames_key = frames_key(*resolution)
        if self.frames_key not in {entry["key"] for entry in manifest["resolutions"]}:
            raise ValueError(f"{collection_dir} has no rendered {self.frames_key}")

        lengths = episode_lengths(manifest)
        selected = lengths if episode_ids is None else [int(i) for i in episode_ids]
        self.clip_length = clip_length
        # An episode of T transitions has T+1 frames, so T+2-L clip starts.
        self.index = [
            (episode_id, start)
            for episode_id in selected
            for start in range(0, lengths[episode_id] + 2 - clip_length, stride)
        ]
        self._cache = EpisodeCache(
            collection_dir,
            (self.frames_key, "observations", "actions", "rewards"),
            cache_size,
        )

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        episode_id, start = self.index[item]
        episode = self._cache(episode_id)
        steps = slice(start, start + self.clip_length)
        transitions = slice(start, start + self.clip_length - 1)
        frames = torch.from_numpy(episode[self.frames_key][steps])
        return {
            "frames": frames.permute(0, 3, 1, 2).contiguous(),
            "observations": torch.from_numpy(episode["observations"][steps]),
            "actions": torch.from_numpy(episode["actions"][transitions]),
            "rewards": torch.from_numpy(episode["rewards"][transitions]),
            "episode_id": torch.tensor(episode_id),
            "start": torch.tensor(start),
        }

"""Cached frozen-encoder features of episodic collections.

An embeddings directory mirrors a collection: ``embeddings.json`` (the
manifest) and one ``episodes/episode_<id>.npz`` per episode, readable with
``wm.episodes.load_episode``. Each episode of ``T`` transitions stores:

- ``features`` ``(T+1, D)`` float32, the pooled feature of the window ending at
  each step (see ``wm.encoders``)
- ``tokens`` ``(T+1, N, C)`` float16, the patch tokens, only when requested
- ``observations``, ``actions``, ``rewards``, ``terminated``, ``truncated``
  copied from the collection, so probes and latent dynamics models train from
  the embeddings alone
"""

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import torch

from world_model_playground.wm.encoders import FrozenEncoder
from world_model_playground.wm.episodes import (
    episode_lengths,
    episode_path,
    load_episode,
    step_windows,
)

FORMAT_VERSION = 1
MANIFEST_NAME = "embeddings.json"
COPIED_FIELDS = ("observations", "actions", "rewards", "terminated", "truncated")


def default_embeddings_dir(
    collection_id: str, encoder_name: str, context_frames: int
) -> Path:
    return (
        Path("data/embeddings") / collection_id / f"{encoder_name}_ctx{context_frames}"
    )


def embed_frames(
    encoder: FrozenEncoder,
    frames: np.ndarray,
    *,
    batch_size: int,
    keep_tokens: bool = False,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Encodes the window ending at every step of uint8 ``(T+1, H, W, 3)`` frames."""
    windows = step_windows(len(frames), encoder.context_frames)
    features = []
    tokens = []
    for start in range(0, len(windows), batch_size):
        batch = torch.from_numpy(frames[windows[start : start + batch_size]])
        output = encoder(batch.permute(0, 1, 4, 2, 3))
        features.append(output.pooled.cpu().numpy())
        if keep_tokens:
            tokens.append(output.tokens.half().cpu().numpy())
    return (
        np.concatenate(features).astype(np.float32),
        np.concatenate(tokens) if keep_tokens else None,
    )


def write_embedded_episode(
    embeddings_dir: Path,
    episode_id: int,
    episode: Mapping[str, np.ndarray],
    features: np.ndarray,
    tokens: np.ndarray | None = None,
) -> None:
    """Writes one episode's features with the collection fields probes need."""
    if len(features) != len(episode["observations"]):
        raise ValueError("features must have one row per step (T+1)")
    arrays = {name: np.asarray(episode[name]) for name in COPIED_FIELDS}
    arrays["features"] = features
    if tokens is not None:
        arrays["tokens"] = tokens
    path = episode_path(embeddings_dir, episode_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)


def write_embeddings_manifest(embeddings_dir: Path, manifest: Mapping) -> None:
    (embeddings_dir / MANIFEST_NAME).write_text(
        json.dumps({"format_version": FORMAT_VERSION, **manifest}, indent=2) + "\n",
        encoding="utf-8",
    )


def load_embeddings_manifest(embeddings_dir: Path) -> dict:
    return json.loads((embeddings_dir / MANIFEST_NAME).read_text(encoding="utf-8"))


def load_embedded_episodes(
    embeddings_dir: Path,
    episode_ids: Sequence[int] | np.ndarray | None = None,
    *,
    fields: Sequence[str] = ("features", "observations", "actions"),
) -> dict[int, dict[str, np.ndarray]]:
    """Reads the requested fields of episodes (all of them by default)."""
    if episode_ids is None:
        episode_ids = list(episode_lengths(load_embeddings_manifest(embeddings_dir)))
    return {
        int(i): load_episode(embeddings_dir, int(i), fields=fields) for i in episode_ids
    }

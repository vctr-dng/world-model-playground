"""Flat, memory-mappable patch-token store for patch-level world models.

``embed.py --store-tokens`` writes one ``episodes/episode_<id>.npz`` per episode.
An ``npz`` cannot be memory-mapped, and training samples random windows across
all episodes, so :func:`consolidate_tokens` copies the tokens once into
``<embeddings_dir>/tokens_flat/``:

- ``tokens.npy`` ``(S, N, C)`` float16, every step of every episode back to back
  (``S`` is the sum of ``T+1`` over episodes), opened with ``np.load(mmap_mode="r")``
- ``index.npz`` with per-episode ``episode_ids``, ``offsets`` (first step in the
  flat arrays), and ``lengths`` (``T+1``), and per-step ``observations`` ``(S, 5)``
  and ``actions`` ``(S, A)`` -- the action taken *from* each step, zero on the
  last step of an episode, which has none
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from world_model_playground.wm.embeddings import load_embeddings_manifest
from world_model_playground.wm.episodes import episode_lengths, load_episode

DIRNAME = "tokens_flat"
TOKENS_NAME = "tokens.npy"
INDEX_NAME = "index.npz"


def consolidate_tokens(embeddings_dir: Path, *, overwrite: bool = False) -> Path:
    """Copies the stored tokens of every episode into one flat memmap."""
    manifest = load_embeddings_manifest(embeddings_dir)
    if not manifest.get("stores_tokens"):
        raise ValueError(f"{embeddings_dir} was embedded without --store-tokens")
    output_dir = embeddings_dir / DIRNAME
    if output_dir.exists() and not overwrite:
        raise FileExistsError(f"Token cache already exists: {output_dir}")
    output_dir.mkdir(exist_ok=True)

    lengths = episode_lengths(manifest)
    episode_ids = np.asarray(list(lengths), dtype=np.int64)
    steps = np.asarray([lengths[int(i)] + 1 for i in episode_ids], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(steps)[:-1]))

    first = load_episode(embeddings_dir, int(episode_ids[0]), fields=("tokens",))
    token_shape = first["tokens"].shape[1:]
    tokens = np.lib.format.open_memmap(
        output_dir / TOKENS_NAME,
        mode="w+",
        dtype=np.float16,
        shape=(int(steps.sum()), *token_shape),
    )
    observations, actions = [], []
    for episode_id, offset, length in zip(episode_ids, offsets, steps, strict=True):
        episode = load_episode(
            embeddings_dir,
            int(episode_id),
            fields=("tokens", "observations", "actions"),
        )
        if episode["tokens"].shape != (length, *token_shape):
            raise ValueError(f"episode {episode_id} tokens have an unexpected shape")
        tokens[offset : offset + length] = episode["tokens"]
        observations.append(episode["observations"])
        # The final step has no action; a zero keeps step i aligned with its action.
        actions.append(np.concatenate((episode["actions"], episode["actions"][:1] * 0)))
    tokens.flush()
    np.savez(
        output_dir / INDEX_NAME,
        episode_ids=episode_ids,
        offsets=offsets,
        lengths=steps,
        observations=np.concatenate(observations).astype(np.float32),
        actions=np.concatenate(actions).astype(np.float32),
    )
    (output_dir / "cache.json").write_text(
        json.dumps({"token_shape": list(token_shape), "steps": int(steps.sum())})
        + "\n",
        encoding="utf-8",
    )
    return output_dir


@dataclass(frozen=True)
class TokenStore:
    """Read-only view of a consolidated token cache."""

    tokens: np.ndarray  # (S, N, C) float16 memmap
    episode_ids: np.ndarray
    offsets: np.ndarray
    lengths: np.ndarray
    observations: np.ndarray  # (S, 5)
    actions: np.ndarray  # (S, A), action taken from each step

    @property
    def num_tokens(self) -> int:
        return int(self.tokens.shape[1])

    @property
    def token_dim(self) -> int:
        return int(self.tokens.shape[2])

    def episode_slice(self, episode_id: int) -> slice:
        row = int(np.flatnonzero(self.episode_ids == episode_id)[0])
        start = int(self.offsets[row])
        return slice(start, start + int(self.lengths[row]))

    def window_starts(
        self, length: int, stride: int = 1, episode_ids=None
    ) -> np.ndarray:
        """Flat index of the first step of every window of ``length`` model steps.

        Consecutive model steps of a window are ``stride`` steps apart, so a
        window spans ``(length - 1) * stride + 1`` steps of one episode.
        """
        if length < 1 or stride < 1:
            raise ValueError("length and stride must be at least 1")
        span = (length - 1) * stride + 1
        wanted = None if episode_ids is None else {int(i) for i in episode_ids}
        starts = [
            int(offset) + np.arange(max(0, int(steps) - span + 1))
            for episode_id, offset, steps in zip(
                self.episode_ids, self.offsets, self.lengths, strict=True
            )
            if wanted is None or int(episode_id) in wanted
        ]
        return np.concatenate(starts) if starts else np.zeros(0, dtype=np.int64)


def open_token_store(embeddings_dir: Path) -> TokenStore:
    cache_dir = embeddings_dir / DIRNAME
    with np.load(cache_dir / INDEX_NAME) as index:
        arrays = {name: index[name] for name in index.files}
    return TokenStore(tokens=np.load(cache_dir / TOKENS_NAME, mmap_mode="r"), **arrays)


def grid_side(num_tokens: int) -> int:
    """Side of the square patch grid, which every encoder here produces."""
    side = round(num_tokens**0.5)
    if side * side != num_tokens:
        raise ValueError(f"{num_tokens} tokens do not form a square grid")
    return side


def pool_tokens(tokens: torch.Tensor, side: int) -> torch.Tensor:
    """Average-pools row-major grid tokens ``(..., N, C)`` to ``(..., side**2, C)``."""
    num_tokens = tokens.shape[-2]
    source = grid_side(num_tokens)
    if side == source:
        return tokens
    if side < 1 or source % side:
        raise ValueError(f"cannot pool a {source}x{source} grid to {side}x{side}")
    leading = tokens.shape[:-2]
    grid = tokens.reshape(-1, source, source, tokens.shape[-1]).permute(0, 3, 1, 2)
    pooled = F.avg_pool2d(grid.float(), source // side).to(tokens.dtype)
    return pooled.flatten(2).transpose(1, 2).reshape(*leading, side * side, -1)


def summary_features(
    store: TokenStore,
    episode_id: int,
    side: int,
    device: torch.device | str = "cpu",
    chunk: int = 256,
) -> np.ndarray:
    """Spatial summary ``(T+1, side**2 * C)`` float32 of an episode's token grids.

    The grid is average-pooled to ``side`` x ``side`` and flattened: what
    ``PatchPredictor.pooled`` returns, so probes fitted on it decode predictions.
    """
    steps = store.episode_slice(episode_id)
    features = []
    for start in range(steps.start, steps.stop, chunk):
        block = torch.from_numpy(store.tokens[start : min(start + chunk, steps.stop)])
        pooled = pool_tokens(block.to(device).float(), side)
        features.append(pooled.flatten(1).cpu().numpy())
    return np.concatenate(features)

"""Action-conditioned dynamics on frozen patch tokens (DINO-WM style).

The encoder stays frozen and its patch tokens are cached by ``embed.py
--store-tokens`` (see ``wm.token_cache``). A transformer reads the token grids of
the last ``history`` model steps and predicts the next token grid as a residual in
channel-normalized token space. Attention is block-causal: a token sees every
token of its own frame and of earlier frames, never of later ones. The action
taken from a frame is embedded and added to all of that frame's tokens.

One model step spans ``step_stride`` control steps; the forces applied over the
span are grouped into one ``step_stride * action_dim`` action. Tokens can be
average-pooled to a coarser grid (``pool_side``) to make planning affordable:
attention cost grows with the square of the tokens per sequence.

The model implements the latent world-model interface of ``wm.planning``.
``pooled`` is a fixed spatial summary (the grid pooled to ``summary_side`` x
``summary_side``, flattened), which probes decode back to observations; the mean
token alone loses where things are.
"""

import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from world_model_playground.wm.ac_world_model import group_actions
from world_model_playground.wm.data import ACTION_SIZE
from world_model_playground.wm.eval import EpisodeRollout
from world_model_playground.wm.planning import decode_latents
from world_model_playground.wm.token_cache import (
    TokenStore,
    grid_side,
    pool_tokens,
    summary_features,
)


@dataclass(frozen=True)
class PatchPredictorConfig:
    token_dim: int
    pool_side: int  # side of the token grid the model reads and predicts
    action_dim: int = ACTION_SIZE
    step_stride: int = 5
    history: int = 3
    width: int = 384
    depth: int = 6
    heads: int = 6
    dropout: float = 0.0
    summary_side: int = 4

    def __post_init__(self):
        if self.history < 1 or self.step_stride < 1:
            raise ValueError("history and step_stride must be at least 1")
        if self.width % self.heads:
            raise ValueError("width must be a multiple of heads")
        if self.pool_side % self.summary_side:
            raise ValueError("pool_side must be a multiple of summary_side")

    @property
    def num_tokens(self) -> int:
        return self.pool_side**2

    @property
    def step_action_dim(self) -> int:
        return self.step_stride * self.action_dim


class Block(nn.Module):
    """Pre-norm transformer block with scaled-dot-product attention."""

    def __init__(self, width: int, heads: int, dropout: float):
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.proj = nn.Linear(width, width)
        self.norm2 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(
            nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch, length, width = x.shape
        q, k, v = (
            self.qkv(self.norm1(x))
            .view(batch, length, 3, self.heads, width // self.heads)
            .permute(2, 0, 3, 1, 4)
        )
        attended = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        x = x + self.dropout(self.proj(attended.transpose(1, 2).reshape(x.shape)))
        return x + self.dropout(self.mlp(self.norm2(x)))


class PatchPredictor(nn.Module):
    """Block-causal transformer from token-grid and action histories to next grids."""

    goal_distance = "l2"
    latent_kind = "tokens"

    def __init__(self, config: PatchPredictorConfig):
        super().__init__()
        self.config = config
        width, tokens = config.width, config.num_tokens
        self.latent_in = nn.Linear(config.token_dim, width)
        self.action_in = nn.Linear(config.step_action_dim, width)
        self.frame_position = nn.Parameter(torch.zeros(config.history, 1, width))
        self.patch_position = nn.Parameter(torch.zeros(1, tokens, width))
        nn.init.trunc_normal_(self.frame_position, std=0.02)
        nn.init.trunc_normal_(self.patch_position, std=0.02)
        self.blocks = nn.ModuleList(
            Block(width, config.heads, config.dropout) for _ in range(config.depth)
        )
        self.norm = nn.LayerNorm(width)
        self.latent_out = nn.Linear(width, config.token_dim)
        # A zero residual head starts training from the persistence model.
        nn.init.zeros_(self.latent_out.weight)
        nn.init.zeros_(self.latent_out.bias)
        self.register_buffer("token_mean", torch.zeros(config.token_dim))
        self.register_buffer("token_std", torch.ones(config.token_dim))
        frame = torch.arange(config.history).repeat_interleave(tokens)
        self.register_buffer("frame_mask", frame[None, :] <= frame[:, None])

    @property
    def history(self) -> int:
        return self.config.history

    @property
    def step_stride(self) -> int:
        return self.config.step_stride

    def set_normalization(self, mean: np.ndarray, std: np.ndarray) -> None:
        self.token_mean.copy_(torch.from_numpy(mean))
        self.token_std.copy_(torch.from_numpy(np.maximum(std, 1e-6)))

    def normalize(self, latents: torch.Tensor) -> torch.Tensor:
        """Pools raw token grids ``(..., N, C)`` to ``pool_side`` and normalizes."""
        latents = pool_tokens(latents.float(), self.config.pool_side)
        return (latents - self.token_mean) / self.token_std

    def denormalize(self, latents: torch.Tensor) -> torch.Tensor:
        return latents * self.token_std + self.token_mean

    def pooled(self, latents: torch.Tensor) -> torch.Tensor:
        """Spatial summary ``(..., summary_side**2 * C)`` of token grids."""
        summary = pool_tokens(latents.float(), self.config.summary_side)
        return summary.flatten(-2)

    def forward(self, latents: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Next normalized grids ``(B, L, N, C)`` after each of ``L <= history`` steps.

        ``latents`` are normalized ``(B, L, N, C)`` and ``actions[:, i]``
        ``(B, L, step_stride * A)`` is the action taken from step ``i``.
        """
        batch, length, tokens, _ = latents.shape
        if not 1 <= length <= self.config.history:
            raise ValueError(f"windows must hold 1 to {self.config.history} steps")
        with torch.autocast(
            latents.device.type, dtype=torch.bfloat16, enabled=latents.is_cuda
        ):
            x = self.latent_in(latents) + self.action_in(actions)[:, :, None]
            x = x + self.frame_position[:length] + self.patch_position
            x = x.reshape(batch, length * tokens, -1)
            mask = self.frame_mask[: length * tokens, : length * tokens]
            for block in self.blocks:
                x = block(x, mask)
            delta = self.latent_out(self.norm(x)).view(latents.shape)
        return latents + delta.float()

    def rollout_normalized(
        self,
        context: torch.Tensor,
        context_actions: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Feeds predictions back for ``H`` model steps in normalized token space.

        ``context`` ``(B, L, N, C)`` holds the normalized grids up to the current
        step, ``context_actions`` ``(B, L-1, kA)`` the grouped actions between
        them, and ``actions`` ``(B, H, kA)`` those taken from the current step on.
        Returns the ``H`` predicted grids after each action.
        """
        if context_actions.shape[1] != context.shape[1] - 1:
            raise ValueError("context_actions must hold one action fewer than context")
        history = self.config.history
        latents = context[:, -history:]
        past_actions = context_actions[
            :, context_actions.shape[1] - latents.shape[1] + 1 :
        ]
        predictions = []
        for step in range(actions.shape[1]):
            window_actions = torch.cat((past_actions, actions[:, step : step + 1]), 1)
            following = self(latents, window_actions)[:, -1:]
            predictions.append(following)
            latents = torch.cat((latents, following), 1)[:, -history:]
            past_actions = window_actions[
                :, window_actions.shape[1] - latents.shape[1] + 1 :
            ]
        return torch.cat(predictions, 1)

    def rollout(
        self,
        context: torch.Tensor,
        context_actions: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """:meth:`rollout_normalized` on raw encoder tokens, returning raw tokens."""
        predictions = self.rollout_normalized(
            self.normalize(context), context_actions, actions
        )
        return self.denormalize(predictions)


class TokenWindows:
    """Windows of ``length`` model steps of pooled tokens from a token store.

    Pooled tokens of the selected episodes are held on ``storage_device`` (the
    GPU when they fit, else host memory); ``__getitem__`` gathers a batch onto
    ``device``. Model steps are ``stride`` steps apart, and the ``length - 1``
    grouped actions between them hold the ``stride`` actions of each span.
    """

    def __init__(
        self,
        store: TokenStore,
        episode_ids: Sequence[int] | np.ndarray,
        *,
        length: int,
        stride: int,
        pool_side: int,
        device: torch.device,
        storage_device: torch.device | None = None,
        chunk: int = 512,
    ):
        starts = store.window_starts(length, stride, episode_ids)
        if len(starts) == 0:
            raise ValueError(
                f"no episode spans {length} model steps of stride {stride}"
            )
        steps = np.concatenate(
            [np.arange(s.start, s.stop) for s in map(store.episode_slice, episode_ids)]
        )
        # Windows index the compact array of selected steps, not the flat store.
        position = np.full(len(store.tokens), -1, dtype=np.int64)
        position[steps] = np.arange(len(steps))
        self.length, self.stride, self.device = length, stride, device
        self.pool_side = pool_side
        self.storage_device = storage_device or device
        pooled = torch.empty(
            len(steps),
            pool_side**2,
            store.token_dim,
            dtype=torch.float16,
            device=self.storage_device,
        )
        for begin in range(0, len(steps), chunk):
            rows = steps[begin : begin + chunk]
            block = torch.from_numpy(store.tokens[rows])
            pooled[begin : begin + len(rows)] = pool_tokens(
                block.to(device), pool_side
            ).to(self.storage_device)
        actions = store.actions[steps]
        self.tokens = pooled
        self.actions = torch.as_tensor(actions, device=device)
        self.starts = torch.as_tensor(position[starts], device=device)
        if (self.starts < 0).any():
            raise RuntimeError("window starts outside the selected episodes")

    def __len__(self) -> int:
        return len(self.starts)

    def channel_statistics(self, chunk: int = 4096) -> tuple[np.ndarray, np.ndarray]:
        """Per-channel mean and standard deviation of all selected tokens."""
        total = torch.zeros(self.tokens.shape[-1], dtype=torch.float64)
        square = torch.zeros_like(total)
        count = 0
        for begin in range(0, len(self.tokens), chunk):
            block = self.tokens[begin : begin + chunk].double().flatten(0, 1).cpu()
            total += block.sum(0)
            square += block.square().sum(0)
            count += len(block)
        mean = total / count
        std = (square / count - mean.square()).clamp_min(0).sqrt()
        return mean.float().numpy(), std.float().numpy()

    def __getitem__(self, rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Raw pooled tokens ``(B, L, N, C)`` and grouped actions ``(B, L-1, kA)``."""
        first = self.starts[rows]
        frames = first[:, None] + self.stride * torch.arange(
            self.length, device=first.device
        )
        spans = frames[:, :-1, None] + torch.arange(self.stride, device=first.device)
        tokens = self.tokens[frames.to(self.storage_device)].to(self.device)
        actions = self.actions[spans].flatten(2)
        return tokens.float(), actions


def patch_losses(
    model: PatchPredictor,
    tokens: torch.Tensor,
    actions: torch.Tensor,
    rollout_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Teacher-forced and open-loop MSE in normalized token units.

    Windows hold ``history + rollout_steps`` grids. The teacher-forced loss
    predicts steps ``1..history`` from the first ``history``; the open-loop loss
    predicts the last ``rollout_steps`` from the first ``history`` alone.
    """
    history = model.config.history
    if tokens.shape[1] != history + rollout_steps:
        raise ValueError("windows must hold history + rollout_steps grids")
    tokens = model.normalize(tokens)
    teacher = model(tokens[:, :history], actions[:, :history])
    one_step = torch.mean(torch.square(teacher - tokens[:, 1 : history + 1]))
    predicted = model.rollout_normalized(
        tokens[:, :history], actions[:, : history - 1], actions[:, history - 1 :]
    )
    open_loop = torch.mean(torch.square(predicted - tokens[:, history:]))
    return one_step, open_loop


@torch.no_grad()
def evaluate_patch_losses(
    model: PatchPredictor,
    windows: TokenWindows,
    rollout_steps: int,
    batch_size: int,
    max_windows: int | None = None,
) -> tuple[float, float]:
    """Mean teacher-forced and open-loop losses over (a subset of) the windows."""
    model.eval()
    count = len(windows) if max_windows is None else min(max_windows, len(windows))
    rows = torch.linspace(0, len(windows) - 1, count, device=windows.device).long()
    sums = torch.zeros(2, device=windows.device)
    for begin in range(0, count, batch_size):
        batch = rows[begin : begin + batch_size]
        losses = patch_losses(model, *windows[batch], rollout_steps)
        sums += torch.stack(losses) * len(batch)
    one_step, open_loop = (sums / count).tolist()
    return one_step, open_loop


def fit_patch_predictor(
    train: TokenWindows,
    validation: TokenWindows,
    *,
    config: PatchPredictorConfig,
    rollout_steps: int = 4,
    rollout_weight: float = 1.0,
    epochs: int = 30,
    steps_per_epoch: int | None = None,
    batch_size: int = 32,
    accumulation: int = 1,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.05,
    validation_windows: int = 2048,
    seed: int = 0,
    log=print,
) -> tuple[PatchPredictor, int, list[dict[str, float]]]:
    """Trains with AdamW and keeps the epoch with the lowest validation loss.

    An epoch is ``steps_per_epoch`` optimizer steps on random windows (a full
    pass over the training windows by default); ``accumulation`` micro-batches
    of ``batch_size`` make one step. Returns the model, the best epoch, and
    per-epoch losses, wall-clock seconds, and peak GPU memory.
    """
    if rollout_steps < 1 or accumulation < 1:
        raise ValueError("rollout_steps and accumulation must be at least 1")
    torch.manual_seed(seed)
    device = train.device
    model = PatchPredictor(config).to(device)
    model.set_normalization(*train.channel_statistics())
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    per_epoch = steps_per_epoch or max(1, len(train) // (batch_size * accumulation))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    generator = torch.Generator(device=device).manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    best_loss, best_epoch, best_state = float("inf"), 0, None
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        started = time.perf_counter()
        sums = torch.zeros(2, device=device)
        for _ in range(per_epoch):
            optimizer.zero_grad(set_to_none=True)
            for _ in range(accumulation):
                rows = torch.randint(
                    len(train), (batch_size,), device=device, generator=generator
                )
                one_step, open_loop = patch_losses(model, *train[rows], rollout_steps)
                loss = one_step + rollout_weight * open_loop
                (loss / accumulation).backward()
                sums += torch.stack((one_step, open_loop)).detach() / accumulation
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()
        train_losses = (sums / per_epoch).tolist()
        validation_losses = evaluate_patch_losses(
            model, validation, rollout_steps, batch_size, validation_windows
        )
        validation_loss = validation_losses[0] + rollout_weight * validation_losses[1]
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        record = {
            "train_one_step": train_losses[0],
            "train_open_loop": train_losses[1],
            "validation_one_step": validation_losses[0],
            "validation_open_loop": validation_losses[1],
            "epoch_seconds": time.perf_counter() - started,
            "peak_memory_gb": (
                torch.cuda.max_memory_allocated(device) / 1e9
                if device.type == "cuda"
                else 0.0
            ),
        }
        history.append(record)
        log(
            f"epoch {epoch}/{epochs} "
            + " ".join(f"{k}={v:.4g}" for k, v in record.items())
        )
        if validation_loss < best_loss:
            best_loss, best_epoch = validation_loss, epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model.eval(), best_epoch, history


def save_patch_predictor(
    path: Path, model: PatchPredictor, metadata: Mapping[str, object]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "kind": "patch_predictor",
            "config": asdict(model.config),
            "state_dict": model.state_dict(),
            "metadata": dict(metadata),
        },
        path,
    )


def is_patch_artifact(path: Path) -> bool:
    artifact = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    return artifact.get("kind") == "patch_predictor"


def load_patch_predictor(
    path: Path, device: str | torch.device = "cpu"
) -> tuple[PatchPredictor, dict]:
    artifact = torch.load(path, map_location=device, weights_only=True)
    model = PatchPredictor(PatchPredictorConfig(**artifact["config"]))
    model.load_state_dict(artifact["state_dict"])
    return model.to(device).eval(), artifact["metadata"]


def native_pool_side(store: TokenStore) -> int:
    """Side of the token grid as cached."""
    return grid_side(store.num_tokens)


def token_episodes(
    store: TokenStore,
    episode_ids: Sequence[int] | np.ndarray,
    summary_side: int,
    device: torch.device | str = "cpu",
) -> dict[int, dict[str, np.ndarray]]:
    """Episodes for :func:`patch_rollouts`: memmapped ``tokens`` and the rest.

    ``features`` are the spatial summaries probes decode, ``observations`` and
    ``actions`` the ones stored with the tokens (``T+1`` and ``T`` rows).
    """
    episodes = {}
    for episode_id in map(int, episode_ids):
        steps = store.episode_slice(episode_id)
        episodes[episode_id] = {
            "tokens": store.tokens[steps],
            "features": summary_features(store, episode_id, summary_side, device),
            "observations": store.observations[steps],
            "actions": store.actions[steps][:-1],
        }
    return episodes


def patch_rollouts(model: PatchPredictor, probe) -> EpisodeRollout:
    """Open-loop rollouts from true context, decoded to observations.

    Each model step's prediction is held over its ``step_stride`` control steps,
    so only horizons that are multiples of the stride are exact. The probe's
    history uses the true summaries of earlier model steps, then the predictions.
    """
    if probe.stride != model.step_stride:
        raise ValueError("the probe must stack steps model.step_stride apart")
    device = next(model.parameters()).device
    stride, history = model.step_stride, model.history

    @torch.no_grad()
    def rollout(episode, start, num_steps):
        model_steps = math.ceil(num_steps / stride)
        context_steps = start - stride * np.arange(history)[::-1]
        context_steps = context_steps[context_steps >= 0]
        context = torch.as_tensor(
            np.asarray(episode["tokens"][context_steps], dtype=np.float32),
            device=device,
        )[None]
        actions = episode["actions"]
        context_actions = group_actions(actions, context_steps[:-1], stride)
        future = group_actions(actions, start + stride * np.arange(model_steps), stride)
        predicted = model.rollout(
            context,
            torch.as_tensor(context_actions, device=device)[None],
            torch.as_tensor(future, device=device)[None],
        )
        past_steps = np.maximum(start - stride * np.arange(probe.history)[::-1], 0)
        past = torch.as_tensor(episode["features"][past_steps], device=device)[None]
        observations = decode_latents(probe, past, model.pooled(predicted))[0]
        held = np.arange(num_steps) // stride
        return observations.cpu().numpy()[held]

    return rollout

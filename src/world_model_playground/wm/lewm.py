"""LeWM: a JEPA trained end to end from pixels (Maes et al., arXiv 2603.19312).

An encoder maps every frame to one embedding and an action-conditioned
predictor maps the embeddings of the last ``history`` frames to those of the
next ones. Both train together on two terms and nothing else: the mean squared
error between predicted and actual next embeddings, and SIGReg
(``wm.sigreg``), which pulls the embeddings of each step toward an isotropic
Gaussian so they cannot collapse. There is no stop-gradient, target network,
or reconstruction.

The architecture follows the released code (``config/train/model/lewm.yaml``):

- Encoder: ViT-tiny (12 layers, 3 heads, 192 dimensions); the final CLS token
  goes through a one-hidden-layer MLP with batch normalization, because the
  ViT's last layer norm would otherwise fight SIGReg.
- Predictor: 6 transformer blocks (16 heads of 64, MLP 2048, dropout 0.1) with
  learned positions and causal attention over the embedding history, the action
  embedding injected by AdaLN-zero in every block, then another batch-normalized
  MLP.
- One model step spans ``step_stride`` control steps (the paper's frame skip of
  5): its action is the block of forces applied between two frames.

Adapted to CartPole: frames are small (64x64 with 8x8 patches by default), and
the embedding of a step is a single vector ``(D,)`` (``latent_kind == "pooled"``),
already close to ``N(0, I)``, so ``normalize`` is the identity. ``LeWM`` follows
the ``wm.planning.LatentWorldModel`` protocol and ``LeWMEncoder`` exposes the
trained encoder to ``embed.py``, the state probes, and latent MPC.
"""

import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset, RandomSampler
from transformers import ViTConfig, ViTModel

from world_model_playground.wm.ac_world_model import group_actions
from world_model_playground.wm.data import ACTION_SIZE
from world_model_playground.wm.encoders import (
    EncoderSpec,
    FrozenEncoder,
    preprocess_frames,
)
from world_model_playground.wm.episodes import (
    episode_lengths,
    frames_key,
    is_episodic,
    load_episode,
    load_manifest,
)
from world_model_playground.wm.eval import EpisodeRollout
from world_model_playground.wm.planning import decode_latents
from world_model_playground.wm.sigreg import SIGReg

KIND = "lewm"
ENCODER_PREFIX = "lewm:"


@dataclass(frozen=True)
class LeWMConfig:
    """Sizes of the encoder and predictor and how model steps map to the control."""

    image_size: int = 64
    patch_size: int = 8
    embed_dim: int = 192
    encoder_depth: int = 12
    encoder_heads: int = 3
    predictor_depth: int = 6
    predictor_heads: int = 16
    predictor_head_dim: int = 64
    predictor_mlp_dim: int = 2048
    projector_hidden: int = 2048
    dropout: float = 0.1
    history: int = 3
    step_stride: int = 5
    action_dim: int = ACTION_SIZE
    action_smoothed_dim: int = 10
    gradient_checkpointing: bool = False

    def __post_init__(self):
        if self.image_size % self.patch_size:
            raise ValueError("image_size must be a multiple of patch_size")
        if self.history < 1 or self.step_stride < 1:
            raise ValueError("history and step_stride must be at least 1")

    @property
    def action_inputs(self) -> int:
        return self.step_stride * self.action_dim

    @property
    def frames(self) -> int:
        """Frames of a training window: the history and the frame it predicts."""
        return self.history + 1


class _MLP(nn.Module):
    """Linear, batch norm, GELU, linear: the projector after encoder and predictor."""

    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Maps ``(B, T, D)`` through the MLP, batch-normalizing over ``B * T``."""
        batch, steps, _ = x.shape
        return self.net(x.reshape(batch * steps, -1)).view(batch, steps, -1)


class _ActionEmbedder(nn.Module):
    """Embeds the block of forces of a model step."""

    def __init__(self, inputs: int, smoothed: int, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(inputs, smoothed),
            nn.Linear(smoothed, 4 * dim),
            nn.SiLU(),
            nn.Linear(4 * dim, dim),
        )

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        return self.net(actions.float())


class _Attention(nn.Module):
    def __init__(self, dim: int, heads: int, head_dim: int, dropout: float):
        super().__init__()
        inner = heads * head_dim
        self.heads = heads
        self.dropout = dropout
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, 3 * inner, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner, dim), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, steps, _ = x.shape
        qkv = self.to_qkv(self.norm(x)).view(batch, steps, 3, self.heads, -1)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=True
        )
        return self.to_out(out.transpose(1, 2).reshape(batch, steps, -1))


class _FeedForward(nn.Module):
    def __init__(self, dim: int, hidden: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _ConditionalBlock(nn.Module):
    """A transformer block whose shifts, scales, and gates come from the action."""

    def __init__(self, config: LeWMConfig):
        super().__init__()
        dim = config.embed_dim
        self.attention = _Attention(
            dim, config.predictor_heads, config.predictor_head_dim, config.dropout
        )
        self.feed_forward = _FeedForward(dim, config.predictor_mlp_dim, config.dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        # Zero-initialized, so action conditioning is learned progressively.
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        shift1, scale1, gate1, shift2, scale2, gate2 = self.modulation(condition).chunk(
            6, dim=-1
        )
        x = x + gate1 * self.attention(self.norm1(x) * (1 + scale1) + shift1)
        return x + gate2 * self.feed_forward(self.norm2(x) * (1 + scale2) + shift2)


class _Predictor(nn.Module):
    """Causal transformer over the embedding history, conditioned on the actions."""

    def __init__(self, config: LeWMConfig):
        super().__init__()
        self.positions = nn.Parameter(torch.randn(1, config.history, config.embed_dim))
        self.blocks = nn.ModuleList(
            _ConditionalBlock(config) for _ in range(config.predictor_depth)
        )
        self.norm = nn.LayerNorm(config.embed_dim)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        x = x + self.positions[:, : x.shape[1]]
        for block in self.blocks:
            x = block(x, condition)
        return self.norm(x)


class LeWM(nn.Module):
    """The JEPA world model behind the latent world-model interface of ``wm.planning``.

    Latents are per-step embeddings ``(..., D)``; actions are grouped per model
    step into ``step_stride * action_dim`` values.
    """

    goal_distance = "l2"
    latent_kind = "pooled"

    def __init__(self, config: LeWMConfig):
        super().__init__()
        self.config = config
        dim = config.embed_dim
        self.encoder = ViTModel(
            ViTConfig(
                hidden_size=dim,
                num_hidden_layers=config.encoder_depth,
                num_attention_heads=config.encoder_heads,
                intermediate_size=4 * dim,
                image_size=config.image_size,
                patch_size=config.patch_size,
                num_channels=3,
                qkv_bias=True,
            ),
            add_pooling_layer=False,
        )
        if config.gradient_checkpointing:
            self.encoder.gradient_checkpointing_enable()
        self.projector = _MLP(dim, config.projector_hidden)
        self.action_encoder = _ActionEmbedder(
            config.action_inputs, config.action_smoothed_dim, dim
        )
        self.predictor = _Predictor(config)
        self.prediction_projector = _MLP(dim, config.projector_hidden)

    @property
    def history(self) -> int:
        return self.config.history

    @property
    def step_stride(self) -> int:
        return self.config.step_stride

    def embed(self, pixels: torch.Tensor) -> torch.Tensor:
        """Embeddings ``(B, T, D)`` of normalized frames ``(B, T, 3, S, S)``."""
        batch, steps = pixels.shape[:2]
        hidden = self.encoder(pixel_values=pixels.flatten(0, 1)).last_hidden_state
        return self.projector(hidden[:, :1]).view(batch, steps, -1)

    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        """Embeddings ``(B, T, D)`` of uint8 frames ``(B, T, 3, H, W)``."""
        return self.embed(preprocess_frames(frames, self.config.image_size))

    def forward(self, latents: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Embeddings ``(B, L, D)`` of the step after each of ``L`` steps.

        ``actions[:, i]`` ``(B, L, step_stride * A)`` are the forces applied
        from step ``i``; attention is causal, so every prefix is a valid context.
        """
        steps = latents.shape[1]
        if not 1 <= steps <= self.config.history or actions.shape[1] != steps:
            raise ValueError(
                f"windows must hold 1 to {self.config.history} steps with one "
                "action block each"
            )
        hidden = self.predictor(latents, self.action_encoder(actions))
        return self.prediction_projector(hidden)

    def losses(
        self, frames: torch.Tensor, actions: torch.Tensor, sigreg: SIGReg
    ) -> dict[str, torch.Tensor]:
        """Prediction, SIGReg, and embeddings of windows of ``history + 1`` frames.

        ``frames`` ``(B, history + 1, 3, H, W)`` are uint8 and ``actions``
        ``(B, history, step_stride * A)`` the force blocks between them.
        """
        embeddings = self.encode(frames)
        predicted = self(embeddings[:, :-1], actions)
        return {
            "prediction": (predicted.float() - embeddings[:, 1:].float())
            .square()
            .mean(),
            "sigreg": sigreg(embeddings.transpose(0, 1)),
            "embeddings": embeddings.detach(),
        }

    @staticmethod
    def normalize(latents: torch.Tensor) -> torch.Tensor:
        return latents

    @staticmethod
    def denormalize(latents: torch.Tensor) -> torch.Tensor:
        return latents

    @staticmethod
    def pooled(latents: torch.Tensor) -> torch.Tensor:
        return latents

    def rollout_normalized(
        self,
        context: torch.Tensor,
        context_actions: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Feeds predictions back for ``H`` model steps.

        ``context`` ``(B, L, D)`` holds the embeddings up to the current step,
        ``context_actions`` ``(B, L-1, kA)`` the grouped forces between them, and
        ``actions`` ``(B, H, kA)`` those from the current step on.
        """
        if context_actions.shape[1] != context.shape[1] - 1:
            raise ValueError("context_actions must hold one action fewer than context")

        def last(values: torch.Tensor, count: int) -> torch.Tensor:
            return values[:, values.shape[1] - count :]

        latents = context[:, -self.history :]
        past = last(context_actions, latents.shape[1] - 1)
        predictions = []
        for step in range(actions.shape[1]):
            window = torch.cat((past, actions[:, step : step + 1]), dim=1)
            following = self(latents, window)[:, -1:]
            predictions.append(following)
            latents = torch.cat((latents, following), dim=1)[:, -self.history :]
            past = last(window, latents.shape[1] - 1)
        return torch.cat(predictions, dim=1)

    def rollout(
        self,
        context: torch.Tensor,
        context_actions: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        return self.rollout_normalized(context, context_actions, actions)


class LeWMEncoder(FrozenEncoder):
    """The trained LeWM encoder as an ``embed.py`` encoder: one frame, one embedding."""

    def __init__(self, spec: EncoderSpec, model: LeWM):
        super().__init__(spec, model, context_frames=1)
        self.feature_dim = self.token_dim = model.config.embed_dim

    def _encode(self, pixels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embeddings = self.backbone.embed(pixels)[:, 0]
        return embeddings, embeddings[:, None]


def encoder_name(path: Path) -> str:
    """The ``load_encoder`` name of the encoder in the LeWM artifact at ``path``."""
    return f"{ENCODER_PREFIX}{path.resolve()}"


def load_lewm_encoder(
    name: str, device: str | torch.device, dtype: torch.dtype | None = None
) -> LeWMEncoder:
    """Restores the encoder named ``lewm:<artifact path>`` in evaluation mode."""
    path = Path(name.removeprefix(ENCODER_PREFIX))
    model, _ = load_lewm(path, device)
    if dtype is not None:
        model = model.to(dtype)
    spec = EncoderSpec(
        name=name,
        family="lewm",
        checkpoint=str(path),
        input_size=model.config.image_size,
        default_context_frames=1,
    )
    return LeWMEncoder(spec, model)


class FrameWindows(Dataset):
    """Windows of ``frames`` frames ``stride`` control steps apart, held in memory.

    Items hold uint8 ``frames`` ``(frames, 3, H, W)``, the ``actions`` blocks
    between them ``(frames - 1, stride * A)``, and the ``observations`` at the
    frames. The frames of all episodes are loaded up front, so batches cost
    one gather instead of one shard decompression per item.
    """

    def __init__(
        self,
        collection_dir: Path,
        episode_ids: Sequence[int] | np.ndarray,
        *,
        resolution: tuple[int, int],
        frames: int,
        stride: int,
        start_every: int = 1,
    ):
        if frames < 2 or stride < 1 or start_every < 1:
            raise ValueError(
                "frames must be at least 2; stride, start_every at least 1"
            )
        manifest = load_manifest(collection_dir)
        if not is_episodic(manifest):
            raise ValueError(f"{collection_dir} is not an episodic collection")
        self.key = frames_key(*resolution)
        if self.key not in {entry["key"] for entry in manifest["resolutions"]}:
            raise ValueError(f"{collection_dir} has no rendered {self.key}")
        self.frames, self.stride = frames, stride
        lengths = episode_lengths(manifest)
        span = stride * (frames - 1)
        self.episodes: dict[int, dict[str, np.ndarray]] = {}
        self.index: list[tuple[int, int]] = []
        for episode_id in (int(i) for i in episode_ids):
            if lengths[episode_id] < span:
                continue
            self.episodes[episode_id] = load_episode(
                collection_dir, episode_id, fields=(self.key, "observations", "actions")
            )
            self.index += [
                (episode_id, start)
                for start in range(0, lengths[episode_id] - span + 1, start_every)
            ]

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        episode_id, start = self.index[item]
        episode = self.episodes[episode_id]
        steps = start + self.stride * np.arange(self.frames)
        frames = torch.from_numpy(episode[self.key][steps]).permute(0, 3, 1, 2)
        return {
            "frames": frames.contiguous(),
            "actions": torch.from_numpy(
                group_actions(episode["actions"], steps[:-1], self.stride)
            ),
            "observations": torch.from_numpy(episode["observations"][steps]),
        }


def embedding_statistics(embeddings: torch.Tensor) -> dict[str, float]:
    """Collapse monitors of embeddings ``(..., D)``: spread and effective rank.

    ``std_min`` falling to zero is dimensional collapse; ``effective_rank`` is
    the exponential of the entropy of the covariance spectrum, at most ``D``.
    """
    flat = embeddings.detach().float().reshape(-1, embeddings.shape[-1])
    std = flat.std(dim=0)
    spectrum = torch.linalg.eigvalsh(torch.cov(flat.T)).clamp_min(0)
    probabilities = spectrum / spectrum.sum().clamp_min(1e-12)
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum()
    return {
        "std_mean": std.mean().item(),
        "std_min": std.min().item(),
        "effective_rank": entropy.exp().item(),
    }


def ridge_r2(
    features: np.ndarray, targets: np.ndarray, *, fit_share: float = 0.8
) -> np.ndarray:
    """Held-out R² per target of a ridge regression from ``features``.

    A quick monitor of whether embeddings still encode the state, fitted on the
    first ``fit_share`` of the rows and scored on the rest; the state probes
    of ``train_probe.py`` are the real measurement.
    """
    count = int(len(features) * fit_share)
    mean, std = features[:count].mean(0), features[:count].std(0) + 1e-6
    x = (features - mean) / std
    x = np.concatenate((x, np.ones((len(x), 1))), axis=1)
    gram = x[:count].T @ x[:count] + 1e-2 * count * np.eye(x.shape[1])
    weights = np.linalg.solve(gram, x[:count].T @ targets[:count])
    residual = ((x[count:] @ weights - targets[count:]) ** 2).sum(0)
    total = ((targets[count:] - targets[count:].mean(0)) ** 2).sum(0)
    return 1.0 - residual / np.maximum(total, 1e-12)


def _autocast(device: torch.device):
    return torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
    )


@torch.no_grad()
def evaluate_lewm(
    model: LeWM,
    windows: Dataset,
    items: np.ndarray,
    sigreg: SIGReg,
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    """Losses, collapse monitors, and the state-recovery monitor on ``items``."""
    was_training = model.training
    model.eval()
    prediction, regularizer, embeddings, observations = [], [], [], []
    for start in range(0, len(items), batch_size):
        batch = [windows[int(i)] for i in items[start : start + batch_size]]
        frames = torch.stack([b["frames"] for b in batch]).to(device)
        actions = torch.stack([b["actions"] for b in batch]).to(device)
        # Float32, not bfloat16: embeddings of a model without SIGReg have a tiny
        # spread that bfloat16 rounds away, which would make the monitors read
        # collapse where the information is still there.
        losses = model.losses(frames, actions, sigreg)
        prediction.append(losses["prediction"].item() * len(batch))
        regularizer.append(losses["sigreg"].item() * len(batch))
        embeddings.append(losses["embeddings"][:, 0].float().cpu())
        observations.append(torch.stack([b["observations"][0] for b in batch]))
    model.train(was_training)
    count = len(items)
    embeddings_tensor = torch.cat(embeddings)
    result = {
        "prediction": sum(prediction) / count,
        "sigreg": sum(regularizer) / count,
        **{k: v for k, v in embedding_statistics(embeddings_tensor).items()},
    }
    if len(embeddings_tensor) >= 50:
        r2 = ridge_r2(
            embeddings_tensor.numpy().astype(np.float64),
            torch.cat(observations).numpy().astype(np.float64),
        )
        result["monitor_r2_pose"] = float(r2[:3].mean())
        result["monitor_r2_velocity"] = float(r2[3:].mean())
    return result


def fit_lewm(
    model: LeWM,
    train: Dataset,
    validation: Dataset,
    *,
    sigreg: SIGReg,
    sigreg_weight: float = 0.09,
    iterations: int,
    batch_size: int = 128,
    learning_rate: float = 5e-5,
    weight_decay: float = 1e-3,
    warmup: int = 100,
    grad_clip: float = 1.0,
    eval_every: int = 500,
    eval_windows: int = 512,
    num_workers: int = 2,
    seed: int = 0,
    device: torch.device,
    log: Callable[[dict], None] | None = None,
) -> tuple[LeWM, int, list[dict]]:
    """Trains ``prediction + sigreg_weight * SIGReg`` as LeWM does.

    AdamW with a linear warmup into a cosine decay, bfloat16 autocast on CUDA,
    and gradient clipping, the released configuration. Returns the model
    restored to the iteration with the lowest validation total loss, that
    iteration, and the log of training and validation losses (also passed to
    ``log`` as it is produced).
    """
    generator = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    model = model.to(device).train()
    sigreg = sigreg.to(device)
    loader = DataLoader(
        train,
        batch_size=batch_size,
        sampler=RandomSampler(
            train,
            replacement=True,
            num_samples=iterations * batch_size,
            generator=generator,
        ),
        num_workers=num_workers,
        drop_last=True,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    def schedule(iteration: int) -> float:
        if iteration < warmup:
            return (iteration + 1) / warmup
        progress = (iteration - warmup) / max(1, iterations - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    items = np.unique(
        np.linspace(0, len(validation) - 1, min(eval_windows, len(validation))).astype(
            np.int64
        )
    )
    losses: list[dict] = []
    best_total, best_iteration, best_state = math.inf, 0, None
    window = {"prediction": 0.0, "sigreg": 0.0, "count": 0}
    start_time = time.perf_counter()

    def record(entry: dict) -> None:
        losses.append(entry)
        if log is not None:
            log(entry)

    for iteration, batch in enumerate(loader, start=1):
        frames = batch["frames"].to(device, non_blocking=True)
        actions = batch["actions"].to(device, non_blocking=True)
        with _autocast(device):
            parts = model.losses(frames, actions, sigreg)
        loss = parts["prediction"] + sigreg_weight * parts["sigreg"]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()
        window["prediction"] += parts["prediction"].item()
        window["sigreg"] += parts["sigreg"].item()
        window["count"] += 1
        if iteration % eval_every == 0 or iteration == iterations:
            stats = embedding_statistics(parts["embeddings"])
            validation_metrics = evaluate_lewm(
                model, validation, items, sigreg, batch_size=batch_size, device=device
            )
            total = validation_metrics["prediction"] + (
                sigreg_weight * validation_metrics["sigreg"]
            )
            record(
                {
                    "iteration": iteration,
                    "train_prediction": window["prediction"] / window["count"],
                    "train_sigreg": window["sigreg"] / window["count"],
                    "train_std_min": stats["std_min"],
                    "train_effective_rank": stats["effective_rank"],
                    "grad_norm": float(grad_norm),
                    "learning_rate": scheduler.get_last_lr()[0],
                    "seconds": time.perf_counter() - start_time,
                    **{f"validation_{k}": v for k, v in validation_metrics.items()},
                    "validation_total": total,
                }
            )
            window = {"prediction": 0.0, "sigreg": 0.0, "count": 0}
            if total < best_total:
                best_total, best_iteration = total, iteration
                best_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
    if best_state is not None:
        model.load_state_dict(best_state)
    return model.eval(), best_iteration, losses


def save_lewm(path: Path, model: LeWM, metadata: Mapping[str, object]) -> None:
    """Saves the model with its config and the metadata of its training run."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "kind": KIND,
            "config": asdict(model.config),
            "state_dict": model.state_dict(),
            "metadata": dict(metadata),
        },
        path,
    )


def is_lewm_artifact(path: Path) -> bool:
    artifact = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    return artifact.get("kind") == KIND


def load_lewm(path: Path, device: str | torch.device = "cpu") -> tuple[LeWM, dict]:
    """Restores a model written by :func:`save_lewm` in evaluation mode."""
    artifact = torch.load(path, map_location="cpu", weights_only=True)
    model = LeWM(LeWMConfig(**artifact["config"]))
    model.load_state_dict(artifact["state_dict"])
    return model.to(device).eval(), artifact["metadata"]


def episode_embeddings(
    model: LeWM, frames: np.ndarray, *, device: torch.device, batch_size: int = 256
) -> np.ndarray:
    """Embeddings ``(N, D)`` of uint8 frames ``(N, H, W, 3)`` of one episode."""
    outputs = []
    with torch.no_grad():
        for start in range(0, len(frames), batch_size):
            chunk = torch.from_numpy(frames[start : start + batch_size])
            chunk = chunk.permute(0, 3, 1, 2)[None].to(device)
            with _autocast(device):
                outputs.append(model.encode(chunk)[0].float().cpu())
    return torch.cat(outputs).numpy()


def lewm_rollouts(model: LeWM, probe) -> tuple[EpisodeRollout, EpisodeRollout]:
    """Open-loop rollouts from true context, as ``wm.eval`` episode rollouts.

    Episodes hold the embeddings of the trained encoder as ``features``
    ``(T+1, D)``, and ``actions``. The first rollout returns probe-decoded
    observations, the second embeddings (target ``features``). Both hold each
    model step's prediction over its ``step_stride`` control steps, so only
    horizons that are multiples of the stride are exact.
    """
    if probe.stride != model.step_stride:
        raise ValueError("the probe must stack steps model.step_stride apart")
    device = next(model.parameters()).device
    stride, history = model.step_stride, model.history

    @torch.no_grad()
    def predict(episode, start, num_steps):
        model_steps = math.ceil(num_steps / stride)
        context_steps = start - stride * np.arange(history)[::-1]
        context_steps = context_steps[context_steps >= 0]
        context = torch.as_tensor(
            np.asarray(episode["features"][context_steps], dtype=np.float32),
            device=device,
        )[None]
        actions = episode["actions"]
        context_actions = group_actions(actions, context_steps[:-1], stride)
        future = group_actions(actions, start + stride * np.arange(model_steps), stride)
        predicted = model.rollout(
            context,
            torch.as_tensor(context_actions, device=device)[None],
            torch.as_tensor(future, device=device)[None],
        ).float()
        # The probe's history: true embeddings of earlier model steps.
        past_steps = np.maximum(start - stride * np.arange(probe.history)[::-1], 0)
        past = torch.as_tensor(episode["features"][past_steps], device=device)[None]
        observations = decode_latents(probe, past, predicted)[0].cpu().numpy()
        held = np.arange(num_steps) // stride
        return observations[held], predicted[0].cpu().numpy()[held]

    def observations(episode, start, num_steps):
        return predict(episode, start, num_steps)[0]

    def latents(episode, start, num_steps):
        return predict(episode, start, num_steps)[1]

    return observations, latents

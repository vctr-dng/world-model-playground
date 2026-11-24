"""V-JEPA 2-AC as a CartPole world model: adapter, token windows, and training.

The released predictor (``wm.vendor.vjepa2``, 24 blocks, 300M parameters) reads
the 16x16 patch tokens of each frame from the frozen ViT-g encoder, each frame
preceded by an action token and a state token, under a block-causal mask over
frames, and predicts every next frame's tokens. It was post-trained on DROID
robot video at 4 frames per second with 7-D end-effector actions and states.

For CartPole:

- One model step spans ``step_stride`` control steps (0.01 s each), so the
  action token reads the ``step_stride`` forces applied over that span through
  a new linear layer, initialized from the pretrained action layer's first
  columns (equivalent to zero-padding the forces into its 7-D input).
- The state token receives zeros, so it becomes a constant learned token:
  feeding the true observation would leak the state the model should infer
  from pixels.
- Latents are layer-normalized tokens ``(..., N, C)`` (no affine), the space
  the predictor was trained in; cached tokens are already in it
  (``EncoderSpec.layer_norm_tokens``). ``pooled`` averages them into one
  feature per step for state probes.

Training follows V-JEPA 2-AC: an L1 loss on the teacher-forced predictions of a
window plus an L1 loss on predictions fed back as inputs.
"""

import math
import zipfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset, RandomSampler

from world_model_playground.wm.ac_checkpoint import (
    PREDICTOR_KWARGS,
    clean_keys,
    load_checkpoint,
)
from world_model_playground.wm.data import ACTION_SIZE
from world_model_playground.wm.episodes import episode_path
from world_model_playground.wm.eval import EpisodeRollout
from world_model_playground.wm.vendor.vjepa2 import (
    VisionTransformerPredictorAC,
    vit_ac_predictor,
)

KIND = "vjepa2-ac"


@dataclass(frozen=True)
class ACConfig:
    """How the AC predictor is adapted to CartPole and what it starts from."""

    step_stride: int = 5
    history: int = 3
    action_dim: int = ACTION_SIZE
    pretrained: bool = True
    activation_checkpointing: bool = False
    # Pretrained blocks kept frozen, counted from the input (with the input layer).
    freeze_blocks: int = 0
    # Vendored predictor settings; None means the released checkpoint's.
    predictor: dict | None = None

    def __post_init__(self):
        if self.step_stride < 1 or self.history < 1:
            raise ValueError("step_stride and history must be at least 1")

    @property
    def action_inputs(self) -> int:
        return self.step_stride * self.action_dim


class ACWorldModel(nn.Module):
    """The AC predictor behind the latent world-model interface of ``wm.planning``.

    Latents are ``(..., N, C)`` token grids, one per model step; actions are
    grouped per model step into ``step_stride * action_dim`` values.
    """

    goal_distance = "l1"
    latent_kind = "tokens"

    def __init__(self, config: ACConfig, predictor: VisionTransformerPredictorAC):
        super().__init__()
        self.config = config
        pretrained_actions = predictor.action_encoder
        actions = nn.Linear(config.action_inputs, pretrained_actions.out_features)
        with torch.no_grad():
            if config.action_inputs <= pretrained_actions.in_features:
                actions.weight.copy_(
                    pretrained_actions.weight[:, : config.action_inputs]
                )
                actions.bias.copy_(pretrained_actions.bias)
            else:
                nn.init.trunc_normal_(actions.weight, std=0.02)
                nn.init.zeros_(actions.bias)
        predictor.action_encoder = actions
        predictor.use_activation_checkpointing = config.activation_checkpointing
        if config.freeze_blocks:
            predictor.predictor_embed.requires_grad_(False)
            for block in predictor.predictor_blocks[: config.freeze_blocks]:
                block.requires_grad_(False)
        self.predictor = predictor
        self.tokens_per_step = predictor.grid_height * predictor.grid_width
        self.token_dim = predictor.predictor_proj.out_features
        self.state_inputs = predictor.state_encoder.in_features
        # The block-causal mask covers this many steps of (2 + N) tokens.
        self.max_steps = predictor.attn_mask.shape[0] // (self.tokens_per_step + 2)
        if config.history > self.max_steps:
            raise ValueError(f"history must be at most {self.max_steps} steps")

    @property
    def history(self) -> int:
        return self.config.history

    @property
    def step_stride(self) -> int:
        return self.config.step_stride

    @property
    def dtype(self) -> torch.dtype:
        return self.predictor.predictor_embed.weight.dtype

    @staticmethod
    def normalize(latents: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(latents, latents.shape[-1:])

    @staticmethod
    def denormalize(latents: torch.Tensor) -> torch.Tensor:
        return latents

    @staticmethod
    def pooled(latents: torch.Tensor) -> torch.Tensor:
        """One feature per step: the mean token, as the probes were fitted on."""
        return latents.mean(dim=-2)

    def forward(self, latents: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Next normalized latents ``(B, L, N, C)`` after each of ``L`` steps.

        ``actions[:, i]`` ``(B, L, step_stride * A)`` are the forces applied
        from step ``i``; the block-causal mask makes every prefix a valid context.
        """
        batch, length, tokens, channels = latents.shape
        if not 1 <= length <= self.max_steps:
            raise ValueError(f"windows must hold 1 to {self.max_steps} steps")
        if tokens != self.tokens_per_step or channels != self.token_dim:
            raise ValueError(
                f"latents must be (B, L, {self.tokens_per_step}, {self.token_dim})"
            )
        dtype = self.dtype
        states = latents.new_zeros((batch, length, self.state_inputs), dtype=dtype)
        predicted = self.predictor(
            latents.reshape(batch, length * tokens, channels).to(dtype),
            actions.to(dtype),
            states,
        )
        predicted = F.layer_norm(predicted.float(), (channels,))
        return predicted.view(batch, length, tokens, channels)

    def rollout_normalized(
        self,
        context: torch.Tensor,
        context_actions: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Feeds predictions back for ``H`` model steps.

        ``context`` ``(B, L, N, C)`` holds normalized latents up to the current
        step, ``context_actions`` ``(B, L-1, kA)`` the grouped forces between
        them, and ``actions`` ``(B, H, kA)`` those from the current step on.
        """
        if context_actions.shape[1] != context.shape[1] - 1:
            raise ValueError("context_actions must hold one action fewer than context")

        def between(steps: torch.Tensor, step_actions: torch.Tensor) -> torch.Tensor:
            return step_actions[:, step_actions.shape[1] - steps.shape[1] + 1 :]

        history = self.history
        latents = context[:, -history:]
        past_actions = between(latents, context_actions)
        predictions = []
        for step in range(actions.shape[1]):
            window_actions = torch.cat((past_actions, actions[:, step : step + 1]), 1)
            following = self(latents, window_actions)[:, -1:]
            predictions.append(following)
            latents = torch.cat((latents, following), 1)[:, -history:]
            past_actions = between(latents, window_actions)
        return torch.cat(predictions, 1)

    def rollout(
        self,
        context: torch.Tensor,
        context_actions: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        return self.rollout_normalized(
            self.normalize(context), context_actions, actions
        )


def build_ac_model(config: ACConfig, checkpoint: Mapping | None = None) -> ACWorldModel:
    """The adapted predictor, from the released weights when ``config.pretrained``.

    Without them it is the same architecture from random initialization, the
    ablation that tells whether DROID pretraining helps.
    """
    predictor = vit_ac_predictor(**(config.predictor or PREDICTOR_KWARGS))
    if config.pretrained:
        checkpoint = load_checkpoint() if checkpoint is None else checkpoint
        predictor.load_state_dict(clean_keys(checkpoint["predictor"]), strict=True)
    return ACWorldModel(config, predictor)


def group_actions(
    actions: np.ndarray, starts: Sequence[int], stride: int
) -> np.ndarray:
    """Forces applied from each start for ``stride`` steps, ``(len(starts), stride*A)``.

    Groups running past the episode end repeat its last action.
    """
    last = len(actions) - 1
    rows = np.minimum(
        np.asarray(starts, dtype=np.int64)[:, None] + np.arange(stride), last
    )
    return actions[rows].reshape(len(rows), stride * actions.shape[-1])


def npz_member_memmap(path: Path, name: str) -> np.memmap:
    """Memory-maps one array of an uncompressed ``.npz`` (as ``np.savez`` writes).

    Reading windows of a few steps then touches only those steps' tokens rather
    than decompressing the whole episode.
    """
    with zipfile.ZipFile(path) as archive:
        info = archive.getinfo(f"{name}.npy")
    if info.compress_type != zipfile.ZIP_STORED:
        raise ValueError(f"{path}:{name} is compressed and cannot be memory-mapped")
    with open(path, "rb") as file:
        file.seek(info.header_offset)
        local_header = file.read(30)
        name_length = int.from_bytes(local_header[26:28], "little")
        extra_length = int.from_bytes(local_header[28:30], "little")
        file.seek(info.header_offset + 30 + name_length + extra_length)
        version = np.lib.format.read_magic(file)
        read_header = (
            np.lib.format.read_array_header_1_0
            if version == (1, 0)
            else np.lib.format.read_array_header_2_0
        )
        shape, fortran_order, dtype = read_header(file)
        offset = file.tell()
    if fortran_order:
        raise ValueError(f"{path}:{name} is stored in Fortran order")
    return np.memmap(path, dtype=dtype, mode="r", offset=offset, shape=shape)


class TokenWindows(Dataset):
    """Windows of ``length`` model steps, ``stride`` control steps apart.

    Windows start at every ``start_every``-th control step of each episode, so
    all phases of the stride are used. Each item is the float16 tokens
    ``(length, N, C)`` of the window's steps and the grouped forces
    ``(length - 1, stride * A)`` between them. Tokens are memory-mapped from the
    embeddings' episode files, lazily in each data-loader worker.
    """

    def __init__(
        self,
        embeddings_dir: Path,
        episode_ids: Sequence[int],
        *,
        length: int,
        stride: int,
        start_every: int = 1,
    ):
        if length < 2:
            raise ValueError("windows need at least two steps")
        self.length = length
        self.stride = stride
        self.paths = [episode_path(embeddings_dir, int(i)) for i in episode_ids]
        self.actions = []
        index = []
        span = (length - 1) * stride
        for position, path in enumerate(self.paths):
            with np.load(path) as episode:
                actions = episode["actions"].astype(np.float32)
            self.actions.append(actions)
            num_steps = len(actions) + 1
            index += [(position, s) for s in range(0, num_steps - span, start_every)]
        if not index:
            raise ValueError(f"no episode is long enough for {length} steps")
        self.index = np.asarray(index, dtype=np.int64)
        self._tokens: dict[int, np.memmap] = {}

    def __len__(self) -> int:
        return len(self.index)

    def tokens(self, position: int) -> np.memmap:
        if position not in self._tokens:
            self._tokens[position] = npz_member_memmap(self.paths[position], "tokens")
        return self._tokens[position]

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        position, start = self.index[item]
        steps = start + self.stride * np.arange(self.length)
        tokens = np.ascontiguousarray(self.tokens(int(position))[steps])
        actions = group_actions(self.actions[position], steps[:-1], self.stride)
        return torch.from_numpy(tokens), torch.from_numpy(actions)


def ac_losses(
    model: ACWorldModel,
    latents: torch.Tensor,
    actions: torch.Tensor,
    rollout_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Teacher-forced and open-loop L1 losses on normalized token windows.

    Windows hold ``history + rollout_steps`` steps, as in ``predictor_losses``.
    """
    history = model.history
    if latents.shape[1] != history + rollout_steps:
        raise ValueError("windows must hold history + rollout_steps steps")
    latents = model.normalize(latents.float())
    teacher = model(latents[:, :history], actions[:, :history])
    one_step = torch.mean(torch.abs(teacher - latents[:, 1 : history + 1]))
    predicted = model.rollout_normalized(
        latents[:, :history], actions[:, : history - 1], actions[:, history - 1 :]
    )
    open_loop = torch.mean(torch.abs(predicted - latents[:, history:]))
    return one_step, open_loop


def _autocast(device: torch.device):
    return torch.autocast(
        device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
    )


@torch.no_grad()
def evaluate_ac_losses(
    model: ACWorldModel,
    windows: TokenWindows,
    items: Sequence[int],
    *,
    rollout_steps: int,
    batch_size: int,
    device: torch.device,
) -> tuple[float, float]:
    """Mean teacher-forced and open-loop losses over the given windows."""
    model.eval()
    sums = np.zeros(2)
    for start in range(0, len(items), batch_size):
        batch = [windows[int(i)] for i in items[start : start + batch_size]]
        tokens = torch.stack([b[0] for b in batch]).to(device)
        actions = torch.stack([b[1] for b in batch]).to(device)
        with _autocast(device):
            losses = ac_losses(model, tokens, actions, rollout_steps)
        sums += np.array([float(loss) for loss in losses]) * len(batch)
    one_step, open_loop = (sums / len(items)).tolist()
    return one_step, open_loop


def fit_ac_model(
    model: ACWorldModel,
    train: TokenWindows,
    validation: TokenWindows,
    *,
    rollout_steps: int,
    rollout_weight: float = 1.0,
    iterations: int = 3000,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    warmup: int = 100,
    weight_decay: float = 0.04,
    eval_every: int = 250,
    eval_windows: int = 512,
    num_workers: int = 4,
    seed: int = 0,
    device: torch.device,
    log: Callable[[str], None] = print,
) -> tuple[ACWorldModel, int, list[dict[str, float]]]:
    """AdamW with warmup and cosine decay, in bfloat16 autocast on CUDA.

    Validation runs every ``eval_every`` iterations on ``eval_windows`` windows
    spread over the validation set; the best weights are kept. Returns the
    model, the best iteration, and the logged losses.
    """
    torch.manual_seed(seed)
    model = model.to(device)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters, lr=learning_rate, weight_decay=weight_decay
    )

    def schedule(iteration: int) -> float:
        if iteration < warmup:
            return (iteration + 1) / warmup
        progress = (iteration - warmup) / max(1, iterations - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    sampler = RandomSampler(
        train,
        replacement=True,
        num_samples=iterations * batch_size,
        generator=torch.Generator().manual_seed(seed),
    )
    loader = DataLoader(
        train,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    items = np.linspace(0, len(validation) - 1, min(eval_windows, len(validation)))
    items = np.unique(items.astype(np.int64))

    def total(one_step, open_loop):
        return one_step + rollout_weight * open_loop

    best_loss, best_iteration, best_state = float("inf"), 0, None
    history = []
    sums, count = torch.zeros(2, device=device), 0
    model.train()
    for iteration, (tokens, actions) in enumerate(loader, start=1):
        tokens = tokens.to(device, non_blocking=True)
        actions = actions.to(device, non_blocking=True)
        with _autocast(device):
            losses = ac_losses(model, tokens, actions, rollout_steps)
        optimizer.zero_grad(set_to_none=True)
        total(*losses).backward()
        nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        scheduler.step()
        sums += torch.stack(losses).detach()
        count += 1
        if iteration % eval_every == 0 or iteration == iterations:
            train_losses = (sums / count).tolist()
            validation_losses = evaluate_ac_losses(
                model,
                validation,
                items,
                rollout_steps=rollout_steps,
                batch_size=batch_size,
                device=device,
            )
            model.train()
            record = {
                "iteration": iteration,
                "train_one_step": train_losses[0],
                "train_open_loop": train_losses[1],
                "validation_one_step": validation_losses[0],
                "validation_open_loop": validation_losses[1],
            }
            history.append(record)
            log(
                f"iteration {iteration}/{iterations}: train "
                f"{train_losses[0]:.4f}/{train_losses[1]:.4f}, validation "
                f"{validation_losses[0]:.4f}/{validation_losses[1]:.4f}"
            )
            sums, count = torch.zeros(2, device=device), 0
            if total(*validation_losses) < best_loss:
                best_loss, best_iteration = total(*validation_losses), iteration
                best_state = {
                    k: v.detach().to("cpu", copy=True)
                    for k, v in model.state_dict().items()
                }
    model.load_state_dict(best_state)
    return model.eval(), best_iteration, history


def ac_rollouts(model: ACWorldModel, probe) -> tuple[EpisodeRollout, EpisodeRollout]:
    """Open-loop rollouts from true context, as ``wm.eval`` episode rollouts.

    Episodes hold ``tokens`` ``(T+1, N, C)``, pooled ``features`` ``(T+1, C)``,
    and ``actions``. The first rollout returns probe-decoded observations, the
    second pooled latents (target ``features``). Both hold each model step's
    prediction over its ``step_stride`` control steps, so only horizons that
    are multiples of the stride are exact.
    """
    from world_model_playground.wm.planning import decode_latents

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
            np.asarray(episode["tokens"][context_steps], dtype=np.float32),
            device=device,
        )[None]
        actions = episode["actions"]
        context_actions = group_actions(actions, context_steps[:-1], stride)
        future = group_actions(actions, start + stride * np.arange(model_steps), stride)
        with _autocast(device):
            predicted = model.rollout(
                context,
                torch.as_tensor(context_actions, device=device)[None],
                torch.as_tensor(future, device=device)[None],
            )
        pooled = model.pooled(predicted.float())
        # The probe's history: true pooled features of earlier model steps.
        past_steps = np.maximum(start - stride * np.arange(probe.history)[::-1], 0)
        past = torch.as_tensor(episode["features"][past_steps], device=device)[None]
        observations = decode_latents(probe, past, pooled)[0].cpu().numpy()
        held = np.arange(num_steps) // stride
        return observations[held], pooled[0].cpu().numpy()[held]

    def observations(episode, start, num_steps):
        return predict(episode, start, num_steps)[0]

    def latents(episode, start, num_steps):
        return predict(episode, start, num_steps)[1]

    return observations, latents


def save_ac_model(
    path: Path, model: ACWorldModel, metadata: Mapping[str, object]
) -> None:
    """Saves the adapted predictor in bfloat16 with its config and metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "kind": KIND,
            "config": asdict(model.config),
            "state_dict": {
                k: v.to(torch.bfloat16) if v.is_floating_point() else v
                for k, v in model.state_dict().items()
            },
            "metadata": dict(metadata),
        },
        path,
    )


def is_ac_artifact(path: Path) -> bool:
    artifact = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    return artifact.get("kind") == KIND


def load_ac_model(
    path: Path,
    device: str | torch.device = "cpu",
    dtype: torch.dtype | None = None,
) -> tuple[ACWorldModel, dict]:
    """Restores a model written by :func:`save_ac_model`.

    Weights default to bfloat16 on CUDA and float32 on CPU.
    """
    artifact = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    config = replace(ACConfig(**artifact["config"]), pretrained=False)
    model = build_ac_model(config)
    model.load_state_dict(artifact["state_dict"])
    device = torch.device(device)
    if dtype is None:
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    return model.to(device=device, dtype=dtype).eval(), artifact["metadata"]

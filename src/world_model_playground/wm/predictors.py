"""Action-conditioned latent dynamics on frozen encoder features (DINO-WM style).

The encoder stays frozen, so a predictor trains entirely on the pooled features
cached by embed.py and its targets cannot collapse. A causal transformer reads
the latents of the last ``history`` steps, each summed with the embedding of the
action taken from that step, and predicts every next latent as a residual in
normalized feature space. Training adds to the teacher-forced one-step loss at
every position an open-loop loss over ``rollout_steps`` predictions fed back as
inputs, which is the regime planning runs in.

Predicted latents decode to observations through a state probe (``wm.probes``),
so open-loop error is reported in the units of the state-space models.
"""

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from world_model_playground.wm.data import ACTION_SIZE
from world_model_playground.wm.episodes import step_windows
from world_model_playground.wm.eval import EpisodeRollout
from world_model_playground.wm.linear import fit_normalization
from world_model_playground.wm.probes import StateProbe


@dataclass(frozen=True)
class PredictorConfig:
    feature_dim: int
    action_dim: int = ACTION_SIZE
    history: int = 4
    width: int = 256
    depth: int = 4
    heads: int = 4
    dropout: float = 0.0


class LatentPredictor(nn.Module):
    """Causal transformer from latent and action histories to next latents."""

    def __init__(self, config: PredictorConfig):
        super().__init__()
        if config.history < 1:
            raise ValueError("history must be at least 1")
        self.config = config
        self.latent_in = nn.Linear(config.feature_dim, config.width)
        self.action_in = nn.Linear(config.action_dim, config.width)
        self.position = nn.Parameter(torch.zeros(config.history, config.width))
        nn.init.trunc_normal_(self.position, std=0.02)
        layer = nn.TransformerEncoderLayer(
            config.width,
            config.heads,
            4 * config.width,
            config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(
            layer, config.depth, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(config.width)
        self.latent_out = nn.Linear(config.width, config.feature_dim)
        # A zero residual head starts training from the persistence model.
        nn.init.zeros_(self.latent_out.weight)
        nn.init.zeros_(self.latent_out.bias)
        self.register_buffer("feature_mean", torch.zeros(config.feature_dim))
        self.register_buffer("feature_std", torch.ones(config.feature_dim))
        self.register_buffer(
            "causal_mask",
            nn.Transformer.generate_square_subsequent_mask(config.history),
            persistent=False,
        )

    def set_normalization(self, features: np.ndarray) -> None:
        stats = fit_normalization(features)
        self.feature_mean.copy_(torch.from_numpy(stats.mean))
        self.feature_std.copy_(torch.from_numpy(stats.std))

    def normalize(self, latents: torch.Tensor) -> torch.Tensor:
        return (latents - self.feature_mean) / self.feature_std

    def denormalize(self, latents: torch.Tensor) -> torch.Tensor:
        return latents * self.feature_std + self.feature_mean

    def forward(self, latents: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Next normalized latents ``(B, L, D)`` after each of ``L <= history`` steps.

        ``latents`` are normalized ``(B, L, D)`` and ``actions[:, i]`` is the
        action taken from step ``i``; position ``i`` only attends to steps
        ``0..i``, so every prefix of a window is a valid shorter context.
        """
        length = latents.shape[1]
        if not 1 <= length <= self.config.history:
            raise ValueError(f"windows must hold 1 to {self.config.history} steps")
        tokens = self.latent_in(latents) + self.action_in(actions)
        tokens = tokens + self.position[:length]
        hidden = self.blocks(
            tokens, mask=self.causal_mask[:length, :length], is_causal=True
        )
        return latents + self.latent_out(self.norm(hidden))

    def rollout_normalized(
        self,
        context: torch.Tensor,
        context_actions: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Feeds predictions back for ``H`` steps in normalized latent space.

        ``context`` ``(B, L, D)`` holds the latents up to the current step,
        ``context_actions`` ``(B, L-1, A)`` the actions between them, and
        ``actions`` ``(B, H, A)`` the actions taken from the current step on.
        Returns the ``H`` predicted latents after each action.
        """
        if context_actions.shape[1] != context.shape[1] - 1:
            raise ValueError("context_actions must hold one action fewer than context")

        def between(steps: torch.Tensor, step_actions: torch.Tensor) -> torch.Tensor:
            """The last ``len(steps) - 1`` actions, those between ``steps``."""
            return step_actions[:, step_actions.shape[1] - steps.shape[1] + 1 :]

        history = self.config.history
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
        """:meth:`rollout_normalized` on raw encoder features."""
        predictions = self.rollout_normalized(
            self.normalize(context), context_actions, actions
        )
        return self.denormalize(predictions)


class LatentWindows:
    """Every window of ``length`` consecutive steps within one episode, on a device.

    Episodes are concatenated with one padding action after each final step, so
    step ``i`` of the flat arrays carries both its latent and the action taken
    from it, and a window's ``length - 1`` actions are those between its steps.
    """

    def __init__(
        self,
        episodes: Mapping[int, Mapping[str, np.ndarray]],
        length: int,
        device: torch.device,
    ):
        features, actions, starts = [], [], []
        offset = 0
        for episode in episodes.values():
            steps = len(episode["features"])
            if len(episode["actions"]) != steps - 1:
                raise ValueError("an episode needs one action fewer than features")
            features.append(episode["features"])
            actions.append(episode["actions"])
            actions.append(np.zeros_like(episode["actions"][:1]))
            starts.append(offset + np.arange(max(0, steps - length + 1)))
            offset += steps
        if not features:
            raise ValueError("no episodes to build latent windows from")
        self.length = length
        self.features = torch.as_tensor(np.concatenate(features), device=device)
        self.actions = torch.as_tensor(np.concatenate(actions), device=device)
        self.starts = torch.as_tensor(np.concatenate(starts), device=device)
        if len(self.starts) == 0:
            raise ValueError(f"no episode has {length} steps for a full window")

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Latents ``(B, length, D)`` and actions ``(B, length-1, A)``."""
        steps = self.starts[rows, None] + torch.arange(
            self.length, device=self.starts.device
        )
        return self.features[steps], self.actions[steps[:, :-1]]


def predictor_losses(
    predictor: LatentPredictor,
    latents: torch.Tensor,
    actions: torch.Tensor,
    rollout_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Teacher-forced and open-loop MSE in normalized latent units.

    Windows hold ``history + rollout_steps`` latents. The teacher-forced loss
    predicts steps ``1..history`` from the first ``history``; the open-loop loss
    predicts the last ``rollout_steps`` from the first ``history`` alone.
    """
    history = predictor.config.history
    if latents.shape[1] != history + rollout_steps:
        raise ValueError("windows must hold history + rollout_steps latents")
    latents = predictor.normalize(latents)
    teacher = predictor(latents[:, :history], actions[:, :history])
    one_step = torch.mean(torch.square(teacher - latents[:, 1 : history + 1]))
    predicted = predictor.rollout_normalized(
        latents[:, :history], actions[:, : history - 1], actions[:, history - 1 :]
    )
    open_loop = torch.mean(torch.square(predicted - latents[:, history:]))
    return one_step, open_loop


def fit_predictor(
    train_episodes: Mapping[int, Mapping[str, np.ndarray]],
    validation_episodes: Mapping[int, Mapping[str, np.ndarray]],
    *,
    config: PredictorConfig,
    rollout_steps: int = 8,
    rollout_weight: float = 1.0,
    epochs: int = 100,
    batch_size: int = 256,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.05,
    seed: int = 0,
    device: torch.device,
) -> tuple[LatentPredictor, int, list[dict[str, float]]]:
    """Trains with AdamW and keeps the epoch with the lowest validation loss.

    Returns the predictor, the best epoch, and per-epoch mean losses.
    """
    if rollout_steps < 1:
        raise ValueError("rollout_steps must be at least 1")
    torch.manual_seed(seed)
    length = config.history + rollout_steps
    train = LatentWindows(train_episodes, length, device)
    validation = LatentWindows(validation_episodes, length, device)
    predictor = LatentPredictor(config).to(device)
    predictor.set_normalization(
        np.concatenate([e["features"] for e in train_episodes.values()])
    )
    optimizer = torch.optim.AdamW(
        predictor.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    generator = torch.Generator(device=device).manual_seed(seed)

    def total(one_step, open_loop):
        return one_step + rollout_weight * open_loop

    best_loss, best_epoch, best_state = float("inf"), 0, None
    history = []
    for epoch in range(1, epochs + 1):
        predictor.train()
        sums = torch.zeros(2, device=device)
        order = torch.randperm(len(train), device=device, generator=generator)
        for start in range(0, len(order), batch_size):
            rows = order[start : start + batch_size]
            losses = predictor_losses(predictor, *train[rows], rollout_steps)
            optimizer.zero_grad(set_to_none=True)
            total(*losses).backward()
            nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
            optimizer.step()
            sums += torch.stack(losses).detach() * len(rows)
        scheduler.step()
        train_losses = (sums / len(train)).tolist()
        validation_losses = evaluate_losses(
            predictor, validation, rollout_steps, batch_size
        )
        validation_loss = total(*validation_losses)
        history.append(
            {
                "train_one_step": train_losses[0],
                "train_open_loop": train_losses[1],
                "validation_one_step": validation_losses[0],
                "validation_open_loop": validation_losses[1],
            }
        )
        if validation_loss < best_loss:
            best_loss, best_epoch = validation_loss, epoch
            best_state = {k: v.clone() for k, v in predictor.state_dict().items()}
    predictor.load_state_dict(best_state)
    return predictor.eval(), best_epoch, history


@torch.no_grad()
def evaluate_losses(
    predictor: LatentPredictor,
    windows: LatentWindows,
    rollout_steps: int,
    batch_size: int,
) -> tuple[float, float]:
    """Mean teacher-forced and open-loop losses over every window."""
    predictor.eval()
    sums = torch.zeros(2, device=windows.features.device)
    rows = torch.arange(len(windows), device=windows.features.device)
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        losses = predictor_losses(predictor, *windows[batch], rollout_steps)
        sums += torch.stack(losses) * len(batch)
    one_step, open_loop = (sums / len(windows)).tolist()
    return one_step, open_loop


def decode_steps(probe: StateProbe, features: np.ndarray, steps: slice) -> np.ndarray:
    """Probe observations at ``steps`` of a feature sequence, stacking its history."""
    windows = step_windows(len(features), probe.history)[steps]
    return probe.predict(features[windows].reshape(len(windows), -1))


def latent_rollout(predictor: LatentPredictor, probe: StateProbe) -> EpisodeRollout:
    """Open-loop latent rollouts from true context, decoded to observations.

    The context is the true features of up to ``history`` steps ending at the
    start step; the probe stacks true features before it with the predictions.
    """
    device = predictor.feature_mean.device
    history = predictor.config.history

    @torch.no_grad()
    def rollout(episode, start, num_steps):
        features, actions = episode["features"], episode["actions"]
        first = max(0, start - history + 1)
        predicted = predictor.rollout(
            torch.as_tensor(features[None, first : start + 1], device=device),
            torch.as_tensor(actions[None, first:start], device=device),
            torch.as_tensor(actions[None, start : start + num_steps], device=device),
        )[0]
        sequence = np.concatenate(
            (features[: start + 1], predicted.cpu().numpy().astype(features.dtype))
        )
        return decode_steps(probe, sequence, slice(start + 1, None))

    return rollout


def normalized_latent_rollout(predictor: LatentPredictor) -> EpisodeRollout:
    """Open-loop rollouts returning normalized predicted latents (target ``latents``)."""
    device = predictor.feature_mean.device
    history = predictor.config.history

    @torch.no_grad()
    def rollout(episode, start, num_steps):
        features, actions = episode["features"], episode["actions"]
        first = max(0, start - history + 1)
        context = predictor.normalize(
            torch.as_tensor(features[None, first : start + 1], device=device)
        )
        predicted = predictor.rollout_normalized(
            context,
            torch.as_tensor(actions[None, first:start], device=device),
            torch.as_tensor(actions[None, start : start + num_steps], device=device),
        )
        return predicted[0].cpu().numpy()

    return rollout


def save_predictor(
    path: Path, predictor: LatentPredictor, metadata: Mapping[str, object]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": asdict(predictor.config),
            "state_dict": predictor.state_dict(),
            "metadata": dict(metadata),
        },
        path,
    )


def load_predictor(
    path: Path, device: str | torch.device = "cpu"
) -> tuple[LatentPredictor, dict]:
    """Restores a predictor written by :func:`save_predictor` and its metadata."""
    artifact = torch.load(path, map_location=device, weights_only=True)
    predictor = LatentPredictor(PredictorConfig(**artifact["config"]))
    predictor.load_state_dict(artifact["state_dict"])
    return predictor.to(device).eval(), artifact["metadata"]

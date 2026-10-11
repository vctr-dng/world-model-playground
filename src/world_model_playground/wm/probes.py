"""State probes that read the CartPole observation back out of frozen features.

A probe maps the features of the last ``history`` steps, concatenated, to the
5-dimensional observation of the newest step. How well it does tells which
encoder keeps the cart position, pole angle, and velocities; the fitted probe is
then the decoder from latents back to observations for costs and for metrics
comparable with the state-space models.

Linear probes are closed-form ridge regressions and MLP probes are trained with
AdamW; both are stored as the same ``StateProbe`` module.
"""

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import torch
from torch import nn

from world_model_playground.wm.data import OBSERVATION_NAMES, project_angle_pair
from world_model_playground.wm.episodes import step_windows
from world_model_playground.wm.eval import rmse
from world_model_playground.wm.linear import fit_normalization

# Encoders pass the Phase 2 gate when every one of these exceeds the threshold.
GATE_OBSERVATIONS = ("cart_position", "cos_theta", "sin_theta")
GATE_R2 = 0.95


def stack_history(features: np.ndarray, history: int) -> np.ndarray:
    """Concatenates each step's features with those of the ``history-1`` before it."""
    return features[step_windows(len(features), history)].reshape(len(features), -1)


def probe_arrays(
    episodes: Mapping[int, Mapping[str, np.ndarray]], history: int
) -> tuple[np.ndarray, np.ndarray]:
    """Stacked features ``(N, history*D)`` and observations ``(N, 5)`` of all steps."""
    if not episodes:
        raise ValueError("no episodes to build probe arrays from")
    features = [stack_history(e["features"], history) for e in episodes.values()]
    observations = [e["observations"] for e in episodes.values()]
    return (
        np.concatenate(features).astype(np.float32),
        np.concatenate(observations).astype(np.float32),
    )


class StateProbe(nn.Module):
    """Normalized features -> linear or MLP head -> denormalized observations."""

    def __init__(
        self,
        input_dim: int,
        *,
        history: int,
        hidden_sizes: Sequence[int] = (),
        output_dim: int = len(OBSERVATION_NAMES),
    ):
        super().__init__()
        self.history = history
        self.hidden_sizes = tuple(hidden_sizes)
        layers: list[nn.Module] = []
        width = input_dim
        for hidden in self.hidden_sizes:
            layers += [nn.Linear(width, hidden), nn.GELU()]
            width = hidden
        layers.append(nn.Linear(width, output_dim))
        self.head = nn.Sequential(*layers)
        for name, size in (("feature", input_dim), ("target", output_dim)):
            self.register_buffer(f"{name}_mean", torch.zeros(size))
            self.register_buffer(f"{name}_std", torch.ones(size))

    @property
    def kind(self) -> str:
        return "mlp" if self.hidden_sizes else "linear"

    def set_normalization(self, features: np.ndarray, targets: np.ndarray) -> None:
        for name, values in (("feature", features), ("target", targets)):
            stats = fit_normalization(values)
            getattr(self, f"{name}_mean").copy_(torch.from_numpy(stats.mean))
            getattr(self, f"{name}_std").copy_(torch.from_numpy(stats.std))

    def normalized(self, features: torch.Tensor) -> torch.Tensor:
        """Head output in normalized target units, for training losses."""
        return self.head((features - self.feature_mean) / self.feature_std)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.normalized(features) * self.target_std + self.target_mean

    @torch.no_grad()
    def predict(self, features: np.ndarray, batch_size: int = 8192) -> np.ndarray:
        """Observations for stacked features, with (cos, sin) on the unit circle."""
        device = self.feature_mean.device
        outputs = [
            self(torch.as_tensor(features[i : i + batch_size], device=device)).cpu()
            for i in range(0, len(features), batch_size)
        ]
        return project_angle_pair(torch.cat(outputs).double().numpy())


def regression_metrics(
    predictions: np.ndarray, targets: np.ndarray
) -> dict[str, object]:
    """Per-observation R² and RMSE, plus the absolute pole-angle error in radians."""
    predictions = np.asarray(predictions, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    residual = np.square(predictions - targets).sum(axis=0)
    total = np.square(targets - targets.mean(axis=0)).sum(axis=0)
    r2 = 1.0 - residual / np.where(total > 0, total, np.nan)
    angle_error = np.angle(
        np.exp(
            1j
            * (
                np.arctan2(predictions[:, 2], predictions[:, 1])
                - np.arctan2(targets[:, 2], targets[:, 1])
            )
        )
    )
    return {
        "r2": dict(zip(OBSERVATION_NAMES, r2.tolist(), strict=True)),
        "rmse": rmse(predictions, targets),
        "angle_mae_rad": float(np.abs(angle_error).mean()),
    }


def passes_gate(metrics: Mapping[str, object]) -> bool:
    return all(metrics["r2"][name] >= GATE_R2 for name in GATE_OBSERVATIONS)


def _normalized_mse(probe: StateProbe, features: torch.Tensor, targets: torch.Tensor):
    goal = (targets - probe.target_mean) / probe.target_std
    return torch.mean(torch.square(probe.normalized(features) - goal))


def fit_linear_probe(
    train: tuple[np.ndarray, np.ndarray],
    validation: tuple[np.ndarray, np.ndarray],
    *,
    history: int,
    ridges: Sequence[float],
    device: torch.device,
) -> tuple[StateProbe, float]:
    """Ridge regression on normalized features, picking the ridge on validation.

    Features and targets are centered, so the intercept is zero. One
    eigendecomposition of the Gram matrix solves every ridge strength.
    """
    features, targets = train
    probe = StateProbe(features.shape[1], history=history).to(device)
    probe.set_normalization(features, targets)
    with torch.no_grad():
        x = (torch.as_tensor(features, device=device) - probe.feature_mean) / (
            probe.feature_std
        )
        y = (torch.as_tensor(targets, device=device) - probe.target_mean) / (
            probe.target_std
        )
        gram = (x.T @ x).double().cpu()
        cross = (x.T @ y).double().cpu()
        eigenvalues, eigenvectors = torch.linalg.eigh(gram)
        projected = eigenvectors.T @ cross

        val_x = torch.as_tensor(validation[0], device=device)
        val_y = torch.as_tensor(validation[1], device=device)
        best = None
        for ridge in ridges:
            weights = eigenvectors @ (projected / (eigenvalues + ridge)[:, None])
            probe.head[0].weight.copy_(weights.T.float())
            probe.head[0].bias.zero_()
            loss = float(_normalized_mse(probe, val_x, val_y))
            if best is None or loss < best[0]:
                best = (loss, ridge, weights)
        _, ridge, weights = best
        probe.head[0].weight.copy_(weights.T.float())
    return probe.eval(), float(ridge)


def fit_mlp_probe(
    train: tuple[np.ndarray, np.ndarray],
    validation: tuple[np.ndarray, np.ndarray],
    *,
    history: int,
    hidden_sizes: Sequence[int] = (512, 512),
    epochs: int = 100,
    batch_size: int = 1024,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    seed: int = 0,
    device: torch.device,
) -> tuple[StateProbe, int]:
    """Trains an MLP probe and keeps the epoch with the lowest validation loss."""
    torch.manual_seed(seed)
    features, targets = train
    probe = StateProbe(
        features.shape[1], history=history, hidden_sizes=hidden_sizes
    ).to(device)
    probe.set_normalization(features, targets)
    x = torch.as_tensor(features, device=device)
    y = torch.as_tensor(targets, device=device)
    val_x = torch.as_tensor(validation[0], device=device)
    val_y = torch.as_tensor(validation[1], device=device)
    optimizer = torch.optim.AdamW(
        probe.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    generator = torch.Generator(device=device).manual_seed(seed)

    best_loss, best_epoch, best_state = float("inf"), 0, None
    for epoch in range(1, epochs + 1):
        probe.train()
        order = torch.randperm(len(x), device=device, generator=generator)
        for start in range(0, len(x), batch_size):
            rows = order[start : start + batch_size]
            loss = _normalized_mse(probe, x[rows], y[rows])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        scheduler.step()
        probe.eval()
        with torch.no_grad():
            validation_loss = float(_normalized_mse(probe, val_x, val_y))
        if validation_loss < best_loss:
            best_loss, best_epoch = validation_loss, epoch
            best_state = {k: v.clone() for k, v in probe.state_dict().items()}
    probe.load_state_dict(best_state)
    return probe.eval(), best_epoch


def save_probe(path: Path, probe: StateProbe, metadata: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "input_dim": probe.head[0].in_features,
            "history": probe.history,
            "hidden_sizes": list(probe.hidden_sizes),
            "state_dict": probe.state_dict(),
            "metadata": dict(metadata),
        },
        path,
    )


def load_probe(
    path: Path, device: str | torch.device = "cpu"
) -> tuple[StateProbe, dict]:
    """Restores a probe written by :func:`save_probe` and its metadata."""
    artifact = torch.load(path, map_location=device, weights_only=True)
    probe = StateProbe(
        artifact["input_dim"],
        history=artifact["history"],
        hidden_sizes=artifact["hidden_sizes"],
    )
    probe.load_state_dict(artifact["state_dict"])
    return probe.to(device).eval(), artifact["metadata"]

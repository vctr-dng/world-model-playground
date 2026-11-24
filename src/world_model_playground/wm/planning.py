"""Batched sampling-based planning (CEM) with learned CartPole world models.

A planner searches action sequences for a batch of ``B`` independent episodes at
once. Each iteration samples ``N`` candidates per episode, scores them with an
*evaluator*, a callable mapping actions ``(B, N, H, A)`` to an
:class:`Evaluation` whose ``returns`` are ``(B, N)``, and refits a Gaussian on
the best candidates. Candidates are parametrized by ``H / action_repeat`` knots,
each held for ``action_repeat`` steps, which lengthens the horizon a fixed
sample budget covers.

Evaluators wrap a model and an objective:

- :func:`latent_evaluator` rolls a latent world model (:class:`LatentWorldModel`:
  the Phase 3 predictor on pooled features, or V-JEPA 2-AC on patch tokens) out
  from the true recent latents and scores either the CartPole reward of the
  probe-decoded observations or the distance to a goal latent. A model step may
  span several control steps (``step_stride``); plans then hold each knot for
  exactly one model step (``action_repeat == step_stride``).
- :func:`state_evaluator` rolls an observation-space model (e.g. the linear
  baseline, see :func:`linear_dynamics`) out from the true observation.

:func:`cartpole_reward` is the dense ``CartpoleBalance`` reward computed from an
observation and an action, so plans optimize exactly what episodes are scored on.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

import torch

from world_model_playground.wm.linear import LinearResidualDynamics
from world_model_playground.wm.probes import StateProbe


class LatentWorldModel(Protocol):
    """What planners and diagnostics need from a latent world model.

    Latents have a per-step shape (``(D,)`` pooled features or ``(N, C)``
    tokens); ``pooled`` maps them to the per-step features probes decode.
    Actions are grouped per model step into ``step_stride * A`` values.
    """

    history: int
    step_stride: int
    goal_distance: str  # "l1" or "l2", in normalized latent units
    latent_kind: str  # "pooled" or "tokens"

    def normalize(self, latents: torch.Tensor) -> torch.Tensor: ...

    def denormalize(self, latents: torch.Tensor) -> torch.Tensor: ...

    def pooled(self, latents: torch.Tensor) -> torch.Tensor: ...

    def rollout_normalized(
        self,
        context: torch.Tensor,
        context_actions: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor: ...

    def rollout(
        self,
        context: torch.Tensor,
        context_actions: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor: ...


def tolerance(
    x: torch.Tensor,
    *,
    margin: float,
    sigmoid: str = "gaussian",
    value_at_margin: float = 0.1,
) -> torch.Tensor:
    """``mujoco_playground.reward.tolerance`` with bounds ``(0, 0)``."""
    distance = x.abs() / margin
    if sigmoid == "gaussian":
        scale = math.sqrt(-2 * math.log(value_at_margin))
        return torch.exp(-0.5 * (distance * scale) ** 2)
    if sigmoid == "quadratic":
        scaled = distance * math.sqrt(1 - value_at_margin)
        return torch.where(scaled < 1, 1 - scaled**2, torch.zeros_like(scaled))
    raise ValueError(f"unsupported sigmoid {sigmoid!r}")


def cartpole_reward(observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    """Dense CartpoleBalance reward of reaching ``observations`` by ``actions``.

    ``observations`` are ``(..., 5)`` ``[cart_pos, cos, sin, cart_vel, ang_vel]``
    and ``actions`` ``(..., 1)``; the reward lies in ``[0, 1]``.
    """
    upright = (observations[..., 1] + 1) / 2
    centered = (1 + tolerance(observations[..., 0], margin=2)) / 2
    small_control = (
        4
        + tolerance(actions[..., 0], margin=1, sigmoid="quadratic", value_at_margin=0.0)
    ) / 5
    small_velocity = (1 + tolerance(observations[..., 4], margin=5)) / 2
    return upright * centered * small_control * small_velocity


@dataclass(frozen=True)
class CEMConfig:
    """Cross-entropy method over piecewise-constant action sequences."""

    horizon: int = 30
    candidates: int = 256
    iterations: int = 4
    elites: int = 32
    action_repeat: int = 2
    init_std: float = 0.7
    min_std: float = 0.05
    smoothing: float = 0.1
    action_low: float = -1.0
    action_high: float = 1.0

    def __post_init__(self):
        if self.horizon < 1 or self.candidates < 2 or self.iterations < 1:
            raise ValueError("horizon, candidates, and iterations must be positive")
        if not 1 <= self.elites <= self.candidates:
            raise ValueError("elites must be between 1 and candidates")
        if self.action_repeat < 1:
            raise ValueError("action_repeat must be at least 1")

    @property
    def knots(self) -> int:
        return math.ceil(self.horizon / self.action_repeat)


@dataclass
class Evaluation:
    """Scores of candidate plans and per-candidate tensors to keep for the chosen one.

    ``returns`` are ``(B, N)``; every ``extras`` tensor is ``(B, N, ...)``.
    """

    returns: torch.Tensor
    extras: dict[str, torch.Tensor] = field(default_factory=dict)


Evaluator = Callable[[torch.Tensor], Evaluation]


@dataclass
class Plan:
    """The best candidate of the last iteration for every episode."""

    actions: torch.Tensor
    expected_return: torch.Tensor
    extras: dict[str, torch.Tensor]

    @property
    def first_action(self) -> torch.Tensor:
        return self.actions[:, 0]


class CEMPlanner:
    """Receding-horizon CEM for ``batch_size`` episodes, warm-started across steps."""

    def __init__(
        self,
        config: CEMConfig,
        *,
        batch_size: int,
        action_dim: int = 1,
        device: torch.device | str = "cpu",
        seed: int = 0,
        shift: int = 1,
    ):
        """``shift`` is the number of control steps between successive plans,
        by which the warm start is advanced."""
        if shift < 1:
            raise ValueError("shift must be at least 1")
        self.config = config
        self.shift = shift
        self.batch_size = batch_size
        self.action_dim = action_dim
        self.device = torch.device(device)
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        self.reset()

    def reset(self) -> None:
        """Forgets the warm start, e.g. at the start of new episodes."""
        shape = (self.batch_size, self.config.knots, self.action_dim)
        self.mean = torch.zeros(shape, device=self.device)

    def expand(self, knots: torch.Tensor) -> torch.Tensor:
        """Holds each knot ``(..., K, A)`` for ``action_repeat`` steps -> ``(..., H, A)``."""
        steps = knots.repeat_interleave(self.config.action_repeat, dim=-2)
        return steps[..., : self.config.horizon, :]

    def plan(self, evaluate: Evaluator) -> Plan:
        config = self.config
        mean = self.mean
        std = torch.full_like(mean, config.init_std)
        batch = torch.arange(self.batch_size, device=self.device)
        for _ in range(config.iterations):
            noise = torch.randn(
                (self.batch_size, config.candidates, *mean.shape[1:]),
                device=self.device,
                generator=self.generator,
            )
            knots = mean[:, None] + std[:, None] * noise
            knots[:, 0] = mean  # always reconsider the current mean plan
            knots = knots.clamp(config.action_low, config.action_high)
            evaluation = evaluate(self.expand(knots))
            elite_rows = evaluation.returns.topk(config.elites, dim=1).indices
            elites = knots[batch[:, None], elite_rows]
            mean = config.smoothing * mean + (1 - config.smoothing) * elites.mean(dim=1)
            std = (
                config.smoothing * std
                + (1 - config.smoothing) * elites.std(dim=1, unbiased=False)
            ).clamp_min(config.min_std)

        best = evaluation.returns.argmax(dim=1)
        plan = Plan(
            actions=self.expand(knots[batch, best]),
            expected_return=evaluation.returns[batch, best],
            extras={k: v[batch, best] for k, v in evaluation.extras.items()},
        )
        self.mean = self._shifted(knots[batch, best])
        return plan

    def _shifted(self, knots: torch.Tensor) -> torch.Tensor:
        """Warm start: the plan advanced by ``shift`` steps, repeating its last action."""
        steps = self.expand(knots)
        shift = min(self.shift, steps.shape[1])
        steps = torch.cat((steps[:, shift:], steps[:, -1:].expand(-1, shift, -1)), 1)
        return steps[:, :: self.config.action_repeat]


def decode_latents(
    probe: StateProbe, past: torch.Tensor, predicted: torch.Tensor
) -> torch.Tensor:
    """Probe observations ``(M, H, 5)`` of predicted latents ``(M, H, D)``.

    ``past`` ``(M, P, D)`` holds the true latents before the predictions, of
    which the probe's history uses the last ``probe.history - 1``.
    """
    context = probe.history - 1
    if past.shape[1] < context:
        # Episode start: repeat the first step, as step windows do.
        pad = past[:, :1].expand(-1, context - past.shape[1], -1)
        past = torch.cat((pad, past), dim=1)
    sequence = torch.cat((past[:, past.shape[1] - context :], predicted), dim=1)
    windows = sequence.unfold(1, probe.history, 1)  # (M, H, D, history)
    stacked = windows.permute(0, 1, 3, 2).flatten(2)
    return probe(stacked)


def group_step_actions(actions: torch.Tensor, stride: int) -> torch.Tensor:
    """Control-step actions ``(..., H, A)`` -> model-step groups ``(..., H/k, k*A)``."""
    horizon = actions.shape[-2]
    if horizon % stride:
        raise ValueError(f"the horizon must be a multiple of the model stride {stride}")
    return actions.unflatten(-2, (horizon // stride, stride)).flatten(-2)


def latent_distance(
    difference: torch.Tensor, kind: str, latent_dims: int
) -> torch.Tensor:
    """Mean L1 or squared distance over the trailing ``latent_dims`` dimensions."""
    values = difference.abs() if kind == "l1" else difference.square()
    return values.flatten(-latent_dims).mean(-1)


def latent_evaluator(
    model: LatentWorldModel,
    probe: StateProbe,
    context: torch.Tensor,
    context_actions: torch.Tensor,
    *,
    objective: str = "reward",
    goal: torch.Tensor | None = None,
    goal_cost: str = "sum",
) -> Evaluator:
    """Scores candidates by rolling the latent model out from true context.

    ``context`` ``(B, L, *latent)`` holds raw latents of the model steps up to
    the current step and ``context_actions`` ``(B, L-1, k*A)`` the grouped
    actions between them. Candidate actions ``(B, N, H, A)`` are grouped into
    ``H / k`` model steps. ``objective`` is ``"reward"`` (CartPole reward of
    the probe-decoded observations, each model step counting for its ``k``
    control steps) or ``"goal"`` (negative distance to the raw ``goal`` latent
    in normalized units, the model's ``goal_distance``, summed over the
    horizon, or, with ``goal_cost="final"``, of the last step only, as LeWM
    plans). Extras: decoded ``observations`` ``(B, N, H/k, 5)`` and the
    normalized ``first_latent`` ``(B, N, *latent)``.
    """
    if objective not in ("reward", "goal"):
        raise ValueError("objective must be 'reward' or 'goal'")
    if objective == "goal" and goal is None:
        raise ValueError("the goal objective needs a goal latent")
    if goal_cost not in ("sum", "final"):
        raise ValueError("goal_cost must be 'sum' or 'final'")
    stride = model.step_stride
    if probe.stride != stride:
        raise ValueError("the probe must stack steps model.step_stride apart")
    latent_dims = context.ndim - 2
    normalized_context = model.normalize(context)
    pooled_context = model.pooled(context)
    normalized_goal = None if goal is None else model.normalize(goal)

    @torch.no_grad()
    def evaluate(actions: torch.Tensor) -> Evaluation:
        batch, candidates = actions.shape[:2]
        flat_actions = actions.flatten(0, 1)

        def tile(values: torch.Tensor) -> torch.Tensor:
            return values.repeat_interleave(candidates, dim=0)

        predicted = model.rollout_normalized(
            tile(normalized_context),
            tile(context_actions),
            group_step_actions(flat_actions, stride),
        ).float()
        observations = decode_latents(
            probe,
            tile(pooled_context),
            model.pooled(model.denormalize(predicted)),
        )
        if objective == "reward":
            step_actions = flat_actions[:, ::stride]
            returns = stride * cartpole_reward(observations, step_actions).sum(1)
        else:
            distance = latent_distance(
                predicted - normalized_goal, model.goal_distance, latent_dims
            )
            returns = -(distance[:, -1] if goal_cost == "final" else distance.sum(1))
        return Evaluation(
            returns.view(batch, candidates),
            {
                "observations": observations.view(batch, candidates, -1, 5),
                "first_latent": predicted[:, 0].view(
                    batch, candidates, *predicted.shape[2:]
                ),
            },
        )

    return evaluate


StateDynamics = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def state_evaluator(dynamics: StateDynamics, observations: torch.Tensor) -> Evaluator:
    """Scores candidates by the CartPole reward of an observation-space rollout.

    ``dynamics`` maps observations ``(M, 5)`` and actions ``(M, A)`` to the next
    observations; rollouts start from the true ``observations`` ``(B, 5)``.
    """

    @torch.no_grad()
    def evaluate(actions: torch.Tensor) -> Evaluation:
        batch, candidates, horizon = actions.shape[:3]
        state = observations.repeat_interleave(candidates, dim=0)
        flat_actions = actions.flatten(0, 1)
        trajectory = []
        for step in range(horizon):
            state = dynamics(state, flat_actions[:, step])
            trajectory.append(state)
        predicted = torch.stack(trajectory, dim=1)
        returns = cartpole_reward(predicted, flat_actions).sum(1)
        return Evaluation(
            returns.view(batch, candidates),
            {"observations": predicted.view(batch, candidates, horizon, -1)},
        )

    return evaluate


def linear_dynamics(
    model: LinearResidualDynamics, device: torch.device | str = "cpu"
) -> StateDynamics:
    """The linear residual model's ``predict_next`` on float32 torch tensors."""

    def tensor(values) -> torch.Tensor:
        return torch.as_tensor(values, dtype=torch.float32, device=device)

    feature_mean = tensor(model.feature_normalization.mean)
    feature_std = tensor(model.feature_normalization.std)
    delta_mean = tensor(model.delta_normalization.mean)
    delta_std = tensor(model.delta_normalization.std)
    weights = tensor(model.weights)
    bias = tensor(model.bias)

    def predict_next(observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        features = torch.cat((observations, actions), dim=-1)
        delta = ((features - feature_mean) / feature_std) @ weights.T + bias
        following = observations + delta * delta_std + delta_mean
        return project_angle(following)

    return predict_next


def project_angle(observations: torch.Tensor) -> torch.Tensor:
    """Puts the (cos, sin) pair of observations back on the unit circle."""
    pair = observations[..., 1:3]
    norm = pair.norm(dim=-1, keepdim=True)
    upright = pair.new_tensor([1.0, 0.0]).expand_as(pair)
    pair = torch.where(norm > 1e-12, pair / norm.clamp_min(1e-12), upright)
    return torch.cat((observations[..., :1], pair, observations[..., 3:]), dim=-1)

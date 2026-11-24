"""Controllers for closed-loop CartPole control and the episode runner.

A controller maps the environment state of a batch of episodes to one action
per episode. Planning controllers run CEM (``wm.planning``) through a model of
the dynamics: the true MJX environment, a linear or other state-space model, or
a latent world model fed with the encoded rendered frames. ``run_episodes``
runs any controller on a batch of episodes in lockstep, and ``summarize``
scores the result.
"""

import time

import jax
import jax.numpy as jnp
import numpy as np
import torch

from world_model_playground.cartpole import FrameRenderer
from world_model_playground.wm.episodes import step_windows
from world_model_playground.wm.planning import (
    CEMPlanner,
    Evaluation,
    decode_latents,
    latent_evaluator,
    state_evaluator,
)

MIN_ENCODER_BATCH = 16


def to_torch(values, device) -> torch.Tensor:
    return torch.as_tensor(np.array(values, dtype=np.float32), device=device)


class PPOController:
    def __init__(self, policy, seed: int):
        self.policy = jax.jit(policy)
        self.key = jax.random.key(seed)

    def reset(self) -> None:
        pass

    def act(self, state, observations, qpos, qvel):
        self.key, key = jax.random.split(self.key)
        return np.asarray(self.policy(state.obs, key)[0]), {}


class RandomController:
    def __init__(self, batch_size: int, seed: int):
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)

    def reset(self) -> None:
        pass

    def act(self, state, observations, qpos, qvel):
        actions = self.rng.uniform(-1.0, 1.0, size=(self.batch_size, 1))
        return actions.astype(np.float32), {}


class PlanningController:
    """Shared CEM bookkeeping: replanning cadence, timing, and the chosen plan.

    A plan is made every ``replan_every`` control steps and its first
    ``replan_every`` actions are executed. In between, the plan's info is
    repeated with ``plan_seconds`` 0, so its mean is the cost per control step.
    """

    # Control steps between a plan's first and second predicted observations.
    prediction_stride = 1

    def __init__(self, planner: CEMPlanner, replan_every: int = 1):
        if planner.config.horizon < replan_every:
            raise ValueError("the horizon must cover the steps between plans")
        self.planner = planner
        self.replan_every = replan_every

    def reset(self) -> None:
        self.planner.reset()
        self.step = 0
        self.queue = None
        self.plan_info = {}

    def observe(self, state, observations, qpos, qvel) -> dict:
        """Per-step bookkeeping before planning; returns per-step info."""
        return {}

    def evaluator(self, state, observations, qpos, qvel):
        raise NotImplementedError

    def planned(self, plan) -> None:
        """Called with every new plan."""

    def act(self, state, observations, qpos, qvel):
        info = self.observe(state, observations, qpos, qvel)
        offset = self.step % self.replan_every
        plan_seconds = 0.0
        if offset == 0:
            start = time.perf_counter()
            plan = self.planner.plan(self.evaluator(state, observations, qpos, qvel))
            plan_seconds = time.perf_counter() - start
            self.queue = plan.actions[:, : self.replan_every].cpu().numpy()
            self.plan_info = {"expected_return": plan.expected_return.cpu().numpy()}
            if "observations" in plan.extras:
                self.plan_info["plan_observations"] = (
                    plan.extras["observations"].cpu().numpy()
                )
            self.planned(plan)
        self.step += 1
        batch = self.planner.batch_size
        return self.queue[:, offset], info | self.plan_info | {
            "plan_seconds": np.full(batch, plan_seconds),
            "replanned": np.full(batch, offset == 0),
            "prediction_stride": np.full(batch, self.prediction_stride),
        }


class TrueDynamicsMPC(PlanningController):
    """CEM through batched MJX rollouts of the real environment."""

    def __init__(self, env, planner: CEMPlanner, replan_every: int = 1):
        super().__init__(planner, replan_every)
        candidates = planner.config.candidates

        def rollout(state, actions):
            tiled = jax.tree.map(lambda x: jnp.repeat(x, candidates, axis=0), state)

            def step(carry, action):
                following = jax.vmap(env.step)(carry, action)
                return following, (following.obs, following.reward)

            _, (observations, rewards) = jax.lax.scan(
                step, tiled, jnp.swapaxes(actions, 0, 1)
            )
            return jnp.swapaxes(observations, 0, 1), rewards.sum(axis=0)

        self.rollout = jax.jit(rollout)

    def evaluator(self, state, observations, qpos, qvel):
        device = self.planner.device

        def evaluate(actions: torch.Tensor) -> Evaluation:
            batch, candidates, horizon = actions.shape[:3]
            predicted, returns = self.rollout(
                state, jnp.asarray(actions.flatten(0, 1).cpu().numpy())
            )
            return Evaluation(
                to_torch(returns, device).view(batch, candidates),
                {
                    "observations": to_torch(predicted, device).view(
                        batch, candidates, horizon, -1
                    )
                },
            )

        return evaluate


class StateModelMPC(PlanningController):
    """CEM through an observation-space model rolled out from true observations."""

    def __init__(self, dynamics, planner: CEMPlanner, replan_every: int = 1):
        super().__init__(planner, replan_every)
        self.dynamics = dynamics

    def evaluator(self, state, observations, qpos, qvel):
        return state_evaluator(
            self.dynamics, to_torch(observations, self.planner.device)
        )


class LatentMPC(PlanningController):
    """CEM through a latent world model from the encoded rendered frames.

    Every control step renders and encodes the newest frames, so the encoder
    window and the probe history match how the training embeddings were
    computed (``wm.episodes.step_windows``). The model's context holds the
    latents of its last ``history`` model steps, ``step_stride`` control steps
    apart, and the grouped actions between them; plans are made at model steps.
    """

    def __init__(
        self,
        planner: CEMPlanner,
        *,
        encoder,
        renderer: FrameRenderer,
        resolution: tuple[int, int],
        predictor,
        probe,
        objective: str,
        goal: torch.Tensor | None = None,
        goal_cost: str = "sum",
        goal_metric: torch.Tensor | None = None,
        replan_every: int = 1,
        rollout_chunk: int | None = None,
    ):
        stride = predictor.step_stride
        if replan_every % stride:
            raise ValueError("replan_every must be a multiple of the model stride")
        if stride > 1 and planner.config.action_repeat != stride:
            raise ValueError("action_repeat must equal the model stride")
        super().__init__(planner, replan_every)
        self.prediction_stride = stride
        self.encoder = encoder
        self.renderer = renderer
        self.resolution = resolution
        self.predictor = predictor
        self.probe = probe
        self.objective = objective
        self.goal = goal
        self.goal_cost = goal_cost
        self.goal_metric = goal_metric
        self.rollout_chunk = rollout_chunk
        # Control steps of latents, features, and actions the context reaches.
        self.keep = max(predictor.history, probe.history) * stride + 1

    def reset(self) -> None:
        super().reset()
        self.frames = []
        self.latents = []
        self.features = []
        self.actions = []
        self.predicted_latent = None
        self.predicted_step = None

    def encode_newest(self, qpos, qvel):
        self.frames.append(self.renderer.render(qpos, qvel, resolution=self.resolution))
        window = step_windows(len(self.frames), self.encoder.context_frames)[-1]
        clips = np.stack([self.frames[i] for i in window], axis=1)  # (B, L, H, W, 3)
        self.frames = self.frames[-self.encoder.context_frames :]
        return encode_clips(self.encoder, clips)

    def strided(self, values: list, count: int) -> list:
        """The last ``count`` entries ``step_stride`` apart, oldest first."""
        return values[::-1][:: self.predictor.step_stride][:count][::-1]

    @torch.no_grad()
    def observe(self, state, observations, qpos, qvel):
        device = self.planner.device
        output = self.encode_newest(qpos, qvel)
        latent = (
            output.tokens if self.predictor.latent_kind == "tokens" else output.pooled
        ).to(device)
        # The probe reads what the model's ``pooled`` returns: the encoder's own
        # pooled feature, or for token models their (spatial) summary.
        feature = (
            self.predictor.pooled(latent)
            if self.predictor.latent_kind == "tokens"
            else output.pooled.to(device)
        )
        batch = self.planner.batch_size
        info = {"latent_error": np.full(batch, np.nan, dtype=np.float32)}
        if self.predicted_step == self.step:
            # How far the plan's first predicted latent landed from the truth.
            error = self.predicted_latent - self.predictor.normalize(latent)
            info["latent_error"] = (
                error.square().flatten(1).mean(-1).sqrt().cpu().numpy()
            )
        self.latents = [*self.latents, latent][-self.keep :]
        self.features = [*self.features, feature][-self.keep :]

        # The probe stacks earlier true features; the first step repeats itself.
        stacked = self.strided(self.features, self.probe.history)
        past = torch.stack(stacked[:-1], 1) if len(stacked) > 1 else feature[:, None]
        decoded = decode_latents(self.probe, past, feature[:, None])[:, 0]
        info["decoded_observations"] = decoded.cpu().numpy()
        return info

    def evaluator(self, state, observations, qpos, qvel):
        stride = self.predictor.step_stride
        latents = self.strided(self.latents, self.predictor.history)
        context = torch.stack(latents, dim=1)
        groups = []
        for gap in range(len(latents) - 1, 0, -1):
            # Actions applied from the context step ``gap * stride`` steps back.
            first = len(self.actions) - gap * stride
            groups.append(torch.cat(self.actions[first : first + stride], dim=-1))
        context_actions = (
            torch.stack(groups, dim=1)
            if groups
            else context.new_zeros(context.shape[0], 0, stride)
        )
        return latent_evaluator(
            self.predictor,
            self.probe,
            context,
            context_actions,
            objective=self.objective,
            goal=self.goal,
            goal_cost=self.goal_cost,
            goal_metric=self.goal_metric,
            rollout_chunk=self.rollout_chunk,
        )

    def planned(self, plan) -> None:
        self.predicted_latent = plan.extras["first_latent"]
        self.predicted_step = self.step + self.predictor.step_stride

    def act(self, state, observations, qpos, qvel):
        action, info = super().act(state, observations, qpos, qvel)
        action_tensor = to_torch(action, self.planner.device)
        self.actions = [*self.actions, action_tensor][-self.keep :]
        return action, info


def encode_clips(encoder, clips: np.ndarray):
    """Pooled features and tokens of uint8 clips ``(B, L, H, W, 3)``.

    In bfloat16, batches of fewer than about 10 clips run other GPU kernels and
    drift about 1% from the features embed.py cached in batches of 64, so the
    batch is padded to reproduce the training features exactly.
    """
    clips = torch.from_numpy(clips).permute(0, 1, 4, 2, 3)
    padded = clips[torch.arange(max(len(clips), MIN_ENCODER_BATCH)) % len(clips)]
    output = encoder(padded)
    return type(output)(output.pooled[: len(clips)], output.tokens[: len(clips)])


def encode_goal(encoder, renderer, resolution, device, latent_kind: str):
    """Latent of an upright, centered, still pole: the same frame repeated."""
    frame = renderer.render(np.zeros((1, 2)), np.zeros((1, 2)), resolution=resolution)
    clip = np.repeat(frame[None], encoder.context_frames, axis=1)
    output = encode_clips(encoder, clip)
    latent = output.tokens if latent_kind == "tokens" else output.pooled
    return latent[0].to(device)


def run_episodes(env, controller, reset_keys, steps: int) -> dict[str, np.ndarray]:
    """Runs a batch of episodes in lockstep and stacks every per-step record."""
    reset = jax.jit(jax.vmap(env.reset))
    step = jax.jit(jax.vmap(env.step))
    state = reset(reset_keys)
    controller.reset()
    records: dict[str, list] = {"actions": [], "rewards": []}
    observations = [np.asarray(state.obs)]
    qpos = [np.asarray(state.data.qpos)]
    for _ in range(steps):
        current = jax.device_get((state.obs, state.data.qpos, state.data.qvel))
        actions, info = controller.act(state, *map(np.asarray, current))
        info.pop("_plan", None)
        state = step(state, jnp.asarray(actions, dtype=jnp.float32))
        records["actions"].append(np.asarray(actions, dtype=np.float32))
        records["rewards"].append(np.asarray(state.reward))
        observations.append(np.asarray(state.obs))
        qpos.append(np.asarray(state.data.qpos))
        for name, values in info.items():
            records.setdefault(name, []).append(values)
    result = {name: np.stack(values, axis=1) for name, values in records.items()}
    result["observations"] = np.stack(observations, axis=1)
    result["qpos"] = np.stack(qpos, axis=1)
    return result


def pose_rmse(predicted: np.ndarray, actual: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(predicted[..., :3] - actual[..., :3]))))


def summarize(result: dict[str, np.ndarray]) -> dict[str, float]:
    """Episode returns and, for planners, how well they predicted reality."""
    totals = result["rewards"].sum(axis=1)
    half = result["rewards"].shape[1] // 2
    cos_theta = result["observations"][:, 1:, 1]
    summary = {
        "return_mean": float(totals.mean()),
        "return_stderr": float(totals.std(ddof=1) / np.sqrt(len(totals)))
        if len(totals) > 1
        else 0.0,
        "second_half_reward": float(result["rewards"][:, half:].mean()),
        "upright_fraction_second_half": float((cos_theta[:, half:] > 0.95).mean()),
        "mean_abs_cart_position": float(np.abs(result["observations"][..., 0]).mean()),
    }
    if "plan_observations" in result:
        # Each plan's first predicted observation, one model step ahead, against
        # the one actually reached.
        stride = int(result["prediction_stride"][0, 0])
        steps = np.flatnonzero(result["replanned"][0])
        steps = steps[steps + stride < result["observations"].shape[1]]
        predicted = result["plan_observations"][:, steps, 0]
        summary["one_step_pose_rmse"] = pose_rmse(
            predicted, result["observations"][:, steps + stride]
        )
        # Per control step (plans are amortized over replan_every steps).
        summary["plan_seconds"] = float(result["plan_seconds"].mean())
    if "decoded_observations" in result:
        summary["probe_pose_rmse"] = pose_rmse(
            result["decoded_observations"], result["observations"][:, :-1]
        )
    if "latent_error" in result and np.isfinite(result["latent_error"]).any():
        summary["one_step_latent_rmse"] = float(np.nanmean(result["latent_error"]))
    return summary

"""Closed-loop model-predictive control of CartPole with the latent world model.

Every control step renders the real environment, encodes the newest frames with
the frozen encoder, plans with CEM through the latent predictor (Phase 3), and
executes only the first action before replanning. The same batch of episodes,
from the same reset keys, is run for every controller:

- ``ppo``: the deterministic PPO expert on true observations (needs MLflow)
- ``random``: uniform random actions, the floor
- ``mpc-true``: CEM through the true MJX dynamics, the planner's ceiling
- ``mpc-linear``: CEM through the linear residual model on true observations
- ``mpc-latent``: CEM through the latent model, scoring the CartPole reward of
  probe-decoded latents
- ``mpc-latent-goal``: CEM through the latent model, scoring the distance to the
  latent of an upright, centered, still pole

All MPC variants share the CEM settings and plan for the environment's own
dense reward, so they differ only in the model. Each run writes per-controller
trajectories, a summary, and figures under ``--output-dir``.
"""

import argparse
import json
import os
import tempfile
import time
from pathlib import Path

# Headless EGL renders far faster than GLFW; set MUJOCO_GL to override.
os.environ.setdefault("MUJOCO_GL", "egl")

from world_model_playground.wm.runtime import share_gpu_with_jax

# The MJX environment and PyTorch models share the GPU.
share_gpu_with_jax()

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from world_model_playground.cartpole import FrameRenderer, load_cartpole, positive_int
from world_model_playground.wm.data import split_episode_ids
from world_model_playground.wm.embeddings import (
    load_embedded_episodes,
    load_embeddings_manifest,
)
from world_model_playground.wm.encoders import load_encoder
from world_model_playground.wm.episodes import (
    episode_lengths,
    load_manifest,
    step_windows,
)
from world_model_playground.wm.linear import fit_linear_dynamics
from world_model_playground.wm.planning import (
    CEMConfig,
    CEMPlanner,
    Evaluation,
    decode_latents,
    latent_evaluator,
    linear_dynamics,
    state_evaluator,
)
from world_model_playground.wm.probes import load_probe
from world_model_playground.wm.runtime import torch_device
from world_model_playground.wm.world_models import load_world_model

CONTROLLERS = (
    "ppo",
    "random",
    "mpc-true",
    "mpc-linear",
    "mpc-latent",
    "mpc-latent-goal",
)
RESETS = ("balance", "full-circle")
MIN_ENCODER_BATCH = 16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--embeddings-dir",
        type=Path,
        required=True,
        help="embeddings written by embed.py, whose encoder the predictor reads",
    )
    parser.add_argument(
        "--predictor",
        type=Path,
        help="latent world model from train_latent_wm.py or train_ac_wm.py "
        "(default: artifacts/cartpole_world_model/latent_wm/<collection_id>/"
        "<embeddings>/predictor_h4_r8.pt)",
    )
    parser.add_argument(
        "--probe",
        type=Path,
        help="state probe decoding latents (default: the predictor's probe)",
    )
    parser.add_argument(
        "--collection-dir",
        type=Path,
        help="collection the embeddings were made from (default: the path in the "
        "embeddings manifest, else data/cartpole_trajectories/<collection_id>)",
    )
    parser.add_argument(
        "--controllers", nargs="+", choices=CONTROLLERS, default=list(CONTROLLERS)
    )
    parser.add_argument(
        "--resets",
        nargs="+",
        choices=RESETS,
        default=list(RESETS),
        help="initial states: near upright (balance) or any pole angle, which needs "
        "a swing-up (full-circle) (default: both)",
    )
    parser.add_argument("--episodes", type=positive_int, default=10)
    parser.add_argument("--steps", type=positive_int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--horizon", type=positive_int, default=30)
    parser.add_argument("--candidates", type=positive_int, default=256)
    parser.add_argument("--iterations", type=positive_int, default=4)
    parser.add_argument("--elites", type=positive_int, default=32)
    parser.add_argument(
        "--action-repeat",
        type=positive_int,
        default=2,
        help="steps each sampled action knot is held for; must equal the "
        "predictor's step stride when it spans several control steps (default: 2)",
    )
    parser.add_argument(
        "--replan-every",
        type=positive_int,
        default=1,
        help="control steps between plans, for every MPC controller; each plan's "
        "first actions run in between (default: 1)",
    )
    parser.add_argument("--init-std", type=float, default=0.7)
    parser.add_argument(
        "--split-seed",
        type=int,
        default=0,
        help="episode split seed of the probe and predictor; the linear model is "
        "fitted on the same training episodes (default: 0)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="results directory (default: artifacts/cartpole_world_model/mpc/"
        "<collection_id>/<embeddings>/<timestamp>)",
    )
    parser.add_argument(
        "--mlflow-experiment",
        default=os.environ.get("MLFLOW_EXPERIMENT_NAME", "cartpole-wm"),
    )
    parser.add_argument("--device", help="torch device (default: cuda if available)")
    return parser.parse_args()


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
        replan_every: int = 1,
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
        feature = output.pooled.to(device)
        latent = (
            output.tokens if self.predictor.latent_kind == "tokens" else output.pooled
        ).to(device)
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


def plot_results(results: dict, reset: str, path: Path, control_dt: float) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.2), layout="constrained")
    for name, result in results.items():
        time_axis = np.arange(result["rewards"].shape[1]) * control_dt
        mean = result["rewards"].mean(axis=0)
        axes[0].plot(time_axis, mean, label=name)
        cos_theta = result["observations"][:, 1:, 1].mean(axis=0)
        axes[1].plot(time_axis, cos_theta, label=name)
        cart = np.abs(result["observations"][:, 1:, 0]).mean(axis=0)
        axes[2].plot(time_axis, cart, label=name)
    axes[0].set(title=f"Reward per step ({reset})", xlabel="time (s)", ylim=(0, 1.02))
    axes[1].set(title="Mean cos θ (1 = upright)", xlabel="time (s)", ylim=(-1.05, 1.05))
    axes[2].set(title="Mean |cart position|", xlabel="time (s)")
    axes[0].legend(fontsize=8)
    figure.savefig(path, dpi=120)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    device = torch_device(args.device)
    # TF32 matmuls speed latent rollouts up ~1.7x; after 30 imagined steps they
    # drift about 0.001 normalized RMS, against a 0.14 change per step.
    torch.backends.cuda.matmul.allow_tf32 = True
    manifest = load_embeddings_manifest(args.embeddings_dir)
    collection_id = manifest["collection_id"]
    collection_dir = args.collection_dir or Path(manifest["collection_dir"])
    if not (collection_dir / "collection.json").exists():
        collection_dir = Path("data/cartpole_trajectories") / collection_id
    collection = load_manifest(collection_dir)

    predictor_path = args.predictor or (
        Path("artifacts/cartpole_world_model/latent_wm")
        / collection_id
        / args.embeddings_dir.name
        / "predictor_h4_r8.pt"
    )
    predictor, predictor_metadata = load_world_model(predictor_path, device)
    probe_path = args.probe or Path(predictor_metadata["params"]["probe"])
    probe, _ = load_probe(probe_path, device)
    for name in ("encoder", "context_frames", "collection_id"):
        if predictor_metadata["params"][name] != manifest[name]:
            raise ValueError(f"predictor {name} does not match the embeddings")

    # The linear baseline sees exactly the predictor's training episodes.
    ids = np.fromiter(episode_lengths(manifest), dtype=np.int64)
    train_ids = split_episode_ids(ids, seed=args.split_seed)[0]
    if train_ids.tolist() != predictor_metadata["episode_splits"]["train"]:
        raise ValueError("--split-seed does not reproduce the predictor's split")
    train = load_embedded_episodes(
        args.embeddings_dir, train_ids, fields=("observations", "actions")
    )
    linear_model = fit_linear_dynamics(
        np.concatenate([e["observations"][:-1] for e in train.values()]),
        np.concatenate([e["actions"] for e in train.values()]),
        np.concatenate([e["observations"][1:] for e in train.values()]),
        ridge=1e-6,
    )

    cem = CEMConfig(
        horizon=args.horizon,
        candidates=args.candidates,
        iterations=args.iterations,
        elites=args.elites,
        action_repeat=args.action_repeat,
        init_std=args.init_std,
    )
    resolution = (manifest["input_size"], manifest["input_size"])
    needs_encoder = any(c.startswith("mpc-latent") for c in args.controllers)
    stride = predictor.step_stride
    if (
        needs_encoder
        and stride > 1
        and (
            args.action_repeat != stride
            or args.replan_every % stride
            or args.horizon % stride
        )
    ):
        raise ValueError(
            f"the predictor steps {stride} control steps at once: use "
            f"--action-repeat {stride} and multiples of {stride} for --horizon "
            "and --replan-every"
        )
    encoder = (
        load_encoder(
            manifest["encoder"],
            context_frames=manifest["context_frames"],
            device=device,
        )
        if needs_encoder
        else None
    )

    policy = None
    if "ppo" in args.controllers:
        import mlflow

        from world_model_playground.policy import load_policy, select_run_checkpoint

        mlflow.set_tracking_uri(
            os.environ.get("MLFLOW_TRACKING_URI", collection["mlflow_tracking_uri"])
        )
        with tempfile.TemporaryDirectory(prefix="cartpole_checkpoint_") as directory:
            _, _, checkpoint_path = select_run_checkpoint(
                collection["mlflow_run_id"], Path(directory)
            )
            policy = load_policy(load_cartpole(), checkpoint_path, deterministic=True)

    output_dir = args.output_dir or (
        Path("artifacts/cartpole_world_model/mpc")
        / collection_id
        / args.embeddings_dir.name
        / time.strftime("%Y%m%d-%H%M%S")
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    summary: dict[str, dict] = {}
    reset_keys = jax.random.split(jax.random.key(args.seed), args.episodes)

    with FrameRenderer(
        load_cartpole().mj_model, [resolution], collection["camera"]
    ) as renderer:
        goal = (
            encode_goal(encoder, renderer, resolution, device, predictor.latent_kind)
            if encoder
            else None
        )

        def make_controller(name: str, env):
            def planner(offset: int) -> CEMPlanner:
                return CEMPlanner(
                    cem,
                    batch_size=args.episodes,
                    device=device,
                    seed=args.seed + offset,
                    shift=args.replan_every,
                )

            if name == "ppo":
                return PPOController(policy, args.seed)
            if name == "random":
                return RandomController(args.episodes, args.seed)
            if name == "mpc-true":
                return TrueDynamicsMPC(env, planner(1), args.replan_every)
            if name == "mpc-linear":
                return StateModelMPC(
                    linear_dynamics(linear_model, device),
                    planner(2),
                    args.replan_every,
                )
            return LatentMPC(
                planner(3 if name == "mpc-latent" else 4),
                encoder=encoder,
                renderer=renderer,
                resolution=resolution,
                predictor=predictor,
                probe=probe,
                objective="reward" if name == "mpc-latent" else "goal",
                goal=goal,
                replan_every=args.replan_every,
            )

        for reset in args.resets:
            env = load_cartpole(full_circle_reset=reset == "full-circle")
            results = {}
            for name in args.controllers:
                start = time.perf_counter()
                result = run_episodes(
                    env, make_controller(name, env), reset_keys, args.steps
                )
                seconds = time.perf_counter() - start
                results[name] = result
                summary.setdefault(reset, {})[name] = summarize(result) | {
                    "wall_seconds": seconds
                }
                np.savez_compressed(output_dir / f"{reset}_{name}.npz", **result)
                stats = summary[reset][name]
                print(
                    f"[{reset}] {name}: return {stats['return_mean']:.1f} "
                    f"± {stats['return_stderr']:.1f} ({seconds:.0f}s)"
                )
            plot_results(results, reset, output_dir / f"{reset}_rewards.png", env.dt)

    record = {
        "params": {
            "embeddings_dir": str(args.embeddings_dir.resolve()),
            "collection_id": collection_id,
            "encoder": manifest["encoder"],
            "context_frames": manifest["context_frames"],
            "predictor": str(predictor_path),
            "predictor_step_stride": stride,
            "probe": str(probe_path),
            "replan_every": args.replan_every,
            "episodes": args.episodes,
            "steps": args.steps,
            "seed": args.seed,
            "controllers": args.controllers,
            "resets": args.resets,
            **{f"cem_{k}": v for k, v in vars(cem).items()},
        },
        "summary": summary,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    write_markdown(summary, output_dir / "results.md")
    if os.environ.get("MLFLOW_TRACKING_URI"):
        log_to_mlflow(args.mlflow_experiment, record, output_dir)
    print((output_dir / "results.md").read_text())
    print(f"Results: {output_dir}")


def write_markdown(summary: dict, path: Path) -> None:
    columns = (
        ("return_mean", "return"),
        ("second_half_reward", "reward, 2nd half"),
        ("upright_fraction_second_half", "upright, 2nd half"),
        ("one_step_pose_rmse", "1-step pose RMSE"),
        ("probe_pose_rmse", "probe pose RMSE"),
        ("plan_seconds", "plan s/step"),
    )
    lines = []
    for reset, controllers in summary.items():
        lines += [
            f"### {reset}",
            "",
            "| controller | " + " | ".join(label for _, label in columns) + " |",
            "| --- |" + " --- |" * len(columns),
        ]
        for name, stats in controllers.items():
            cells = []
            for key, _ in columns:
                if key not in stats:
                    cells.append("")
                elif key == "return_mean":
                    cells.append(f"{stats[key]:.1f} ± {stats['return_stderr']:.1f}")
                else:
                    cells.append(f"{stats[key]:.3f}")
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def log_to_mlflow(experiment: str, record: dict, output_dir: Path) -> None:
    import mlflow

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(experiment)
    name = f"{record['params']['encoder']}_ctx{record['params']['context_frames']}"
    with mlflow.start_run(run_name=f"{name}-latent-mpc", tags={"stage": "latent_mpc"}):
        mlflow.log_params(
            {k: v for k, v in record["params"].items() if not isinstance(v, list)}
        )
        mlflow.log_metrics(
            {
                f"{reset}_{controller}_{key}": value
                for reset, controllers in record["summary"].items()
                for controller, stats in controllers.items()
                for key, value in stats.items()
            }
        )
        mlflow.log_artifacts(str(output_dir), artifact_path="latent_mpc")


if __name__ == "__main__":
    main()

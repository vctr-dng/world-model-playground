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
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from world_model_playground.cartpole import FrameRenderer, load_cartpole
from world_model_playground.cli import positive_int
from world_model_playground.wm.control import (
    LatentMPC,
    PPOController,
    RandomController,
    StateModelMPC,
    TrueDynamicsMPC,
    encode_goal,
    run_episodes,
    summarize,
)
from world_model_playground.wm.data import split_episode_ids
from world_model_playground.wm.embeddings import (
    load_embedded_episodes,
    load_embeddings_manifest,
    source_resolution,
)
from world_model_playground.wm.encoders import load_encoder
from world_model_playground.wm.episodes import episode_lengths, load_manifest
from world_model_playground.wm.linear import fit_linear_dynamics
from world_model_playground.wm.planning import (
    CEMConfig,
    CEMPlanner,
    fit_readout_metric,
    linear_dynamics,
)
from world_model_playground.wm.probes import load_probe
from world_model_playground.wm.runtime import torch_device
from world_model_playground.wm.tracking import tracked_run
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
    parser.add_argument(
        "--rollout-chunk",
        type=positive_int,
        help="candidate sequences rolled out at once, to bound GPU memory for "
        "token models at many episodes x candidates; same results (default: all)",
    )
    parser.add_argument("--init-std", type=float, default=0.7)
    parser.add_argument(
        "--goal-metric",
        choices=("euclidean", "readout"),
        default="euclidean",
        help="how mpc-latent-goal measures the distance between latents: the "
        "model's own norm (default) or the squared distance after a ridge "
        "read-out of the standardized state fitted on the training embeddings "
        "(pooled latents only)",
    )
    parser.add_argument(
        "--goal-cost",
        choices=("sum", "final"),
        default="sum",
        help="mpc-latent-goal scores the goal distance summed over the horizon "
        "(default) or at its last step only, as LeWM plans",
    )
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
        args.embeddings_dir, train_ids, fields=("features", "observations", "actions")
    )
    goal_metric = None
    if args.goal_metric == "readout":
        if predictor.latent_kind != "pooled":
            raise ValueError("--goal-metric readout needs a pooled-latent model")
        goal_metric = fit_readout_metric(
            predictor.normalize(
                torch.as_tensor(np.concatenate([e["features"] for e in train.values()]))
            ),
            torch.as_tensor(
                np.concatenate([e["observations"] for e in train.values()])
            ),
        ).to(device)
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
    # Render as embed.py read frames: collected frames resized by the encoder,
    # not frames rendered directly at the encoder's input size.
    size = manifest["input_size"]
    resolution = source_resolution(collection, size) or (size, size)
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
                goal_cost=args.goal_cost,
                goal_metric=goal_metric,
                replan_every=args.replan_every,
                rollout_chunk=args.rollout_chunk,
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
            "goal_cost": args.goal_cost,
            "goal_metric": args.goal_metric,
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
    name = f"{record['params']['encoder']}_ctx{record['params']['context_frames']}"
    with tracked_run(experiment, f"{name}-latent-mpc", "latent_mpc") as mlflow:
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

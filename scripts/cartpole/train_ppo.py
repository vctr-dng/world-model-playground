"""Train a Brax PPO agent to solve MJX CartpoleBalance."""

import argparse
import datetime
import os
import shutil
import tempfile
import time
import urllib.parse
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import imageio_ffmpeg
import jax
import jax.numpy as jnp
import mediapy
import mlflow
import numpy as np
from brax.training.agents.ppo import checkpoint
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from mujoco_playground import wrapper
from mujoco_playground.config import dm_control_suite_params

from world_model_playground.cartpole import (
    load_cartpole,
    positive_int,
    run_live_rollout,
)

_OBSERVATION_SCALE = (1.8, 1.0, 1.0, 5.0, 5.0)


def scale_observations(observations, processor_params):
    """Applies fixed per-feature scaling to Cartpole observations."""
    del processor_params
    scale = jnp.asarray(_OBSERVATION_SCALE, dtype=observations.dtype)
    return observations / scale


def make_fixed_scale_ppo_networks(
    observation_size, action_size, preprocess_observations_fn=None
):
    """Creates PPO networks that always use fixed observation scaling."""
    del preprocess_observations_fn
    return ppo_networks.make_ppo_networks(
        observation_size,
        action_size,
        preprocess_observations_fn=scale_observations,
    )


def parse_args() -> argparse.Namespace:
    defaults = dm_control_suite_params.brax_ppo_config("CartpoleBalance")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=positive_int, default=1024)
    parser.add_argument(
        "--num-timesteps", type=positive_int, default=defaults.num_timesteps
    )
    parser.add_argument("--num-evals", type=positive_int, default=defaults.num_evals)
    parser.add_argument("--num-eval-envs", type=positive_int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--checkpoint",
        help="local checkpoint directory or MLflow artifact URI "
        "(e.g. runs:/<run_id>/checkpoint)",
    )
    parser.add_argument(
        "--play-only",
        action="store_true",
        help="skip training and load --checkpoint for evaluation",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="show the deterministic policy after training or loading",
    )
    parser.add_argument("--render-steps", type=positive_int, default=1000)
    parser.add_argument(
        "--record-video",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="record a policy rollout at every training evaluation",
    )
    parser.add_argument("--video-steps", type=positive_int, default=1000)
    parser.add_argument("--solve-threshold", type=float, default=950.0)
    parser.add_argument("--batch-size", type=positive_int)
    parser.add_argument("--num-minibatches", type=positive_int)
    parser.add_argument("--num-updates-per-batch", type=positive_int)
    parser.add_argument("--unroll-length", type=positive_int)
    parser.add_argument("--num-resets-per-eval", type=int)
    parser.add_argument("--max-devices", type=positive_int)
    parser.add_argument(
        "--mlflow-experiment",
        default=os.environ.get("MLFLOW_EXPERIMENT_NAME", "cartpole-ppo"),
    )
    parser.add_argument("--mlflow-run-name")
    return parser.parse_args()


def resolve_checkpoint(path: Path) -> Path:
    """Returns a checkpoint directory, resolving a root to its latest step."""
    path = path.resolve()
    if (path / "ppo_network_config.json").exists():
        return path
    candidates = [
        child for child in path.iterdir() if child.is_dir() and child.name.isdigit()
    ]
    if not candidates:
        raise ValueError(f"No Brax PPO checkpoints found under {path}")
    return max(candidates, key=lambda child: int(child.name))


def fetch_checkpoint(checkpoint: str, download_dir: Path) -> Path:
    """Returns a local checkpoint directory, downloading MLflow artifact URIs."""
    if urllib.parse.urlparse(checkpoint).scheme:
        checkpoint = mlflow.artifacts.download_artifacts(
            artifact_uri=checkpoint, dst_path=str(download_dir)
        )
    return resolve_checkpoint(Path(checkpoint))


def load_policy(env, checkpoint_path: Path):
    """Recreates the fixed PPO network and restores its parameters."""
    network = make_fixed_scale_ppo_networks(env.observation_size, env.action_size)
    make_policy = ppo_networks.make_inference_fn(network)
    return make_policy(checkpoint.load(checkpoint_path), deterministic=True)


def scalar_metrics(metrics) -> dict[str, float]:
    """Converts scalar JAX metrics into values accepted by MLflow."""
    result = {}
    for name, value in metrics.items():
        array = np.asarray(jax.device_get(value))
        if array.size == 1 and np.issubdtype(array.dtype, np.number):
            result[name] = float(array.item())
    return result


def evaluate_policy(env, policy, seed: int, num_envs: int, num_steps: int) -> float:
    """Evaluates a deterministic policy over a batch of environments."""
    rng, reset_rng = jax.random.split(jax.random.key(seed))
    states = jax.vmap(env.reset)(jax.random.split(reset_rng, num_envs))

    def rollout_step(carry, _):
        states, rng, returns = carry
        rng, action_rng = jax.random.split(rng)
        action_keys = jax.random.split(action_rng, num_envs)
        actions = jax.vmap(policy)(states.obs, action_keys)[0]
        states = jax.vmap(env.step)(states, actions)
        return (states, rng, returns + states.reward), None

    initial = (states, rng, jnp.zeros(num_envs))
    _, _, returns = jax.jit(
        lambda carry: jax.lax.scan(rollout_step, carry, xs=None, length=num_steps)[0]
    )(initial)
    return float(jnp.mean(returns))


def record_policy_video(env, policy, path: Path, seed: int, num_steps: int) -> None:
    """Records one deterministic policy rollout to an MP4 file."""
    reset = jax.jit(env.reset)
    step = jax.jit(env.step)
    state = jax.block_until_ready(reset(jax.random.key(seed)))
    rng = jax.random.key(seed + 1)
    trajectory = []

    for _ in range(num_steps):
        trajectory.append(state)
        rng, action_rng = jax.random.split(rng)
        action = policy(state.obs, action_rng)[0]
        state = step(state, action)

    path.parent.mkdir(parents=True, exist_ok=True)
    frames = env.render(trajectory, camera="fixed")
    mediapy.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())
    mediapy.write_video(path, frames, fps=1 / env.dt)


def render_policy(env, policy, seed: int, num_steps: int) -> None:
    """Displays a deterministic policy in the live MuJoCo viewer."""
    reset = jax.jit(env.reset)
    state = jax.block_until_ready(reset(jax.random.key(seed)))

    def policy_step(state, rng, episode_return):
        rng, action_rng = jax.random.split(rng)
        action = policy(state.obs, action_rng)[0]
        state = env.step(state, action)
        return state, rng, episode_return + state.reward

    initial = (state, jax.random.key(seed + 1), jnp.zeros(()))
    compiled_step = jax.jit(policy_step).lower(*initial).compile()
    carry, completed_steps, _ = run_live_rollout(
        env,
        initial,
        lambda values: compiled_step(*values),
        lambda values: values[0],
        num_steps,
    )
    print(f"Rendered {completed_steps} steps with return {float(carry[2]):.3f}")


def main() -> None:
    args = parse_args()
    if args.play_only and args.checkpoint is None:
        raise ValueError("--play-only requires --checkpoint")
    if args.num_resets_per_eval is not None and args.num_resets_per_eval < 0:
        raise ValueError("--num-resets-per-eval must be at least 0")

    env = load_cartpole(full_circle_reset=True)
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    mlflow.set_tracking_uri(tracking_uri)

    # Files only live here until they are uploaded to MLflow.
    scratch_dir = Path(tempfile.mkdtemp(prefix="cartpole_ppo_"))
    mlflow_active = False
    try:
        checkpoint_path = (
            fetch_checkpoint(args.checkpoint, scratch_dir / "restore")
            if args.checkpoint
            else None
        )
        if args.play_only:
            print(f"Loading checkpoint: {checkpoint_path}")
            policy = jax.jit(load_policy(env, checkpoint_path))
        else:
            config = dm_control_suite_params.brax_ppo_config("CartpoleBalance")
            config.normalize_observations = False
            config.num_envs = args.num_envs
            config.num_timesteps = args.num_timesteps
            config.num_evals = args.num_evals
            overrides = {
                "batch_size": args.batch_size,
                "num_minibatches": args.num_minibatches,
                "num_updates_per_batch": args.num_updates_per_batch,
                "unroll_length": args.unroll_length,
                "num_resets_per_eval": args.num_resets_per_eval,
            }
            for name, value in overrides.items():
                if value is not None:
                    config[name] = value
            if config.batch_size * config.num_minibatches % config.num_envs:
                raise ValueError(
                    "batch_size * num_minibatches must be divisible by num_envs"
                )

            timestamp = datetime.datetime.now(tz=datetime.UTC).strftime("%Y%m%d-%H%M%S")
            checkpoints_dir = scratch_dir / "checkpoints"
            videos_dir = scratch_dir / "videos"
            checkpoints_dir.mkdir()
            print(f"Training parameters:\n{config}")

            mlflow.set_experiment(args.mlflow_experiment)
            mlflow.start_run(
                run_name=args.mlflow_run_name or timestamp,
                tags={
                    "environment": "CartpoleBalance",
                    "mjx_implementation": "jax",
                    "jax_backend": jax.default_backend(),
                },
            )
            mlflow_active = True
            mlflow.log_params({f"ppo.{name}": value for name, value in config.items()})
            mlflow.log_params(
                {
                    "seed": args.seed,
                    "num_eval_envs": args.num_eval_envs,
                    "full_circle_reset": True,
                    "record_video": args.record_video,
                    "video_steps": args.video_steps,
                    "max_devices": args.max_devices or "all",
                    "restore_checkpoint": args.checkpoint or "",
                }
            )
            mlflow.log_dict(config.to_dict(), "config/ppo.json")
            print(f"MLflow tracking URI: {tracking_uri}")
            print(f"MLflow run ID: {mlflow.active_run().info.run_id}")

            start = time.monotonic()
            evaluation_policies = {}
            pending_video_steps = set()

            def write_evaluation_video(step: int, policy) -> None:
                video_path = videos_dir / f"eval_{step:012d}.mp4"
                record_policy_video(
                    env,
                    policy,
                    video_path,
                    seed=args.seed + step,
                    num_steps=args.video_steps,
                )
                mlflow.log_artifact(str(video_path), artifact_path="videos")
                video_path.unlink()
                print(f"Evaluation video: videos/{video_path.name}")

            def upload_checkpoint(step: int) -> None:
                # Brax saves the step's checkpoint just before calling progress_fn.
                step_dir = checkpoints_dir / f"{step:012d}"
                if not step_dir.exists():
                    return
                mlflow.log_artifacts(
                    str(step_dir), artifact_path=f"checkpoints/{step_dir.name}"
                )
                # Keep only the latest locally; it is re-logged as the final one.
                for old_dir in checkpoints_dir.iterdir():
                    if old_dir != step_dir:
                        shutil.rmtree(old_dir)
                print(f"Checkpoint: checkpoints/{step_dir.name}")

            def capture_policy(step: int, make_policy, params) -> None:
                if not args.record_video:
                    return
                policy = jax.jit(make_policy(params, deterministic=True))
                if step in pending_video_steps:
                    pending_video_steps.remove(step)
                    write_evaluation_video(step, policy)
                else:
                    evaluation_policies.clear()
                    evaluation_policies[step] = policy

            def progress(step: int, metrics) -> None:
                reward = metrics.get("eval/episode_reward")
                elapsed = time.monotonic() - start
                logged_metrics = scalar_metrics(metrics)
                logged_metrics["elapsed_seconds"] = elapsed
                mlflow.log_metrics(logged_metrics, step=step)
                upload_checkpoint(step)
                if reward is not None and args.record_video:
                    policy = evaluation_policies.pop(step, None)
                    if policy is None:
                        pending_video_steps.add(step)
                    else:
                        write_evaluation_video(step, policy)
                if reward is None:
                    print(f"step={step:,} elapsed={elapsed:.1f}s")
                else:
                    print(
                        f"step={step:,} eval_reward={float(reward):.3f} "
                        f"elapsed={elapsed:.1f}s"
                    )

            training_params = dict(config)
            make_policy, params, final_metrics = ppo.train(
                environment=env,
                **training_params,
                network_factory=make_fixed_scale_ppo_networks,
                wrap_env_fn=wrapper.wrap_for_brax_training,
                seed=args.seed,
                num_eval_envs=args.num_eval_envs,
                max_devices_per_host=args.max_devices,
                save_checkpoint_path=str(checkpoints_dir),
                restore_checkpoint_path=(
                    str(checkpoint_path) if checkpoint_path else None
                ),
                policy_params_fn=capture_policy,
                progress_fn=progress,
            )
            policy = jax.jit(make_policy(params, deterministic=True))
            final_reward = final_metrics.get("eval/episode_reward")
            if final_reward is not None:
                print(f"Final Brax evaluation reward: {float(final_reward):.3f}")

        mean_return = evaluate_policy(
            env,
            policy,
            seed=args.seed + 100,
            num_envs=args.num_eval_envs,
            num_steps=1000,
        )
        solved = mean_return >= args.solve_threshold
        status = "SOLVED" if solved else "NOT SOLVED"
        print(f"Deterministic mean return: {mean_return:.3f} ({status})")

        if mlflow_active:
            mlflow.log_metrics(
                {
                    "deterministic_mean_return": mean_return,
                    "solved": float(solved),
                }
            )
            latest_checkpoint = resolve_checkpoint(checkpoints_dir)
            mlflow.log_artifacts(str(latest_checkpoint), artifact_path="checkpoint")
            print(f"Checkpoint: runs:/{mlflow.active_run().info.run_id}/checkpoint")
    except BaseException:
        if mlflow_active:
            mlflow.end_run(status="FAILED")
        raise
    else:
        if mlflow_active:
            mlflow.end_run(status="FINISHED")
    finally:
        shutil.rmtree(scratch_dir, ignore_errors=True)

    if args.render:
        render_policy(env, policy, args.seed + 200, args.render_steps)


if __name__ == "__main__":
    main()

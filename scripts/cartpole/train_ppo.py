"""Train a Brax PPO agent to solve MJX CartpoleBalance."""

import argparse
import datetime
import os
import time
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
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
    parser.add_argument("--output-dir", type=Path, default=Path("runs/cartpole_ppo"))
    parser.add_argument("--checkpoint", type=Path)
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

    env = load_cartpole()
    checkpoint_path = resolve_checkpoint(args.checkpoint) if args.checkpoint else None

    mlflow_active = False
    try:
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
            run_dir = (args.output_dir / timestamp).resolve()
            checkpoints_dir = run_dir / "checkpoints"
            checkpoints_dir.mkdir(parents=True, exist_ok=True)
            print(f"Run directory: {run_dir}")
            print(f"Training parameters:\n{config}")

            tracking_uri = os.environ.get(
                "MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"
            )
            mlflow.set_tracking_uri(tracking_uri)
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
                    "max_devices": args.max_devices or "all",
                    "restore_checkpoint": str(checkpoint_path or ""),
                    "run_directory": str(run_dir),
                }
            )
            mlflow.log_dict(config.to_dict(), "config/ppo.json")
            print(f"MLflow tracking URI: {tracking_uri}")
            print(f"MLflow run ID: {mlflow.active_run().info.run_id}")

            start = time.monotonic()

            def progress(step: int, metrics) -> None:
                reward = metrics.get("eval/episode_reward")
                elapsed = time.monotonic() - start
                logged_metrics = scalar_metrics(metrics)
                logged_metrics["elapsed_seconds"] = elapsed
                mlflow.log_metrics(logged_metrics, step=step)
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
                progress_fn=progress,
            )
            policy = jax.jit(make_policy(params, deterministic=True))
            final_reward = final_metrics.get("eval/episode_reward")
            if final_reward is not None:
                print(f"Final Brax evaluation reward: {float(final_reward):.3f}")
            print(f"Checkpoints: {checkpoints_dir}")

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
    except BaseException:
        if mlflow_active:
            mlflow.end_run(status="FAILED")
        raise
    else:
        if mlflow_active:
            mlflow.end_run(status="FINISHED")

    if args.render:
        render_policy(env, policy, args.seed + 200, args.render_steps)


if __name__ == "__main__":
    main()

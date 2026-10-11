"""Collect Cartpole episodes with rendered frames for world-model training.

Episodes are rolled out in parallel MJX environments and written as episode
shards (see ``world_model_playground.wm.episodes``), each with its physics
state and frames at every requested resolution, plus a flat
``transitions.npz`` for the state-only models.
"""

import argparse
import json
import os
import tempfile
import uuid
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
# Headless EGL renders far faster than GLFW; set MUJOCO_GL to override.
os.environ.setdefault("MUJOCO_GL", "egl")

import jax
import jax.numpy as jnp
import mlflow
import numpy as np

from world_model_playground.cartpole import (
    FrameRenderer,
    load_cartpole,
    parse_resolution,
    positive_int,
)
from world_model_playground.policy import (
    load_policy,
    select_best_checkpoint,
    select_run_checkpoint,
)
from world_model_playground.wm.data import OBSERVATION_NAMES
from world_model_playground.wm.episodes import (
    FORMAT_VERSION,
    MANIFEST_NAME,
    episode_path,
    episodes_to_transitions,
    frames_key,
    split_batched_episodes,
    write_episode,
)

_ACTION_SOURCES = ("deterministic", "stochastic", "random")
_MAX_EPISODE_LENGTH = 1000  # CartpoleBalance's own time limit


def fraction(value: str) -> float:
    """Parses a fraction in the inclusive range [0, 1]."""
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return parsed


def episode_length(value: str) -> int:
    """Parses an episode horizon capped at the environment's time limit."""
    parsed = positive_int(value)
    if parsed > _MAX_EPISODE_LENGTH:
        raise argparse.ArgumentTypeError(f"must be at most {_MAX_EPISODE_LENGTH}")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    policy_source = parser.add_mutually_exclusive_group()
    policy_source.add_argument(
        "--mlflow-experiment",
        default=os.environ.get("MLFLOW_EXPERIMENT_NAME", "cartpole-ppo"),
        help="experiment from which to select the best PPO run (default: cartpole-ppo)",
    )
    policy_source.add_argument(
        "--mlflow-run-id",
        help="specific MLflow run whose checkpoint should be used",
    )
    parser.add_argument("--num-episodes", type=positive_int, default=500)
    parser.add_argument(
        "--num-envs",
        type=positive_int,
        default=100,
        help="episodes rolled out in parallel (default: 100)",
    )
    parser.add_argument(
        "--episode-length",
        type=episode_length,
        default=200,
        help=f"transitions per episode, at most {_MAX_EPISODE_LENGTH} (default: 200)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--deterministic-fraction",
        type=fraction,
        default=0.4,
        help="share of actions from the deterministic expert policy (default: 0.4)",
    )
    parser.add_argument(
        "--stochastic-fraction",
        type=fraction,
        default=0.3,
        help="share of actions sampled from the expert policy (default: 0.3)",
    )
    parser.add_argument(
        "--resolutions",
        type=parse_resolution,
        nargs="+",
        default=[(64, 64), (256, 256)],
        metavar="HxW",
        help="frame sizes to render, e.g. 64x64 for DreamerV3 and 256x256 for "
        "V-JEPA 2 (default: 64x64 256x256)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/cartpole_trajectories"),
        help="directory containing uniquely named collection directories",
    )
    parser.add_argument(
        "--collection-id",
        help="name for this collection directory (default: generated UUID)",
    )
    parser.add_argument(
        "--full-circle-reset",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="use the PPO training reset distribution (default: enabled)",
    )
    parser.add_argument("--camera", default="fixed")
    return parser.parse_args()


def action_counts(
    num_transitions: int, deterministic_fraction: float, stochastic_fraction: float
) -> np.ndarray:
    """Apportions the requested transition count across the three action sources."""
    random_fraction = 1.0 - deterministic_fraction - stochastic_fraction
    if random_fraction < -1e-12:
        raise ValueError(
            "--deterministic-fraction and --stochastic-fraction must sum to at most 1"
        )
    random_fraction = max(0.0, random_fraction)

    requested = (
        np.array([deterministic_fraction, stochastic_fraction, random_fraction])
        * num_transitions
    )
    counts = np.floor(requested).astype(np.int64)
    remainder = num_transitions - int(counts.sum())
    for index in np.argsort(-(requested - counts), kind="stable")[:remainder]:
        counts[index] += 1
    return counts


def shuffled_action_sources(counts: np.ndarray, seed: int) -> np.ndarray:
    """Returns an exact, reproducibly shuffled action-source schedule."""
    sources = np.repeat(np.arange(len(_ACTION_SOURCES), dtype=np.uint8), counts)
    np.random.default_rng(seed).shuffle(sources)
    return sources


def make_batched_rollout(env, deterministic_policy, stochastic_policy):
    """Builds a jitted rollout of parallel episodes following a source schedule.

    The returned function maps reset keys ``(B,)``, a schedule ``(B, T)`` of
    action-source indices, and a key to the initial and per-step arrays, all
    batch-major: ``initial`` has ``(B, ...)`` and ``steps`` ``(B, T, ...)``.
    """

    def rollout(reset_keys, sources, rng):
        state = jax.vmap(env.reset)(reset_keys)

        def step(state, inputs):
            source, key = inputs
            deterministic_key, stochastic_key, random_key = jax.random.split(key, 3)
            deterministic = deterministic_policy(state.obs, deterministic_key)[0]
            stochastic = stochastic_policy(state.obs, stochastic_key)[0]
            uniform = jax.random.uniform(
                random_key, deterministic.shape, deterministic.dtype, -1.0, 1.0
            )
            source = source[:, None]
            action = jnp.where(
                source == 0, deterministic, jnp.where(source == 1, stochastic, uniform)
            )
            next_state = jax.vmap(env.step)(state, action)
            record = {
                "observations": next_state.obs,
                "actions": action,
                "rewards": next_state.reward,
                "dones": next_state.done,
                "qpos": next_state.data.qpos,
                "qvel": next_state.data.qvel,
            }
            return next_state, record

        step_keys = jax.random.split(rng, sources.shape[1])
        _, steps = jax.lax.scan(step, state, (sources.T, step_keys))
        initial = {
            "observations": state.obs,
            "qpos": state.data.qpos,
            "qvel": state.data.qvel,
        }
        steps = jax.tree.map(lambda values: jnp.swapaxes(values, 0, 1), steps)
        return initial, steps

    return jax.jit(rollout)


def main() -> None:
    args = parse_args()
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        raise ValueError("MLFLOW_TRACKING_URI must be set")
    mlflow.set_tracking_uri(tracking_uri)

    counts = action_counts(
        args.num_episodes * args.episode_length,
        args.deterministic_fraction,
        args.stochastic_fraction,
    )
    schedule = shuffled_action_sources(counts, args.seed).reshape(
        args.num_episodes, args.episode_length
    )
    resolutions = list(dict.fromkeys(args.resolutions))
    collection_id = args.collection_id or uuid.uuid4().hex
    output_dir = args.output_dir / collection_id
    if output_dir.exists():
        raise FileExistsError(f"Collection directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    env = load_cartpole(full_circle_reset=args.full_circle_reset)

    with tempfile.TemporaryDirectory(prefix="cartpole_checkpoint_") as checkpoint_dir:
        if args.mlflow_run_id:
            run_id, score, checkpoint_path = select_run_checkpoint(
                args.mlflow_run_id, Path(checkpoint_dir)
            )
            selection_mode = "specified_run"
        else:
            run_id, score, checkpoint_path = select_best_checkpoint(
                args.mlflow_experiment, Path(checkpoint_dir)
            )
            selection_mode = "best_experiment_run"
        rollout = make_batched_rollout(
            env,
            load_policy(env, checkpoint_path, deterministic=True),
            load_policy(env, checkpoint_path, deterministic=False),
        )

    batch_size = min(args.num_envs, args.num_episodes)
    rng = jax.random.key(args.seed)
    episodes_low_dim = []
    manifest_episodes = []
    with FrameRenderer(env.mj_model, resolutions, args.camera) as renderer:
        for batch_index, first in enumerate(range(0, args.num_episodes, batch_size)):
            ids = np.arange(first, min(first + batch_size, args.num_episodes))
            # Pad the last batch to the compiled batch size and drop the extras.
            batch_sources = schedule[np.resize(ids, batch_size)]
            batch_rng = jax.random.fold_in(rng, batch_index)
            reset_rng, step_rng = jax.random.split(batch_rng)
            initial, steps = jax.device_get(
                rollout(
                    jax.random.split(reset_rng, batch_size),
                    jnp.asarray(batch_sources),
                    step_rng,
                )
            )
            batch_episodes = split_batched_episodes(initial, steps, batch_sources)
            for episode_id, episode in zip(ids, batch_episodes, strict=False):
                episodes_low_dim.append((int(episode_id), episode))
                frames = {
                    frames_key(*resolution): renderer.render(
                        episode["qpos"], episode["qvel"], resolution=resolution
                    )
                    for resolution in resolutions
                }
                length = write_episode(output_dir, int(episode_id), episode | frames)
                manifest_episodes.append(
                    {
                        "id": int(episode_id),
                        "file": str(
                            episode_path(output_dir, int(episode_id)).relative_to(
                                output_dir
                            )
                        ),
                        "length": length,
                        "terminated": bool(episode["terminated"]),
                    }
                )
            print(f"Collected episodes {ids[0]}-{ids[-1]} of {args.num_episodes}")
        resolution_entries = [
            {
                "key": frames_key(height, width),
                "height": height,
                "width": width,
                "fovy": renderer.fovy(height, width),
            }
            for height, width in resolutions
        ]

    transitions = episodes_to_transitions(episodes_low_dim)
    collected_counts = np.bincount(
        transitions["action_source"], minlength=len(_ACTION_SOURCES)
    )
    collection_parameters = {
        "format_version": FORMAT_VERSION,
        "collection_id": collection_id,
        "environment": "CartpoleBalance",
        "mlflow_tracking_uri": tracking_uri,
        "mlflow_run_id": run_id,
        "selection_mode": selection_mode,
        "selection_metric": "deterministic_mean_return",
        "selection_metric_value": score,
        "seed": args.seed,
        "full_circle_reset": args.full_circle_reset,
        "camera": args.camera,
        "control_dt": float(env.dt),
        "observation_names": list(OBSERVATION_NAMES),
        "resolutions": resolution_entries,
        "num_envs": batch_size,
        "num_episodes": args.num_episodes,
        "num_transitions": len(transitions["actions"]),
        "episode_length": args.episode_length,
        "num_terminated": int(transitions["terminated"].sum()),
        "num_truncated": int(transitions["truncated"].sum()),
        "action_fractions": {
            "deterministic": args.deterministic_fraction,
            "stochastic": args.stochastic_fraction,
            "random": round(
                max(
                    0.0,
                    1.0 - args.deterministic_fraction - args.stochastic_fraction,
                ),
                12,
            ),
        },
        "action_counts": dict(
            zip(_ACTION_SOURCES, map(int, collected_counts), strict=True)
        ),
        "episodes": manifest_episodes,
    }
    np.savez_compressed(
        output_dir / "transitions.npz",
        **transitions,
        action_source_names=np.asarray(_ACTION_SOURCES),
        metadata=np.asarray(
            json.dumps(
                {k: v for k, v in collection_parameters.items() if k != "episodes"},
                sort_keys=True,
            )
        ),
    )
    (output_dir / MANIFEST_NAME).write_text(
        json.dumps(collection_parameters, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"MLflow tracking URI: {tracking_uri}")
    if score is None:
        print(f"Selected run: {run_id} (deterministic_mean_return not logged)")
    else:
        print(f"Selected run: {run_id} (deterministic_mean_return={score:.3f})")
    print(f"Action counts: {collection_parameters['action_counts']}")
    print(f"Collection: {output_dir}")
    print(f"Episodes: {output_dir / 'episodes'} ({args.num_episodes} shards)")
    print(f"Frames: {', '.join(entry['key'] for entry in resolution_entries)}")
    print(
        f"Transitions: {output_dir / 'transitions.npz'} {transitions['actions'].shape}"
    )


if __name__ == "__main__":
    main()

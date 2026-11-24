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

from world_model_playground.cartpole import FrameRenderer, load_cartpole
from world_model_playground.cli import fraction, parse_resolution, positive_int
from world_model_playground.policy import (
    load_policy,
    select_best_checkpoint,
    select_run_checkpoint,
)
from world_model_playground.wm.collection import (
    ACTION_SOURCES,
    held_schedule,
    make_batched_rollout,
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

_MAX_EPISODE_LENGTH = 1000  # CartpoleBalance's own time limit


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
        "--source-hold",
        type=positive_int,
        default=1,
        help="steps each scheduled action source is kept, with one uniform random "
        "action held over a random segment; longer holds show the sustained effect "
        "of a force, which planning models must predict (default: 1)",
    )
    parser.add_argument(
        "--perturb-probability",
        type=fraction,
        default=0.0,
        help="per-step chance, while the pole is upright, of starting a "
        "perturbation: one uniform random force held over the scheduled actions, "
        "after which they recover the pole; 0 disables (default: 0)",
    )
    parser.add_argument(
        "--perturb-steps",
        type=positive_int,
        nargs=2,
        default=[10, 30],
        metavar=("MIN", "MAX"),
        help="range of perturbation lengths in steps (default: 10 30)",
    )
    parser.add_argument(
        "--perturb-upright",
        type=float,
        default=0.9,
        help="cos θ above which perturbations may start (default: 0.9)",
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


def main() -> None:
    args = parse_args()
    if args.perturb_steps[0] > args.perturb_steps[1]:
        raise ValueError("--perturb-steps MIN must not exceed MAX")
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        raise ValueError("MLFLOW_TRACKING_URI must be set")
    mlflow.set_tracking_uri(tracking_uri)

    schedule, random_actions = held_schedule(
        args.num_episodes,
        args.episode_length,
        (args.deterministic_fraction, args.stochastic_fraction),
        args.source_hold,
        args.seed,
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
            perturb_probability=args.perturb_probability,
            perturb_steps=tuple(args.perturb_steps),
            perturb_upright=args.perturb_upright,
        )

    batch_size = min(args.num_envs, args.num_episodes)
    rng = jax.random.key(args.seed)
    episodes_low_dim = []
    manifest_episodes = []
    with FrameRenderer(env.mj_model, resolutions, args.camera) as renderer:
        for batch_index, first in enumerate(range(0, args.num_episodes, batch_size)):
            ids = np.arange(first, min(first + batch_size, args.num_episodes))
            # Pad the last batch to the compiled batch size and drop the extras.
            batch_rows = np.resize(ids, batch_size)
            batch_sources = schedule[batch_rows]
            batch_rng = jax.random.fold_in(rng, batch_index)
            reset_rng, step_rng = jax.random.split(batch_rng)
            initial, steps = jax.device_get(
                rollout(
                    jax.random.split(reset_rng, batch_size),
                    jnp.asarray(batch_sources),
                    jnp.asarray(random_actions[batch_rows]),
                    step_rng,
                )
            )
            batch_episodes = split_batched_episodes(
                initial, steps, steps["sources"].astype(np.uint8)
            )
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
        transitions["action_source"], minlength=len(ACTION_SOURCES)
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
        "source_hold": args.source_hold,
        "perturbation": {
            "probability": args.perturb_probability,
            "steps": list(args.perturb_steps),
            "upright_cos": args.perturb_upright,
        },
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
            zip(ACTION_SOURCES, map(int, collected_counts), strict=True)
        ),
        "episodes": manifest_episodes,
    }
    np.savez_compressed(
        output_dir / "transitions.npz",
        **transitions,
        action_source_names=np.asarray(ACTION_SOURCES),
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

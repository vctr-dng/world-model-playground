"""Collect Cartpole transitions and aligned rendered images for world-model training."""

import argparse
import json
import math
import os
import tempfile
import urllib.parse
import uuid
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import mlflow
import numpy as np
from brax.training.agents.ppo import checkpoint
from brax.training.agents.ppo import networks as ppo_networks

from world_model_playground.cartpole import load_cartpole, positive_int

_OBSERVATION_SCALE = (1.8, 1.0, 1.0, 5.0, 5.0)
_ACTION_SOURCES = ("deterministic", "stochastic", "random")


def fraction(value: str) -> float:
    """Parses a fraction in the inclusive range [0, 1]."""
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return parsed


def episode_length(value: str) -> int:
    """Parses an episode horizon capped at 200 transitions."""
    parsed = positive_int(value)
    if parsed > 200:
        raise argparse.ArgumentTypeError("must be at most 200")
    return parsed


def scale_observations(observations, processor_params):
    """Applies the fixed observation scaling used by the trained PPO policy."""
    del processor_params
    scale = jnp.asarray(_OBSERVATION_SCALE, dtype=observations.dtype)
    return observations / scale


def make_fixed_scale_ppo_networks(
    observation_size, action_size, preprocess_observations_fn=None
):
    """Builds the PPO network architecture used by train_ppo.py."""
    del preprocess_observations_fn
    return ppo_networks.make_ppo_networks(
        observation_size,
        action_size,
        preprocess_observations_fn=scale_observations,
    )


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


def fetch_checkpoint(checkpoint_uri: str, download_dir: Path) -> Path:
    """Downloads an MLflow artifact URI and returns its checkpoint directory."""
    checkpoint_path = checkpoint_uri
    if urllib.parse.urlparse(checkpoint_uri).scheme:
        checkpoint_path = mlflow.artifacts.download_artifacts(
            artifact_uri=checkpoint_uri, dst_path=str(download_dir)
        )
    return resolve_checkpoint(Path(checkpoint_path))


def load_policy(env, checkpoint_path: Path, *, deterministic: bool):
    """Restores a PPO policy with the selected action sampling mode."""
    network = make_fixed_scale_ppo_networks(env.observation_size, env.action_size)
    make_policy = ppo_networks.make_inference_fn(network)
    return make_policy(checkpoint.load(checkpoint_path), deterministic=deterministic)


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
    parser.add_argument("--num-transitions", type=positive_int, default=10_000)
    parser.add_argument(
        "--episode-length",
        type=episode_length,
        default=200,
        help="maximum transitions per episode, capped at 200 (default: 200)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--deterministic-fraction",
        type=fraction,
        default=0.7,
        help="share of actions from the deterministic expert policy (default: 0.7)",
    )
    parser.add_argument(
        "--stochastic-fraction",
        type=fraction,
        default=0.2,
        help="share of actions sampled from the expert policy (default: 0.2)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/cartpole_trajectories"),
        help="directory containing uniquely named collection directories (default: data)",
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
    parser.add_argument("--render-batch-size", type=positive_int, default=128)
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


def select_best_checkpoint(
    experiment_name: str, download_dir: Path
) -> tuple[str, float, Path]:
    """Downloads the highest-scoring completed PPO checkpoint from MLflow."""
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"MLflow experiment not found: {experiment_name}")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=[
            "metrics.deterministic_mean_return DESC",
            "attributes.start_time DESC",
        ],
        max_results=1_000,
    )
    for run in runs:
        score = run.data.metrics.get("deterministic_mean_return")
        if score is None or not math.isfinite(score):
            continue
        try:
            checkpoint_path = fetch_checkpoint(
                f"runs:/{run.info.run_id}/checkpoint", download_dir / run.info.run_id
            )
        except Exception as error:  # noqa: BLE001
            # A corrupt artifact should not prevent trying the next-ranked policy.
            print(f"Skipping run {run.info.run_id}: checkpoint unavailable ({error})")
            continue
        return run.info.run_id, score, checkpoint_path

    raise ValueError(
        "No finished run with deterministic_mean_return and a checkpoint was found"
    )


def select_run_checkpoint(
    run_id: str, download_dir: Path
) -> tuple[str, float | None, Path]:
    """Downloads the checkpoint from a specific MLflow run."""
    run = mlflow.tracking.MlflowClient().get_run(run_id)
    checkpoint_path = fetch_checkpoint(
        f"runs:/{run.info.run_id}/checkpoint", download_dir / run.info.run_id
    )
    return (
        run.info.run_id,
        run.data.metrics.get("deterministic_mean_return"),
        checkpoint_path,
    )


def shuffled_action_sources(counts: np.ndarray, seed: int) -> np.ndarray:
    """Returns an exact, reproducibly shuffled action-source schedule."""
    sources = np.repeat(np.arange(len(_ACTION_SOURCES), dtype=np.uint8), counts)
    np.random.default_rng(seed).shuffle(sources)
    return sources


def render_images(
    env, states, camera: str, batch_size: int, directory: Path, name: str
):
    """Renders states in batches into a disk-backed array to limit host memory use."""
    images = None
    for start in range(0, len(states), batch_size):
        stop = min(start + batch_size, len(states))
        frames = np.asarray(env.render(states[start:stop], camera=camera))
        if images is None:
            images = np.lib.format.open_memmap(
                directory / f"{name}.npy",
                mode="w+",
                dtype=frames.dtype,
                shape=(len(states), *frames.shape[1:]),
            )
        images[start:stop] = frames
    if images is None:
        raise ValueError("Cannot render an empty transition dataset")
    images.flush()
    return images


def main() -> None:
    args = parse_args()
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        raise ValueError("MLFLOW_TRACKING_URI must be set")
    mlflow.set_tracking_uri(tracking_uri)

    counts = action_counts(
        args.num_transitions, args.deterministic_fraction, args.stochastic_fraction
    )
    action_sources = shuffled_action_sources(counts, args.seed)
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
        deterministic_policy = jax.jit(
            load_policy(env, checkpoint_path, deterministic=True)
        )
        stochastic_policy = jax.jit(
            load_policy(env, checkpoint_path, deterministic=False)
        )

        reset = jax.jit(env.reset)
        step = jax.jit(env.step)
        rng = jax.random.key(args.seed)
        rng, reset_rng = jax.random.split(rng)
        state = jax.block_until_ready(reset(reset_rng))

        observations = []
        actions = []
        next_observations = []
        rewards = []
        dones = []
        terminated = []
        truncated = []
        episode_ids = []
        timesteps = []
        states = []
        next_states = []
        episode_id = 0
        timestep = 0

        for transition_index, source in enumerate(action_sources):
            rng, action_rng = jax.random.split(rng)
            if source == 0:
                action = deterministic_policy(state.obs, action_rng)[0]
            elif source == 1:
                action = stochastic_policy(state.obs, action_rng)[0]
            else:
                action = jax.random.uniform(
                    action_rng,
                    (env.action_size,),
                    minval=-1.0,
                    maxval=1.0,
                    dtype=state.obs.dtype,
                )

            next_state = step(state, action)
            observation, action_value, next_observation, reward, done = jax.device_get(
                (
                    state.obs,
                    action,
                    next_state.obs,
                    next_state.reward,
                    next_state.done,
                )
            )
            observations.append(np.asarray(observation, dtype=np.float32))
            actions.append(np.asarray(action_value, dtype=np.float32))
            next_observations.append(np.asarray(next_observation, dtype=np.float32))
            rewards.append(np.float32(reward))
            terminated_transition = bool(done)
            truncated_transition = timestep + 1 >= args.episode_length
            episode_done = terminated_transition or truncated_transition
            dones.append(episode_done)
            terminated.append(terminated_transition)
            truncated.append(truncated_transition)
            episode_ids.append(episode_id)
            timesteps.append(timestep)
            states.append(state)
            next_states.append(next_state)

            if episode_done and transition_index + 1 < len(action_sources):
                rng, reset_rng = jax.random.split(rng)
                state = jax.block_until_ready(reset(reset_rng))
                episode_id += 1
                timestep = 0
            else:
                state = next_state
                timestep += 1

    observations_array = np.stack(observations)
    actions_array = np.stack(actions)
    next_observations_array = np.stack(next_observations)
    rewards_array = np.asarray(rewards, dtype=np.float32)
    dones_array = np.asarray(dones, dtype=np.bool_)
    terminated_array = np.asarray(terminated, dtype=np.bool_)
    truncated_array = np.asarray(truncated, dtype=np.bool_)
    episode_ids_array = np.asarray(episode_ids, dtype=np.int32)
    timesteps_array = np.asarray(timesteps, dtype=np.int32)
    if not (
        len(action_sources)
        == len(observations_array)
        == len(actions_array)
        == len(next_observations_array)
        == len(states)
        == len(next_states)
    ):
        raise RuntimeError("Transition fields are not aligned")
    if not all(
        np.isfinite(values).all()
        for values in (observations_array, actions_array, next_observations_array)
    ):
        raise RuntimeError("Collected non-finite observation or action values")
    if not np.array_equal(
        dones_array, np.logical_or(terminated_array, truncated_array)
    ):
        raise RuntimeError("Episode boundary fields are inconsistent")
    if np.any(np.bincount(episode_ids_array) > args.episode_length):
        raise RuntimeError("Collected an episode longer than --episode-length")

    collection_parameters = {
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
        "render_batch_size": args.render_batch_size,
        "num_transitions": args.num_transitions,
        "episode_length": args.episode_length,
        "num_episodes": int(episode_ids_array[-1]) + 1,
        "num_terminated": int(terminated_array.sum()),
        "num_truncated": int(truncated_array.sum()),
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
        "action_counts": dict(zip(_ACTION_SOURCES, map(int, counts), strict=True)),
    }
    metadata = json.dumps(collection_parameters, sort_keys=True)
    shared_arrays = {
        "observations": observations_array,
        "actions": actions_array,
        "next_observations": next_observations_array,
        "rewards": rewards_array,
        "dones": dones_array,
        "terminated": terminated_array,
        "truncated": truncated_array,
        "episode_ids": episode_ids_array,
        "timesteps": timesteps_array,
        "action_source": action_sources,
        "action_source_names": np.asarray(_ACTION_SOURCES),
        "metadata": np.asarray(metadata),
    }

    with tempfile.TemporaryDirectory(
        prefix="cartpole_render_", dir=output_dir
    ) as render_dir:
        images = render_images(
            env,
            states,
            args.camera,
            args.render_batch_size,
            Path(render_dir),
            "images",
        )
        next_images = render_images(
            env,
            next_states,
            args.camera,
            args.render_batch_size,
            Path(render_dir),
            "next_images",
        )
        if len(images) != len(observations_array) or len(next_images) != len(
            next_observations_array
        ):
            raise RuntimeError("Rendered images are not aligned with transitions")
        np.savez_compressed(
            output_dir / "image_transitions.npz",
            **shared_arrays,
            images=images,
            next_images=next_images,
        )

    np.savez_compressed(output_dir / "transitions.npz", **shared_arrays)
    (output_dir / "collection.json").write_text(
        json.dumps(collection_parameters, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"MLflow tracking URI: {tracking_uri}")
    if score is None:
        print(f"Selected run: {run_id} (deterministic_mean_return not logged)")
    else:
        print(f"Selected run: {run_id} (deterministic_mean_return={score:.3f})")
    print(f"Action counts: {dict(zip(_ACTION_SOURCES, map(int, counts), strict=True))}")
    print(f"Collection: {output_dir}")
    print(f"Parameters: {output_dir / 'collection.json'}")
    print(f"Transitions: {output_dir / 'transitions.npz'} {observations_array.shape}")
    print(f"Image transitions: {output_dir / 'image_transitions.npz'} {images.shape}")


if __name__ == "__main__":
    main()

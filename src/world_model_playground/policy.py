"""Brax PPO policy construction and MLflow checkpoint loading for CartpoleBalance."""

import math
import urllib.parse
from pathlib import Path

import jax.numpy as jnp
import mlflow
from brax.training.agents.ppo import checkpoint
from brax.training.agents.ppo import networks as ppo_networks

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
    """Returns a local checkpoint directory, downloading MLflow artifact URIs."""
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

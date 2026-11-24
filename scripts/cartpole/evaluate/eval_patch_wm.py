"""Open-loop rollout error of a patch-token world model, against baselines.

Rolls the model out from true context on validation and test episodes with the
logged actions, decodes the predicted token grids with the patch state probe,
and compares the decoded observations with the linear state model, the probe on
true tokens (the decoder's error floor), and persistence. A model step spans
``step_stride`` control steps, so horizons are multiples of it. Optionally also
scores a pooled-feature latent predictor (``--pooled-predictor``) on the same
episodes and horizons.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from world_model_playground.cli import positive_int
from world_model_playground.wm.data import split_episode_ids
from world_model_playground.wm.embeddings import (
    load_embedded_episodes,
    load_embeddings_manifest,
)
from world_model_playground.wm.episodes import episode_lengths
from world_model_playground.wm.eval import (
    OBSERVATION_GROUPS,
    episode_rollout_rmse,
    group_rmse,
    observation_rollout,
    persistence_rollout,
)
from world_model_playground.wm.linear import fit_linear_dynamics
from world_model_playground.wm.patch_predictor import (
    load_patch_predictor,
    patch_rollouts,
    token_episodes,
)
from world_model_playground.wm.predictors import (
    decode_steps,
    latent_rollout,
    load_predictor,
)
from world_model_playground.wm.probes import check_probe, load_probe
from world_model_playground.wm.runtime import torch_device
from world_model_playground.wm.token_cache import open_token_store

SPLITS = ("train", "validation", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings-dir", type=Path, required=True)
    parser.add_argument("--predictor", type=Path, required=True)
    parser.add_argument(
        "--probe", type=Path, help="default: the one recorded with the predictor"
    )
    parser.add_argument(
        "--pooled-predictor",
        type=Path,
        help="a pooled-feature predictor of the same embeddings to compare "
        "(needs --pooled-probe)",
    )
    parser.add_argument("--pooled-probe", type=Path)
    parser.add_argument(
        "--horizons",
        type=positive_int,
        nargs="+",
        default=[5, 10, 25, 50, 100],
        help="control steps; multiples of the model stride are exact",
    )
    parser.add_argument("--start-every", type=positive_int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, help="write the metrics here as JSON")
    parser.add_argument("--device", help="torch device (default: cuda if available)")
    return parser.parse_args()


def by_episode_id(rollout, episodes):
    """Runs a rollout on another episode set's copy of the episode being scored."""

    def run(episode, start, num_steps):
        return rollout(episodes[episode["id"]], start, num_steps)

    return run


def main() -> None:
    args = parse_args()
    device = torch_device(args.device)
    manifest = load_embeddings_manifest(args.embeddings_dir)
    store = open_token_store(args.embeddings_dir)
    model, metadata = load_patch_predictor(args.predictor, device)
    probe, probe_metadata = load_probe(
        args.probe or Path(metadata["params"]["probe"]), device
    )
    ids = np.fromiter(episode_lengths(manifest), dtype=np.int64)
    splits = dict(zip(SPLITS, split_episode_ids(ids, seed=args.seed), strict=True))
    if splits["train"].tolist() != metadata["episode_splits"]["train"]:
        raise ValueError("--seed does not reproduce the predictor's episode split")
    check_probe(probe_metadata, manifest, splits)
    horizons = tuple(args.horizons)

    train = load_embedded_episodes(
        args.embeddings_dir, splits["train"], fields=("observations", "actions")
    )
    linear = fit_linear_dynamics(
        np.concatenate([e["observations"][:-1] for e in train.values()]),
        np.concatenate([e["actions"] for e in train.values()]),
        np.concatenate([e["observations"][1:] for e in train.values()]),
        ridge=1e-6,
    )

    pooled = None
    if args.pooled_predictor:
        pooled_model, pooled_metadata = load_predictor(args.pooled_predictor, device)
        pooled_probe, _ = load_probe(args.pooled_probe, device)
        pooled = (pooled_model, pooled_probe)
        if pooled_metadata["episode_splits"]["train"] != splits["train"].tolist():
            raise ValueError("the pooled predictor used another episode split")

    results = {}
    for split in ("validation", "test"):
        episodes = token_episodes(store, splits[split], model.config.summary_side)
        for episode_id, episode in episodes.items():
            episode["id"] = episode_id
        rollouts = {
            "patch_model": patch_rollouts(model, probe),
            "linear_model": observation_rollout(linear.predict_next),
            "probe_on_true_tokens": lambda episode, start, num_steps: decode_steps(
                probe, episode["features"], slice(start + 1, start + 1 + num_steps)
            ),
            "persistence": persistence_rollout(),
        }
        if pooled:
            rollouts["pooled_model"] = by_episode_id(
                latent_rollout(*pooled),
                load_embedded_episodes(args.embeddings_dir, splits[split]),
            )
        results[split] = {
            name: episode_rollout_rmse(
                rollout, episodes, horizons=horizons, start_every=args.start_every
            )
            for name, rollout in rollouts.items()
        }
        del episodes

    test = results["test"]
    for group in OBSERVATION_GROUPS:
        print(f"\nTest decoded {group} rollout RMSE")
        print(" | ".join(["horizon", *test]))
        for horizon in map(str, horizons):
            if horizon in test["patch_model"]:
                cells = [f"{group_rmse(test[m][horizon], group):.4f}" for m in test]
                print(" | ".join([horizon, *cells]))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()

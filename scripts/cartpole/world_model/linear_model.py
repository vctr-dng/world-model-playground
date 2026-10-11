"""Train a linear residual world model from CartPole observation transitions."""

import argparse
import json
from pathlib import Path

from world_model_playground.wm.data import (
    OBSERVATION_NAMES,
    load_transitions,
    partition_mask,
    split_episode_ids,
    validate_transitions,
)
from world_model_playground.wm.eval import evaluate_partition
from world_model_playground.wm.linear import fit_linear_dynamics, save_linear_model


def positive_float(value: str) -> float:
    """Parse a strictly positive floating-point value."""
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--collection-dir",
        type=Path,
        required=True,
        help="directory containing transitions.npz",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/cartpole_world_model/linear_model.npz"),
        help="output model artifact (default: artifacts/cartpole_world_model/linear_model.npz)",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="episode split seed (default: 0)"
    )
    parser.add_argument(
        "--ridge",
        type=positive_float,
        default=1e-6,
        help="ridge regularization strength (default: 1e-6)",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="replace an existing output artifact"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    transitions = load_transitions(args.collection_dir)
    validate_transitions(transitions)
    train_ids, validation_ids, test_ids = split_episode_ids(
        transitions["episode_ids"], seed=args.seed
    )
    train_mask = partition_mask(transitions["episode_ids"], train_ids)
    model = fit_linear_dynamics(
        transitions["observations"][train_mask],
        transitions["actions"][train_mask],
        transitions["next_observations"][train_mask],
        ridge=args.ridge,
    )
    metrics = {
        "validation": evaluate_partition(
            model.predict_next, transitions, validation_ids
        ),
        "test": evaluate_partition(model.predict_next, transitions, test_ids),
    }
    collection_metadata = json.loads(str(transitions.get("metadata", "{}")))
    artifact_metadata = {
        "architecture": "linear_residual_ridge",
        "collection_id": collection_metadata.get("collection_id"),
        "collection_dir": str(args.collection_dir.resolve()),
        "feature_order": [*OBSERVATION_NAMES, "action"],
        "target_order": [f"{name}_delta" for name in OBSERVATION_NAMES],
        "ridge": args.ridge,
        "seed": args.seed,
        "episode_splits": {
            "train": train_ids.tolist(),
            "validation": validation_ids.tolist(),
            "test": test_ids.tolist(),
        },
        "metrics": metrics,
    }
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {args.output}; pass --overwrite to replace it"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_linear_model(args.output, model, artifact_metadata)
    print(f"Saved linear world model: {args.output}")
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

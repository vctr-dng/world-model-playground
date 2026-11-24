"""Train an action-conditioned world model on cached frozen-encoder patch tokens.

A block-causal transformer predicts the next token grid from the grids and
actions of the last ``--history`` model steps (see ``wm.patch_predictor``), with
the encoder frozen. Model steps are ``--step-stride`` control steps apart and
tokens are average-pooled to ``--pool-side`` x ``--pool-side`` cells. Training
adds an open-loop loss over ``--rollout-steps`` predictions fed back as inputs
to the teacher-forced one-step loss.

Tokens come from ``embed.py --store-tokens`` followed by ``consolidate_tokens.py``.
The episode split matches the probes and the pooled latent model (same
``--seed``). Reports epoch time and peak GPU memory next to the losses.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from world_model_playground.cartpole import positive_int
from world_model_playground.wm.data import split_episode_ids
from world_model_playground.wm.embeddings import load_embeddings_manifest
from world_model_playground.wm.episodes import episode_lengths
from world_model_playground.wm.patch_predictor import (
    PatchPredictorConfig,
    TokenWindows,
    evaluate_patch_losses,
    fit_patch_predictor,
    native_pool_side,
    save_patch_predictor,
)
from world_model_playground.wm.probes import check_probe, load_probe
from world_model_playground.wm.runtime import torch_device
from world_model_playground.wm.token_cache import open_token_store

SPLITS = ("train", "validation", "test")
# Pooled tokens of a split stay on the GPU below this size, else in host memory.
GPU_STORAGE_BUDGET_GB = 2.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--embeddings-dir",
        type=Path,
        required=True,
        help="directory written by embed.py --store-tokens and consolidate_tokens.py",
    )
    parser.add_argument("--seed", type=int, default=0, help="episode split seed")
    parser.add_argument(
        "--pool-side",
        type=positive_int,
        default=8,
        help="side of the pooled token grid the model reads (default: 8)",
    )
    parser.add_argument(
        "--summary-side",
        type=positive_int,
        default=4,
        help="side of the grid probes see after pooling (default: 4)",
    )
    parser.add_argument(
        "--step-stride",
        type=positive_int,
        default=5,
        help="control steps per model step (default: 5)",
    )
    parser.add_argument("--history", type=positive_int, default=3)
    parser.add_argument(
        "--rollout-steps",
        type=positive_int,
        default=4,
        help="open-loop model steps in the training loss (default: 4)",
    )
    parser.add_argument("--rollout-weight", type=float, default=1.0)
    parser.add_argument("--width", type=positive_int, default=384)
    parser.add_argument("--depth", type=positive_int, default=6)
    parser.add_argument("--heads", type=positive_int, default=6)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--epochs", type=positive_int, default=30)
    parser.add_argument(
        "--steps-per-epoch",
        type=positive_int,
        help="optimizer steps per epoch (default: one pass over the windows)",
    )
    parser.add_argument("--batch-size", type=positive_int, default=32)
    parser.add_argument(
        "--accumulation",
        type=positive_int,
        default=1,
        help="micro-batches per optimizer step, for memory (default: 1)",
    )
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument(
        "--storage",
        choices=("auto", "gpu", "cpu"),
        default="auto",
        help="where pooled tokens live while training (default: auto)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/cartpole_world_model/patch_wm"),
        help="models go to <root>/<collection_id>/<embeddings>/<name>.pt",
    )
    parser.add_argument("--name", help="artifact name (default: from the settings)")
    parser.add_argument(
        "--probe",
        type=Path,
        help="patch state probe decoding the model's spatial summary, fitted by "
        "train_patch_probe.py with the same --seed, --summary-side and a "
        "--history-stride of --step-stride; recorded for planning (default: "
        "<probe-root>/<collection_id>/<embeddings>/patch_mlp_h2_k<stride>_s<side>.pt)",
    )
    parser.add_argument(
        "--probe-root",
        type=Path,
        default=Path("artifacts/cartpole_world_model/probes"),
    )
    parser.add_argument(
        "--mlflow-experiment",
        default=os.environ.get("MLFLOW_EXPERIMENT_NAME", "cartpole-wm"),
    )
    parser.add_argument("--device", help="torch device (default: cuda if available)")
    return parser.parse_args()


def storage_device(choice: str, device: torch.device, gigabytes: float):
    if choice == "gpu" or (
        choice == "auto" and device.type == "cuda" and gigabytes < GPU_STORAGE_BUDGET_GB
    ):
        return device
    return torch.device("cpu")


def log_to_mlflow(experiment: str, run_name: str, record: dict, path: Path) -> None:
    import mlflow

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name, tags={"stage": "patch_dynamics"}):
        mlflow.log_params(record["params"])
        for epoch, losses in enumerate(record["training"]["losses"], start=1):
            mlflow.log_metrics(losses, step=epoch)
        mlflow.log_metrics(
            {f"test_{k}": v for k, v in record["test_losses"].items()}
            | {"model_parameters": record["model_parameters"]}
        )
        mlflow.log_dict(record, "patch_wm.json")
        mlflow.log_artifact(str(path), artifact_path="patch_wm")


def main() -> None:
    args = parse_args()
    device = torch_device(args.device)
    manifest = load_embeddings_manifest(args.embeddings_dir)
    store = open_token_store(args.embeddings_dir)
    native = native_pool_side(store)
    if args.pool_side > native or native % args.pool_side:
        raise ValueError(f"cannot pool a {native}x{native} grid to {args.pool_side}")
    ids = np.fromiter(episode_lengths(manifest), dtype=np.int64)
    splits = dict(zip(SPLITS, split_episode_ids(ids, seed=args.seed), strict=True))

    probe_path = args.probe or (
        args.probe_root
        / manifest["collection_id"]
        / args.embeddings_dir.name
        / f"patch_mlp_h2_k{args.step_stride}_s{args.summary_side}.pt"
    )
    if probe_path.exists():
        probe, probe_metadata = load_probe(probe_path, device)
        check_probe(probe_metadata, manifest, splits)
        if probe.stride != args.step_stride:
            raise ValueError("the probe must stack steps --step-stride apart")
        if probe_metadata["params"].get("summary_side") != args.summary_side:
            raise ValueError("the probe was fitted on another --summary-side")
    else:
        print(f"Warning: probe {probe_path} not found; fit it before planning")

    length = args.history + args.rollout_steps
    windows = {}
    for split, episode_ids in splits.items():
        steps = int(sum(store.lengths[np.isin(store.episode_ids, episode_ids)]))
        gigabytes = steps * args.pool_side**2 * store.token_dim * 2 / 1e9
        windows[split] = TokenWindows(
            store,
            episode_ids,
            length=length,
            stride=args.step_stride,
            pool_side=args.pool_side,
            device=device,
            storage_device=storage_device(args.storage, device, gigabytes),
        )
        print(
            f"{split}: {len(episode_ids)} episodes, {len(windows[split])} windows, "
            f"{gigabytes:.2f} GB of pooled tokens on {windows[split].storage_device}"
        )

    config = PatchPredictorConfig(
        token_dim=store.token_dim,
        pool_side=args.pool_side,
        step_stride=args.step_stride,
        history=args.history,
        width=args.width,
        depth=args.depth,
        heads=args.heads,
        dropout=args.dropout,
        summary_side=args.summary_side,
    )
    model, best_epoch, losses = fit_patch_predictor(
        windows["train"],
        windows["validation"],
        config=config,
        rollout_steps=args.rollout_steps,
        rollout_weight=args.rollout_weight,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        batch_size=args.batch_size,
        accumulation=args.accumulation,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
    )
    test_one_step, test_open_loop = evaluate_patch_losses(
        model, windows["test"], args.rollout_steps, args.batch_size
    )
    parameters = sum(p.numel() for p in model.parameters())
    print(
        f"{parameters / 1e6:.1f}M parameters; best epoch {best_epoch}/{args.epochs}; "
        f"test one-step {test_one_step:.4f}, open-loop {test_open_loop:.4f}"
    )

    name = args.name or (
        f"patch_p{args.pool_side}_k{args.step_stride}_h{args.history}"
        f"_r{args.rollout_steps}"
    )
    record = {
        "params": {
            "encoder": manifest["encoder"],
            "checkpoint": manifest["checkpoint"],
            "context_frames": manifest["context_frames"],
            "collection_id": manifest["collection_id"],
            "probe": str(probe_path),
            "seed": args.seed,
            "rollout_steps": args.rollout_steps,
            "rollout_weight": args.rollout_weight,
            "epochs": args.epochs,
            "steps_per_epoch": args.steps_per_epoch,
            "batch_size": args.batch_size,
            "accumulation": args.accumulation,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            **{f"predictor_{k}": v for k, v in vars(config).items()},
        },
        "embeddings_dir": str(args.embeddings_dir.resolve()),
        "episode_splits": {k: v.tolist() for k, v in splits.items()},
        "model_parameters": parameters,
        "training": {"best_epoch": best_epoch, "losses": losses},
        "test_losses": {"one_step": test_one_step, "open_loop": test_open_loop},
    }
    path = (
        args.output_root
        / manifest["collection_id"]
        / args.embeddings_dir.name
        / f"{name}.pt"
    )
    save_patch_predictor(path, model, record)
    if os.environ.get("MLFLOW_TRACKING_URI"):
        log_to_mlflow(
            args.mlflow_experiment,
            f"{manifest['encoder']}-{name}",
            record,
            path,
        )
    print(f"Saved {path}")
    print(json.dumps({"best_epoch": best_epoch, **losses[best_epoch - 1]}, indent=2))


if __name__ == "__main__":
    main()

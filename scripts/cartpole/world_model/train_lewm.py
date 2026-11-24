"""Train a LeWM JEPA end to end from CartPole frames.

The encoder and the action-conditioned predictor train together on windows of
``--history + 1`` frames ``--step-stride`` control steps apart, with the mean
squared error between predicted and actual next embeddings plus
``--sigreg-weight`` times SIGReg (see ``wm.lewm``), the recipe of LeWM
(arXiv 2603.19312): AdamW, a cosine schedule, bfloat16, and no stop-gradient or
target network. Collapse monitors (embedding spread, effective rank, a ridge
read-out of the state) are logged with every validation pass.

The model is saved under ``--output-root`` and reaches the rest of the pipeline
as an encoder named ``lewm:<artifact>``: ``embed.py`` caches its embeddings,
``train_probe.py`` fits state probes on them, ``eval_lewm.py`` scores rollouts,
and ``latent_mpc.py`` / ``action_response.py`` load the model itself. Runs are
logged to MLflow, checkpoint included, when MLFLOW_TRACKING_URI is set.
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from world_model_playground.cartpole import non_negative_int, positive_int
from world_model_playground.wm.data import split_episode_ids
from world_model_playground.wm.episodes import (
    episode_lengths,
    is_episodic,
    load_manifest,
)
from world_model_playground.wm.lewm import (
    FrameWindows,
    LeWM,
    LeWMConfig,
    encoder_name,
    fit_lewm,
    save_lewm,
)
from world_model_playground.wm.runtime import torch_device
from world_model_playground.wm.sigreg import SIGReg

SPLITS = ("train", "validation", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--collection-dir",
        type=Path,
        required=True,
        help="episodic collection written by collect_obs.py",
    )
    parser.add_argument(
        "--resolution",
        type=positive_int,
        default=64,
        help="square frame size, which must have been collected; it is also the "
        "encoder's input size (default: 64)",
    )
    parser.add_argument(
        "--patch-size",
        type=positive_int,
        default=8,
        help="ViT patch size (default: 8, i.e. 8x8 patches of a 64x64 frame; "
        "LeWM uses 14 on 224x224 frames)",
    )
    parser.add_argument("--embed-dim", type=positive_int, default=192)
    parser.add_argument("--seed", type=int, default=0, help="episode split seed")
    parser.add_argument(
        "--history",
        type=positive_int,
        default=3,
        help="model steps of context the predictor reads (default: 3)",
    )
    parser.add_argument(
        "--step-stride",
        type=positive_int,
        default=5,
        help="control steps per model step, LeWM's frame skip (default: 5)",
    )
    parser.add_argument(
        "--sigreg-weight",
        type=float,
        default=0.09,
        help="weight of SIGReg, LeWM's only loss hyperparameter; 0 disables the "
        "regularizer and lets the model collapse (default: 0.09)",
    )
    parser.add_argument("--num-projections", type=positive_int, default=1024)
    parser.add_argument("--knots", type=positive_int, default=17)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--iterations", type=positive_int, default=20_000)
    parser.add_argument("--batch-size", type=positive_int, default=128)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--warmup", type=non_negative_int, default=500)
    parser.add_argument("--eval-every", type=positive_int, default=500)
    parser.add_argument(
        "--eval-windows",
        type=positive_int,
        default=512,
        help="validation windows scored every --eval-every iterations",
    )
    parser.add_argument("--num-workers", type=non_negative_int, default=2)
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="recompute encoder activations in the backward pass, to save memory",
    )
    parser.add_argument(
        "--tag", default="", help="suffix that tells runs with the same settings apart"
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/cartpole_world_model/lewm"),
        help="models go to <root>/<collection_id>/<name>.pt",
    )
    parser.add_argument(
        "--mlflow-experiment",
        default=os.environ.get("MLFLOW_EXPERIMENT_NAME", "cartpole-wm"),
    )
    parser.add_argument("--device", help="torch device (default: cuda if available)")
    return parser.parse_args()


def run_name(args: argparse.Namespace) -> str:
    name = (
        f"lewm_r{args.resolution}_k{args.step_stride}_h{args.history}"
        f"_s{args.sigreg_weight:g}_i{args.iterations}"
    )
    return f"{name}_{args.tag}" if args.tag else name


def parameter_counts(model: LeWM) -> dict[str, int]:
    def count(*modules) -> int:
        return sum(p.numel() for m in modules for p in m.parameters())

    return {
        "encoder": count(model.encoder, model.projector),
        "predictor": count(
            model.predictor, model.action_encoder, model.prediction_projector
        ),
        "total": count(model),
    }


def main() -> None:
    args = parse_args()
    device = torch_device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = True
    manifest = load_manifest(args.collection_dir)
    if not is_episodic(manifest):
        raise ValueError(f"{args.collection_dir} is not an episodic collection")
    ids = np.fromiter(episode_lengths(manifest), dtype=np.int64)
    splits = dict(zip(SPLITS, split_episode_ids(ids, seed=args.seed), strict=True))
    resolution = (args.resolution, args.resolution)

    config = LeWMConfig(
        image_size=args.resolution,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        dropout=args.dropout,
        history=args.history,
        step_stride=args.step_stride,
        gradient_checkpointing=args.gradient_checkpointing,
    )
    start = time.perf_counter()
    windows = {
        split: FrameWindows(
            args.collection_dir,
            splits[split],
            resolution=resolution,
            frames=config.frames,
            stride=args.step_stride,
            start_every=1 if split == "train" else args.step_stride,
        )
        for split in ("train", "validation")
    }
    print(
        f"Windows: {len(windows['train'])} train, {len(windows['validation'])} "
        f"validation ({time.perf_counter() - start:.0f}s to load)"
    )

    torch.manual_seed(args.seed)
    model = LeWM(config)
    counts = parameter_counts(model)
    print(f"Parameters: {counts}")
    name = run_name(args)
    path = args.output_root / manifest["collection_id"] / f"{name}.pt"
    params = {
        "encoder": encoder_name(path),
        "context_frames": 1,
        "collection_id": manifest["collection_id"],
        "probe": None,
        "seed": args.seed,
        "resolution": args.resolution,
        "patch_size": args.patch_size,
        "embed_dim": args.embed_dim,
        "history": args.history,
        "step_stride": args.step_stride,
        "sigreg_weight": args.sigreg_weight,
        "num_projections": args.num_projections,
        "knots": args.knots,
        "dropout": args.dropout,
        "iterations": args.iterations,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup": args.warmup,
        "parameters": counts["total"],
    }

    mlflow = None
    if os.environ.get("MLFLOW_TRACKING_URI"):
        import mlflow

        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
        mlflow.set_experiment(args.mlflow_experiment)
        mlflow.start_run(run_name=name, tags={"stage": "lewm"})
        mlflow.log_params(params)

    def log(entry: dict) -> None:
        print(
            f"it {entry['iteration']:>6}  train pred {entry['train_prediction']:.4f} "
            f"sigreg {entry['train_sigreg']:.2f}  val pred "
            f"{entry['validation_prediction']:.4f} sigreg {entry['validation_sigreg']:.2f}  "
            f"rank {entry['validation_effective_rank']:.1f}  std_min "
            f"{entry['validation_std_min']:.2f}  r2 pose "
            f"{entry.get('validation_monitor_r2_pose', float('nan')):.3f} vel "
            f"{entry.get('validation_monitor_r2_velocity', float('nan')):.3f}  "
            f"{entry['seconds']:.0f}s",
            flush=True,
        )
        if mlflow is not None:
            mlflow.log_metrics(
                {k: v for k, v in entry.items() if k != "iteration"},
                step=entry["iteration"],
            )

    train_start = time.perf_counter()
    model, best_iteration, losses = fit_lewm(
        model,
        windows["train"],
        windows["validation"],
        sigreg=SIGReg(args.knots, args.num_projections),
        sigreg_weight=args.sigreg_weight,
        iterations=args.iterations,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup=args.warmup,
        eval_every=args.eval_every,
        eval_windows=args.eval_windows,
        num_workers=args.num_workers,
        seed=args.seed,
        device=device,
        log=log,
    )
    train_seconds = time.perf_counter() - train_start
    peak_gb = (
        torch.cuda.max_memory_allocated(device) / 1e9 if device.type == "cuda" else None
    )

    record = {
        "params": params,
        "collection_dir": str(args.collection_dir.resolve()),
        "episode_splits": {k: v.tolist() for k, v in splits.items()},
        "training": {
            "best_iteration": best_iteration,
            "losses": losses,
            "seconds": train_seconds,
            "peak_gpu_gb": peak_gb,
            "parameters": counts,
        },
    }
    save_lewm(path, model, record)
    path.with_suffix(".json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    if mlflow is not None:
        summary = {"best_iteration": best_iteration, "train_seconds": train_seconds}
        if peak_gb is not None:
            summary["peak_gpu_gb"] = peak_gb
        mlflow.log_metrics(summary)
        mlflow.log_artifact(str(path), "lewm")
        mlflow.log_artifact(str(path.with_suffix(".json")), "lewm")
        mlflow.end_run()
    print(
        f"Saved {path} (best iteration {best_iteration}/{args.iterations}, "
        f"{train_seconds:.0f}s"
        + (f", peak {peak_gb:.1f} GB" if peak_gb is not None else "")
        + ")"
    )
    print(f"Encoder name for embed.py: {params['encoder']}")


if __name__ == "__main__":
    main()

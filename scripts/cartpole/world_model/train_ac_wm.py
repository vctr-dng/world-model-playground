"""Post-train V-JEPA 2-AC's action-conditioned predictor on CartPole tokens.

The predictor released with V-JEPA 2-AC (pretrained on DROID robot video) is
adapted to CartPole (see ``wm.ac_world_model``): one model step spans
``--step-stride`` control steps, whose forces feed a new action layer, and the
state token is constant. It trains on the patch tokens of the frozen AC encoder
cached by ``embed.py --encoder vjepa2-ac-vitg --store-tokens``, with the L1
teacher-forced and open-loop losses of V-JEPA 2-AC training.

``--init random`` trains the same architecture from scratch and
``--iterations 0`` evaluates the adapted model without post-training, which
together tell whether DROID pretraining helps. Rollouts are evaluated like the
Phase 3 latent model, at horizons that are multiples of the stride: decoded by
a state probe stacking steps ``--step-stride`` apart (``train_probe.py
--history-stride``), against the linear model, the probe on true features, and
persistence. Runs are also logged to MLflow when MLFLOW_TRACKING_URI is set.
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from world_model_playground.cli import non_negative_int, positive_int
from world_model_playground.wm.ac_checkpoint import load_checkpoint
from world_model_playground.wm.ac_world_model import (
    ACConfig,
    TokenWindows,
    ac_rollouts,
    build_ac_model,
    evaluate_ac_losses,
    fit_ac_model,
    npz_member_memmap,
    save_ac_model,
)
from world_model_playground.wm.data import split_episode_ids
from world_model_playground.wm.embeddings import (
    load_embedded_episodes,
    load_embeddings_manifest,
)
from world_model_playground.wm.episodes import episode_lengths, episode_path
from world_model_playground.wm.eval import OBSERVATION_GROUPS, group_rmse
from world_model_playground.wm.linear import fit_linear_dynamics
from world_model_playground.wm.probes import check_probe, load_probe
from world_model_playground.wm.rollout_eval import (
    evaluate_rollouts,
    evaluation_horizons,
)
from world_model_playground.wm.runtime import torch_device
from world_model_playground.wm.tracking import flat_rollout_metrics, tracked_run

SPLITS = ("train", "validation", "test")

LATENT_KEY = "pooled_latent_rollout_rmse"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--embeddings-dir",
        type=Path,
        required=True,
        help="embeddings of the vjepa2-ac-vitg encoder with stored tokens",
    )
    parser.add_argument(
        "--probe",
        type=Path,
        help="state probe stacking steps --step-stride apart (default: "
        "<probe-root>/<collection_id>/<embeddings>/mlp_h2_s<stride>.pt)",
    )
    parser.add_argument(
        "--probe-root",
        type=Path,
        default=Path("artifacts/cartpole_world_model/probes"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="V-JEPA 2-AC checkpoint (default: the torch hub cache, downloaded "
        "if missing)",
    )
    parser.add_argument("--seed", type=int, default=0, help="episode split seed")
    parser.add_argument(
        "--init",
        choices=("pretrained", "random"),
        default="pretrained",
        help="start from the DROID-pretrained predictor or from scratch",
    )
    parser.add_argument(
        "--step-stride",
        type=positive_int,
        default=5,
        help="control steps per model step (default: 5, i.e. 0.05 s)",
    )
    parser.add_argument(
        "--history",
        type=positive_int,
        default=3,
        help="model steps of context the predictor reads (default: 3)",
    )
    parser.add_argument(
        "--rollout-steps",
        type=positive_int,
        default=2,
        help="open-loop model steps in the training loss (default: 2)",
    )
    parser.add_argument("--rollout-weight", type=float, default=1.0)
    parser.add_argument(
        "--iterations",
        type=non_negative_int,
        default=3000,
        help="training iterations; 0 evaluates the adapted model as is",
    )
    parser.add_argument("--batch-size", type=positive_int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--warmup", type=non_negative_int, default=100)
    parser.add_argument("--weight-decay", type=float, default=0.04)
    parser.add_argument(
        "--freeze-blocks",
        type=non_negative_int,
        default=0,
        help="pretrained blocks kept frozen from the input, to save memory",
    )
    parser.add_argument(
        "--activation-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="recompute block activations in the backward pass (default: on)",
    )
    parser.add_argument("--eval-every", type=positive_int, default=250)
    parser.add_argument(
        "--eval-windows",
        type=positive_int,
        default=512,
        help="validation windows scored every --eval-every iterations",
    )
    parser.add_argument("--num-workers", type=non_negative_int, default=4)
    parser.add_argument(
        "--start-every",
        type=positive_int,
        default=20,
        help="start evaluation rollouts every this many steps of each episode",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/cartpole_world_model/ac_wm"),
        help="models go to <root>/<collection_id>/<embeddings>/"
        "ac_<init>_k<stride>_h<history>_r<rollout_steps>_i<iterations>.pt",
    )
    parser.add_argument(
        "--mlflow-experiment",
        default=os.environ.get("MLFLOW_EXPERIMENT_NAME", "cartpole-wm"),
    )
    parser.add_argument("--device", help="torch device (default: cuda if available)")
    return parser.parse_args()


def log_to_mlflow(experiment: str, run_name: str, record: dict, path: Path) -> None:
    with tracked_run(experiment, run_name, "ac_world_model") as mlflow:
        mlflow.log_params(record["params"])
        for losses in record["training"]["losses"]:
            step = losses["iteration"]
            mlflow.log_metrics(
                {k: v for k, v in losses.items() if k != "iteration"}, step=step
            )
        mlflow.log_metrics(
            flat_rollout_metrics(
                "validation",
                record["metrics"]["validation"],
                latent_key=LATENT_KEY,
                latent_label="pooled_latent",
            )
            | flat_rollout_metrics(
                "test",
                record["metrics"]["test"],
                latent_key=LATENT_KEY,
                latent_label="pooled_latent",
            )
        )
        mlflow.log_dict(record, "ac_wm.json")


def main() -> None:
    args = parse_args()
    device = torch_device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = True
    manifest = load_embeddings_manifest(args.embeddings_dir)
    if not manifest.get("stores_tokens"):
        raise ValueError("the embeddings need tokens: rerun embed.py --store-tokens")
    if manifest["encoder"] != "vjepa2-ac-vitg":
        raise ValueError("the AC predictor reads tokens of the vjepa2-ac-vitg encoder")
    ids = np.fromiter(episode_lengths(manifest), dtype=np.int64)
    splits = dict(zip(SPLITS, split_episode_ids(ids, seed=args.seed), strict=True))

    stride = args.step_stride
    probe_name = f"mlp_h2_s{stride}.pt" if stride > 1 else "mlp_h2.pt"
    probe_path = args.probe or (
        args.probe_root
        / manifest["collection_id"]
        / args.embeddings_dir.name
        / probe_name
    )
    probe, probe_metadata = load_probe(probe_path, device)
    check_probe(probe_metadata, manifest, splits)
    if probe.stride != stride:
        raise ValueError(f"the probe stacks steps {probe.stride} apart, not {stride}")

    config = ACConfig(
        step_stride=stride,
        history=args.history,
        pretrained=args.init == "pretrained",
        activation_checkpointing=args.activation_checkpointing,
        freeze_blocks=args.freeze_blocks,
    )
    checkpoint = load_checkpoint(args.checkpoint) if config.pretrained else None
    model = build_ac_model(config, checkpoint)
    del checkpoint
    length = args.history + args.rollout_steps
    windows = {
        split: TokenWindows(
            args.embeddings_dir,
            splits[split],
            length=length,
            stride=stride,
            start_every=1 if split == "train" else stride,
        )
        for split in ("train", "validation")
    }
    start = time.perf_counter()
    if args.iterations:
        model, best_iteration, losses = fit_ac_model(
            model,
            windows["train"],
            windows["validation"],
            rollout_steps=args.rollout_steps,
            rollout_weight=args.rollout_weight,
            iterations=args.iterations,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            warmup=args.warmup,
            weight_decay=args.weight_decay,
            eval_every=args.eval_every,
            eval_windows=args.eval_windows,
            num_workers=args.num_workers,
            seed=args.seed,
            device=device,
        )
    else:
        model = model.to(device).eval()
        items = np.linspace(
            0,
            len(windows["validation"]) - 1,
            min(args.eval_windows, len(windows["validation"])),
        )
        one_step, open_loop = evaluate_ac_losses(
            model,
            windows["validation"],
            np.unique(items.astype(np.int64)),
            rollout_steps=args.rollout_steps,
            batch_size=args.batch_size,
            device=device,
        )
        best_iteration = 0
        losses = [
            {
                "iteration": 0,
                "validation_one_step": one_step,
                "validation_open_loop": open_loop,
            }
        ]
    train_seconds = time.perf_counter() - start
    peak_gb = (
        torch.cuda.max_memory_allocated(device) / 1e9 if device.type == "cuda" else None
    )

    train = load_embedded_episodes(
        args.embeddings_dir, splits["train"], fields=("observations", "actions")
    )
    linear_model = fit_linear_dynamics(
        np.concatenate([e["observations"][:-1] for e in train.values()]),
        np.concatenate([e["actions"] for e in train.values()]),
        np.concatenate([e["observations"][1:] for e in train.values()]),
        ridge=1e-6,
    )
    horizons = evaluation_horizons(stride)
    metrics = {}
    for split in ("validation", "test"):
        episodes = load_embedded_episodes(
            args.embeddings_dir,
            splits[split],
            fields=("features", "observations", "actions"),
        )
        for episode_id, episode in episodes.items():
            episode["tokens"] = npz_member_memmap(
                episode_path(args.embeddings_dir, episode_id), "tokens"
            )
        metrics[split] = evaluate_rollouts(
            episodes,
            label="ac_model",
            rollouts=ac_rollouts(model, probe),
            probe=probe,
            linear_model=linear_model,
            start_every=args.start_every,
            horizons=horizons,
            latent_key=LATENT_KEY,
        )

    name = f"ac_{args.init}_k{stride}_h{args.history}_r{args.rollout_steps}"
    record = {
        "params": {
            "encoder": manifest["encoder"],
            "checkpoint": manifest["checkpoint"],
            "context_frames": manifest["context_frames"],
            "collection_id": manifest["collection_id"],
            "probe": str(probe_path),
            "probe_kind": probe.kind,
            "probe_history": probe.history,
            "init": args.init,
            "seed": args.seed,
            "step_stride": stride,
            "history": args.history,
            "rollout_steps": args.rollout_steps,
            "rollout_weight": args.rollout_weight,
            "iterations": args.iterations,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "warmup": args.warmup,
            "weight_decay": args.weight_decay,
            "freeze_blocks": args.freeze_blocks,
            "activation_checkpointing": args.activation_checkpointing,
            "start_every": args.start_every,
        },
        "embeddings_dir": str(args.embeddings_dir.resolve()),
        "episode_splits": {k: v.tolist() for k, v in splits.items()},
        "training": {
            "best_iteration": best_iteration,
            "losses": losses,
            "seconds": train_seconds,
            "peak_gpu_gb": peak_gb,
        },
        "metrics": metrics,
    }
    path = (
        args.output_root
        / manifest["collection_id"]
        / args.embeddings_dir.name
        / f"{name}_i{args.iterations}.pt"
    )
    save_ac_model(path, model, record)
    path.with_suffix(".json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    if os.environ.get("MLFLOW_TRACKING_URI"):
        log_to_mlflow(args.mlflow_experiment, name, record, path)
    print(
        f"Saved {path} (best iteration {best_iteration}/{args.iterations}, "
        f"{train_seconds:.0f}s"
        + (f", peak {peak_gb:.1f} GB" if peak_gb is not None else "")
        + ")"
    )

    observations = metrics["test"]["observation_rollout_rmse"]
    for group in OBSERVATION_GROUPS:
        print(f"\nTest decoded {group} rollout RMSE ({name}, probe {probe.kind})")
        print(" | ".join(["horizon", *observations]))
        for horizon in map(str, horizons):
            if horizon in observations["ac_model"]:
                cells = [
                    f"{group_rmse(errors[horizon], group):.4f}"
                    for errors in observations.values()
                ]
                print(" | ".join([horizon, *cells]))


if __name__ == "__main__":
    main()

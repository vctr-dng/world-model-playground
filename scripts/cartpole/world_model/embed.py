"""Encode an episodic Cartpole collection with a frozen pretrained encoder.

Writes the pooled feature of the frame window ending at every step (and,
optionally, the patch tokens) next to the observations and actions, so state
probes and latent world models train without touching the frames again.
Frames are taken from the closest square resolution of the collection at or
above the encoder's input size, or re-rendered from the stored physics state.
"""

import argparse
import os
import time
from pathlib import Path

# Headless EGL renders far faster than GLFW; set MUJOCO_GL to override.
os.environ.setdefault("MUJOCO_GL", "egl")

from world_model_playground.wm.runtime import share_gpu_with_jax

# Re-rendering loads the MJX environment, so JAX must not reserve the GPU first.
share_gpu_with_jax()

from world_model_playground.cartpole import (
    FrameRenderer,
    load_cartpole,
    positive_int,
)
from world_model_playground.wm.embeddings import (
    COPIED_FIELDS,
    default_embeddings_dir,
    embed_frames,
    write_embedded_episode,
    write_embeddings_manifest,
)
from world_model_playground.wm.encoders import ENCODERS, load_encoder
from world_model_playground.wm.episodes import (
    episode_lengths,
    frames_key,
    is_episodic,
    load_episode,
    load_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--collection-dir",
        type=Path,
        required=True,
        help="episodic collection written by collect_obs.py",
    )
    parser.add_argument("--encoder", choices=sorted(ENCODERS), required=True)
    parser.add_argument(
        "--context-frames",
        type=positive_int,
        help="frames per encoded window; 1 for DINOv2, a multiple of 2 for "
        "V-JEPA 2 (default: the encoder's default)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="embeddings directory "
        "(default: data/embeddings/<collection_id>/<encoder>_ctx<context>)",
    )
    parser.add_argument("--batch-size", type=positive_int, default=64)
    parser.add_argument(
        "--store-tokens",
        action="store_true",
        help="also store float16 patch tokens for patch-level world models; "
        "about 200 KB (DINOv2-S) to 500 KB (V-JEPA 2 ViT-L) per step",
    )
    parser.add_argument("--device", help="torch device (default: cuda if available)")
    return parser.parse_args()


def source_resolution(manifest: dict, size: int) -> tuple[int, int] | None:
    """Smallest square collected resolution of at least ``size``, if any."""
    squares = sorted(
        entry["height"]
        for entry in manifest["resolutions"]
        if entry["height"] == entry["width"] and entry["height"] >= size
    )
    return (squares[0], squares[0]) if squares else None


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.collection_dir)
    if not is_episodic(manifest):
        raise ValueError(
            f"{args.collection_dir} is a format-1 collection; recollect it with "
            "collect_obs.py to get episode shards"
        )
    encoder = load_encoder(
        args.encoder, context_frames=args.context_frames, device=args.device
    )
    output_dir = args.output_dir or default_embeddings_dir(
        manifest["collection_id"], args.encoder, encoder.context_frames
    )
    if output_dir.exists():
        raise FileExistsError(f"Embeddings directory already exists: {output_dir}")

    size = encoder.spec.input_size
    resolution = source_resolution(manifest, size)
    renderer = None
    if resolution is None:
        resolution = (size, size)
        renderer = FrameRenderer(
            load_cartpole(full_circle_reset=manifest["full_circle_reset"]).mj_model,
            [resolution],
            manifest["camera"],
        )
        print(f"No square frames of at least {size}px; re-rendering from qpos/qvel")
    image_key = frames_key(*resolution)

    lengths = episode_lengths(manifest)
    fields = (
        [*COPIED_FIELDS, "qpos", "qvel"] if renderer else [*COPIED_FIELDS, image_key]
    )
    start = time.perf_counter()
    try:
        for count, episode_id in enumerate(lengths, start=1):
            episode = load_episode(args.collection_dir, episode_id, fields=fields)
            frames = (
                renderer.render(episode["qpos"], episode["qvel"], resolution=resolution)
                if renderer
                else episode[image_key]
            )
            features, tokens = embed_frames(
                encoder,
                frames,
                batch_size=args.batch_size,
                keep_tokens=args.store_tokens,
            )
            write_embedded_episode(output_dir, episode_id, episode, features, tokens)
            if count % 10 == 0 or count == len(lengths):
                elapsed = time.perf_counter() - start
                print(f"Encoded {count}/{len(lengths)} episodes ({elapsed:.0f}s)")
    finally:
        if renderer is not None:
            renderer.close()

    write_embeddings_manifest(
        output_dir,
        {
            "collection_id": manifest["collection_id"],
            "collection_dir": str(args.collection_dir.resolve()),
            "control_dt": manifest["control_dt"],
            "encoder": args.encoder,
            "checkpoint": encoder.spec.checkpoint,
            "family": encoder.spec.family,
            "input_size": size,
            "context_frames": encoder.context_frames,
            "feature_dim": encoder.feature_dim,
            "token_dim": encoder.token_dim,
            "stores_tokens": args.store_tokens,
            "source_frames": image_key,
            "rerendered": renderer is not None,
            "dtype": str(encoder.dtype).removeprefix("torch."),
            "episodes": manifest["episodes"],
        },
    )
    print(f"Embeddings: {output_dir} (feature_dim={encoder.feature_dim})")


if __name__ == "__main__":
    main()

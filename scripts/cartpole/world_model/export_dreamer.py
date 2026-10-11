"""Export an episodic Cartpole collection as DreamerV3 episode files.

Writes one ``<episode>-<steps>.npz`` per episode into ``train``,
``validation``, and ``test`` directories (the same episode split as the other
world models), with the step arrays of ``wm.dreamer.to_dreamer_episode``.
Frames missing from the collection are re-rendered from the stored physics state.
"""

import argparse
import json
import os
from pathlib import Path

# Headless EGL renders far faster than GLFW; set MUJOCO_GL to override.
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

from world_model_playground.cartpole import (
    FrameRenderer,
    load_cartpole,
    parse_resolution,
)
from world_model_playground.wm.data import split_episode_ids
from world_model_playground.wm.dreamer import ACTION_ALIGNMENTS, to_dreamer_episode
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
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="export directory (default: data/dreamer/<collection_id>_<H>x<W>)",
    )
    parser.add_argument(
        "--resolution",
        type=parse_resolution,
        default=(64, 64),
        metavar="HxW",
        help="image size of the exported episodes (default: 64x64)",
    )
    parser.add_argument(
        "--action-alignment",
        choices=ACTION_ALIGNMENTS,
        default="preceding",
        help="which action each step carries; 'preceding' matches dreamerv3-torch, "
        "'following' the original dreamerv3 replay (default: preceding)",
    )
    parser.add_argument("--seed", type=int, default=0, help="episode split seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.collection_dir)
    if not is_episodic(manifest):
        raise ValueError(
            f"{args.collection_dir} is a format-1 collection; recollect it with "
            "collect_obs.py to get episode shards"
        )
    image_key = frames_key(*args.resolution)
    output_dir = args.output_dir or Path(
        f"data/dreamer/{manifest['collection_id']}_{args.resolution[0]}x{args.resolution[1]}"
    )
    if output_dir.exists():
        raise FileExistsError(f"Export directory already exists: {output_dir}")

    lengths = episode_lengths(manifest)
    splits = dict(
        zip(
            ("train", "validation", "test"),
            split_episode_ids(np.fromiter(lengths, dtype=np.int64), seed=args.seed),
            strict=True,
        )
    )
    rendered = {entry["key"] for entry in manifest["resolutions"]}
    renderer = None
    if image_key not in rendered:
        renderer = FrameRenderer(
            load_cartpole(full_circle_reset=manifest["full_circle_reset"]).mj_model,
            [args.resolution],
            manifest["camera"],
        )
        print(f"{image_key} was not collected; re-rendering it from qpos/qvel")
    try:
        for split, episode_ids in splits.items():
            split_dir = output_dir / split
            split_dir.mkdir(parents=True)
            for episode_id in sorted(int(i) for i in episode_ids):
                episode = load_episode(args.collection_dir, episode_id)
                if renderer is not None:
                    episode[image_key] = renderer.render(
                        episode["qpos"], episode["qvel"], resolution=args.resolution
                    )
                steps = to_dreamer_episode(
                    episode,
                    image_key=image_key,
                    action_alignment=args.action_alignment,
                )
                np.savez_compressed(
                    split_dir / f"{episode_id:06d}-{len(steps['reward'])}.npz", **steps
                )
    finally:
        if renderer is not None:
            renderer.close()

    export = {
        "collection_id": manifest["collection_id"],
        "collection_dir": str(args.collection_dir.resolve()),
        "resolution": list(args.resolution),
        "action_alignment": args.action_alignment,
        "control_dt": manifest["control_dt"],
        "seed": args.seed,
        "keys": [
            "image",
            "state",
            "action",
            "reward",
            "discount",
            "is_first",
            "is_last",
            "is_terminal",
        ],
        "episode_splits": {
            split: sorted(int(i) for i in ids) for split, ids in splits.items()
        },
    }
    (output_dir / "dreamer.json").write_text(
        json.dumps(export, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for split, ids in splits.items():
        print(f"{split}: {len(ids)} episodes in {output_dir / split}")


if __name__ == "__main__":
    main()

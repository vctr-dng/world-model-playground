# Data collection

How episodes are collected from the PPO policy, what a collection contains, the perturbation recipe that gives world models sustained forces near balance, and the DreamerV3 export.

## Collecting episodes

Collect episodes from the best PPO run in MLflow. Episodes run in parallel MJX
environments (`--num-envs`) and mix deterministic, stochastic, and uniformly
random actions (40/30/30 by default) so learned models see off-expert states:

```bash
uv run python scripts/cartpole/data/collect_obs.py \
  --num-episodes 500 --episode-length 200 --resolutions 64x64 256x256
```

Each collection directory holds:

- `episodes/episode_<id>.npz`: one compressed shard per episode with
  `observations` (T+1), `actions`, `rewards`, `action_source` (T), the physics
  state `qpos`/`qvel` (T+1), and `frames_<H>x<W>` (T+1) for every resolution
- `transitions.npz`: the flattened state-only transitions used by the linear model
- `collection.json`: collection parameters, rendered resolutions, and episode index

Every resolution keeps the horizontal view of the 320x240 `fixed` camera, so
square frames still show the cart at both rail ends. 64x64 suits DreamerV3,
256x256 suits V-JEPA 2 (resize it to 224x224 for DINOv2). Rendering defaults to
headless EGL (`MUJOCO_GL=egl`), which is much faster than GLFW.

`world_model_playground.wm.episodes` reads and validates shards,
`wm.datasets.EpisodeClipDataset` serves fixed-length PyTorch clips of frames,
observations, and actions, and `scripts/cartpole/notebooks/visualize_obs.py`
browses both episodic and older single-file collections.

## Perturbations around balance

To show the model sustained forces where balancing happens, collect episodes
that start near upright and, while the pole is upright (cos θ > 0.9), start a
perturbation with 5% chance per step: one uniform random force held for 10 to
30 steps over the scheduled actions, after which the expert recovers the pole:

```bash
uv run python scripts/cartpole/data/collect_obs.py \
  --num-episodes 200 --no-full-circle-reset \
  --deterministic-fraction 0.6 --stochastic-fraction 0.3 --source-hold 10 \
  --perturb-probability 0.05 --perturb-steps 10 30 --resolutions 256x256
```

Perturbations make up 26% of the steps and are recorded as a fourth action
source; 49% of states are near upright (cos θ > 0.9), against 15% with held
actions alone.

The effect on what a latent model learns about actions is in [Latent MPC in closed loop](latent-mpc.md#perturbations-around-balance).

## DreamerV3 export

Export a collection as DreamerV3 episode files (`image`, `state`, `action`,
`reward`, `discount`, `is_first`, `is_last`, `is_terminal`, one entry per step)
in `train`/`validation`/`test` directories using the same episode split as the
other models:

```bash
uv run python scripts/cartpole/data/export_dreamer.py \
  --collection-dir data/cartpole_trajectories/<collection_id> --resolution 64x64
```

`--action-alignment preceding` (default) stores with each step the action that
led to it, as `dreamerv3-torch` expects; `following` stores the action taken
from it, as the original `dreamerv3` replay does. Resolutions that were not
collected are re-rendered from the stored `qpos`/`qvel`.

---

Previous: [Expert policy](expert-policy.md) · Next: [Frozen encoders and state probes](encoders-and-probes.md) · [README](../README.md)

# World Model Playground

## Cartpole Balance

### Parallel MJX Random Agent

Run a random agent in parallel `CartpoleBalance` environments:

```bash
uv run python scripts/cartpole/random_agent.py --num-envs 2048 --num-steps 500 --seed 42
```

Display a specific environment from the parallel batch in a live MuJoCo viewer:

```bash
uv run python scripts/cartpole/random_agent.py --render --render-env 12
```

Select another environment with `--render-env`, which must be smaller than
`--num-envs`.

### Brax PPO

Train a PPO policy using parallel MJX environments and Playground's tuned
`CartpoleBalance` hyperparameters. The default training budget is 60 million
environment steps. Training metrics, parameters, configuration, and the final
checkpoint are logged to MLflow. The tracking URI defaults to
`http://127.0.0.1:5000` and can be overridden with `MLFLOW_TRACKING_URI`:

```bash
MLFLOW_TRACKING_URI=http://127.0.0.1:5000 \
  uv run python scripts/cartpole/train_ppo.py
```

The default MLflow experiment is `cartpole-ppo`; override it with
`--mlflow-experiment` or `MLFLOW_EXPERIMENT_NAME`.

Render the deterministic policy after training:

```bash
uv run python scripts/cartpole/train_ppo.py --render
```

For a short pipeline check rather than a converged policy:

```bash
uv run python scripts/cartpole/train_ppo.py \
  --num-envs 64 --num-timesteps 2048 --num-evals 1 --num-eval-envs 8 \
  --batch-size 64 --num-minibatches 2 --num-updates-per-batch 1 \
  --unroll-length 4 --num-resets-per-eval 0 --max-devices 1
```

Checkpoints and evaluation videos are stored in MLflow.
Every evaluation's checkpoint is logged under `checkpoints/<step>` and the final one is also logged as `checkpoint`.
Load a checkpoint for evaluation and live playback (a local checkpoint directory also works):

```bash
uv run python scripts/cartpole/train_ppo.py \
  --play-only \
  --checkpoint runs:/<run_id>/checkpoint \
  --render
```

### World models

Collect episodes from the best PPO run in MLflow. Episodes run in parallel MJX
environments (`--num-envs`) and mix deterministic, stochastic, and uniformly
random actions (40/30/30 by default) so learned models see off-expert states:

```bash
uv run python scripts/cartpole/world_model/collect_obs.py \
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
observations, and actions, and `scripts/cartpole/world_model/visualize_obs.py`
browses both episodic and older single-file collections.

#### DreamerV3 export

Export a collection as DreamerV3 episode files (`image`, `state`, `action`,
`reward`, `discount`, `is_first`, `is_last`, `is_terminal`, one entry per step)
in `train`/`validation`/`test` directories using the same episode split as the
other models:

```bash
uv run python scripts/cartpole/world_model/export_dreamer.py \
  --collection-dir data/cartpole_trajectories/<collection_id> --resolution 64x64
```

`--action-alignment preceding` (default) stores with each step the action that
led to it, as `dreamerv3-torch` expects; `following` stores the action taken
from it, as the original `dreamerv3` replay does. Resolutions that were not
collected are re-rendered from the stored `qpos`/`qvel`.

#### Frozen encoders and state probes

Encode a collection once with a frozen pretrained encoder (DINOv2 or V-JEPA 2
from the Hugging Face Hub). Each step gets the pooled feature of the frame
window ending at it, stored with the observations and actions under
`data/embeddings/<collection_id>/<encoder>_ctx<frames>/`:

```bash
uv run python scripts/cartpole/world_model/embed.py \
  --collection-dir data/cartpole_trajectories/<collection_id> --encoder vjepa2-vitl
```

Encoders: `dinov2-small|base|large` (single frames, 224x224) and
`vjepa2-vitl|vith|vitg` (clips of `--context-frames`, a multiple of 2, at
256x256). Frames are resized without the processors' center crop, so the rail
ends stay visible; resolutions that were not collected are re-rendered.
`--store-tokens` also keeps float16 patch tokens for patch-level world models.

Fit linear (ridge) and MLP probes from the stacked features of the last
`--history` steps back to the 5-dimensional observation, and compare encoders
on the shared episode split:

```bash
uv run python scripts/cartpole/world_model/train_probe.py \
  --embeddings-dir data/embeddings/<collection_id>/dinov2-small_ctx1 \
                   data/embeddings/<collection_id>/vjepa2-vitl_ctx2
```

An encoder passes the gate when test R² >= 0.95 for the cart position and both
pole-angle components. Probes are saved under
`artifacts/cartpole_world_model/probes/` and restored with
`wm.probes.load_probe`; they decode latents back to observations for later
latent world models. Runs are logged to the `cartpole-wm` MLflow experiment
when `MLFLOW_TRACKING_URI` is set.

On 100 episodes, test R² was:

| Encoder | Probe | cart pos | cos θ | sin θ | cart vel | ang vel |
| --- | --- | --- | --- | --- | --- | --- |
| DINOv2-S, history 2 | linear | 0.996 | 0.990 | 0.977 | 0.227 | 0.360 |
| DINOv2-S, history 2 | MLP | 0.998 | 0.998 | 0.998 | 0.592 | 0.812 |
| V-JEPA 2 ViT-L, 2 frames, history 2 | linear | 0.996 | 0.986 | 0.982 | 0.916 | 0.844 |
| V-JEPA 2 ViT-L, 2 frames, history 2 | MLP | 0.999 | 0.999 | 0.999 | 0.947 | 0.964 |

#### Latent world model

Train an action-conditioned predictor on the cached features of an encoder. A
causal transformer reads the features and actions of the last `--history`
steps and predicts the next feature, with the encoder frozen. Training adds a
teacher-forced one-step loss to an open-loop loss over `--rollout-steps`
predictions fed back as inputs:

```bash
uv run python scripts/cartpole/world_model/train_latent_wm.py \
  --embeddings-dir data/embeddings/<collection_id>/vjepa2-vitl_ctx2
```

Predicted latents are decoded by the state probe of the same embeddings
(`--probe`, by default the MLP probe with history 2). The probe must have been
fitted with the same `--seed`, so it never saw the evaluation episodes. Open-loop
rollouts on the test episodes, starting every `--start-every` steps with the true
actions, are compared with three references on the same episodes: the linear
model, the probe applied to true features (the decoder's error floor), and
persistence. Predictors are saved under
`artifacts/cartpole_world_model/latent_wm/` and restored with
`wm.predictors.load_predictor`. Their batched `rollout` takes many candidate
action sequences at once, for latent planning. Runs are logged to the
`cartpole-wm` MLflow experiment when `MLFLOW_TRACKING_URI` is set.

On the 100-episode collection, with V-JEPA 2 ViT-L features, test rollout RMSE
of the decoded pose (cart position, cos θ, sin θ) was:

| Horizon | Latent model | Linear model | Probe on true features | Persistence |
| --- | --- | --- | --- | --- |
| 1 | 0.029 | 0.024 | 0.025 | 0.029 |
| 10 | 0.060 | 0.241 | 0.030 | 0.283 |
| 25 | 0.144 | 0.544 | 0.024 | 0.640 |
| 50 | 0.351 | 0.812 | 0.032 | 0.952 |
| 100 | 0.715 | 0.761 | 0.027 | 1.083 |

Over 10 to 50 steps, the latent model tracks the pose 2 to 4 times more
accurately than the linear model. Decoded velocities are limited by the probe
(0.67 RMSE even on true features), so the linear model stays better at 1 to 10
steps. With DINOv2-S features, which see one frame at a time, the latent model
only beats the linear model on pose from 10 to 50 steps.

#### Linear Model

Fit the linear residual baseline on a collection, then compare it against the
real environment under MPC:

```bash
uv run python scripts/cartpole/world_model/linear_model.py \
  --collection-dir data/cartpole_trajectories/<collection_id>
uv run marimo edit scripts/cartpole/world_model/mpc_divergence.py
```

Reusable code lives in the `world_model_playground` package: `policy` restores
PPO checkpoints, and `wm` holds the world-model data handling, models, and
metrics. `wm.eval` scores any `predict_next(observations, actions)` callable,
so new models report the same one-step and rollout RMSE as the linear baseline.

Simulation, PPO, and data collection use JAX; learned world models use PyTorch.
Both are built against CUDA 13 and can share the GPU in one process when
`world_model_playground.wm.runtime.share_gpu_with_jax()` runs before the first
JAX computation, which stops JAX from preallocating most of the GPU memory.

Run the tests, including a JAX/PyTorch GPU-sharing check when a GPU is present:

```bash
uv run pytest
```

## Docker

### MLflow

Start the MLflow server :

```bash
./docker/up.sh
```

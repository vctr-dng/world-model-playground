# Getting started

What you need to run the pipeline, how to start MLflow, how to share a GPU
between JAX and PyTorch, and where the code lives.

## Requirements

- Python 3.13 and [uv](https://docs.astral.sh/uv/).
- An NVIDIA GPU with a CUDA 13 driver. JAX (`jax[cuda13]`) and PyTorch (the
  `cu130` wheels) are both built against CUDA 13. The pipeline was developed on
  8 GB cards (two RTX 3070s); the pages note where memory is tight.
- Docker, for the MLflow server.

```bash
uv sync
```

## MLflow

PPO checkpoints, training runs and evaluation results live in MLflow. Start the
server (data goes to `docker/mlflow-data`; the first run creates `docker/.env`
from `docker/.env.example`, where you can change the port and bind address):

```bash
./docker/up.sh
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
```

Some scripts need the server and some only use it when it is configured:

- `train_ppo.py` logs to the `cartpole-ppo` experiment. `collect_obs.py` and the
  `ppo` controller of `latent_mpc.py` load the best PPO run from it, and
  `collect_obs.py` stops if `MLFLOW_TRACKING_URI` is unset.
- The training and evaluation scripts log to the `cartpole-wm` experiment only
  when `MLFLOW_TRACKING_URI` is set. Override the experiment with
  `--mlflow-experiment` or `MLFLOW_EXPERIMENT_NAME`.

To try a script without writing to the server, unset the variable for that
command: `env -u MLFLOW_TRACKING_URI uv run python ...`.

## Sharing a GPU

Simulation, PPO, and data collection use JAX; learned world models use PyTorch.
Both are built against CUDA 13 and can share the GPU in one process when
`world_model_playground.wm.runtime.share_gpu_with_jax()` runs before the first
JAX computation, which stops JAX from preallocating most of the GPU memory.
Scripts that mix the two do this themselves; set
`XLA_PYTHON_CLIENT_PREALLOCATE=false` when running JAX scripts next to other jobs.

Other settings that matter on small or shared cards:

- `CUDA_VISIBLE_DEVICES=0` pins a script to one GPU.
- On the two-GPU development machine, the default multi-device run of
  `train_ppo.py` segfaulted; `--max-devices 1` avoids it.
- `train_lewm.py --gradient-checkpointing` cuts peak memory from 3.7 GB to
  0.9 GB at batch 128 ([LeWM](lewm.md)).
- Rendering uses headless EGL (`MUJOCO_GL=egl`, set by the scripts that render).

## Tests and linting

```bash
uv run pytest
python scripts/check_docs.py   # links, anchors and script paths in these pages
```

The suite includes a JAX/PyTorch GPU-sharing check when a GPU is present.
[pre-commit](https://pre-commit.com) runs ruff and keeps `uv.lock` in sync:
`pre-commit install`. CI runs ruff and the docs check.

## Where the code lives

Scripts are thin command-line entry points, grouped by pipeline stage in
[`scripts/cartpole/`](../scripts/cartpole/README.md). Reusable code is the
`world_model_playground` package in `src/`:

| Module | Contents |
| --- | --- |
| `cartpole`, `policy`, `cli` | the CartpoleBalance environment and rendering, PPO checkpoint restore, argument types |
| `wm.episodes`, `wm.data`, `wm.datasets` | the episode-shard format, loading and splitting, PyTorch datasets |
| `wm.collection` | action-source schedules and the batched rollout behind `collect_obs.py` |
| `wm.encoders`, `wm.embeddings`, `wm.token_cache` | frozen encoders behind one interface, cached features and patch tokens |
| `wm.probes` | state probes that read the observation back out of features |
| `wm.predictors`, `wm.patch_predictor` | latent dynamics on pooled features and on patch tokens |
| `wm.lewm`, `wm.sigreg` | LeWM, the end-to-end JEPA, and its regularizer |
| `wm.ac_checkpoint`, `wm.ac_world_model`, `wm.vendor` | V-JEPA 2-AC, with Meta's predictor code vendored at a pinned commit |
| `wm.linear` | the linear residual baseline |
| `wm.planning`, `wm.control`, `wm.world_models` | CEM, the MPC controllers and episode runner, loading any saved model |
| `wm.eval`, `wm.rollout_eval`, `wm.tracking` | rollout metrics, scoring a model against the baselines, MLflow logging |

`wm.eval` scores any `predict_next(observations, actions)` callable, so a new
model reports the same one-step and rollout RMSE as the linear baseline.

---

[README](../README.md) · Next: [Expert policy](expert-policy.md)

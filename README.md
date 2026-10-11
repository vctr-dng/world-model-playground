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

Collect transitions and rendered frames from the best PPO run in MLflow:

```bash
uv run python scripts/cartpole/world_model/collect_obs.py
```

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

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

Load the latest checkpoint in a checkpoint directory for evaluation and live
playback:

```bash
uv run python scripts/cartpole/train_ppo.py \
  --play-only \
  --checkpoint runs/cartpole_ppo/<run>/checkpoints \
  --render
```

## Docker

### MLflow

Start the MLflow server :

```bash
./docker/up.sh
```

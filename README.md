# World Model Playground

World models on CartPole, from frozen foundation-model encoders to a JEPA trained
end to end from pixels. A PPO expert generates episodes with rendered frames; several
kinds of latent world model are trained on them and then judged the hard way, by
planning with CEM through the model to control the real simulator.

- **Frozen encoders, trained dynamics** (DINO-WM style): DINOv2 and V-JEPA 2 features,
  pooled or as patch-token grids, with an action-conditioned predictor on top.
- **V-JEPA 2-AC**: Meta's released action-conditioned predictor, post-trained on CartPole.
- **LeWM**: a ViT-tiny JEPA trained from pixels with a prediction loss and the SIGReg
  regularizer, with ablations of the regularizer, the history and the frame size.
- **Evaluation beyond rollout error**: state probes, counterfactual response to
  actions, and closed-loop MPC against PPO, random actions, and planners on the true
  and linear dynamics.

## Headline results

Balance return over 10 episodes of 200 steps (maximum 200, ± standard error), planning
with CEM through each model:

| Planner model | Balance return | Page |
| --- | --- | --- |
| PPO expert, for reference | 199.6 ± 0.1 | [LeWM](docs/lewm.md) |
| True dynamics | 193.7 ± 1.5 | [LeWM](docs/lewm.md) |
| LeWM, trained end to end (probe-decoded reward) | 187.5 ± 6.6 | [LeWM](docs/lewm.md) |
| V-JEPA 2 ViT-L patch tokens, 2x2 grid, goal objective (true dynamics: 193.1 ± 1.3; 1x1 to 8x8 grids: 183.8 to 190.3) | 190.3 ± 2.9 | [Patch tokens](docs/patch-token-wm.md) |
| V-JEPA 2 ViT-L pooled features, goal objective (true dynamics: 194.1 ± 1.2) | 183.8 ± 2.1 | [Latent MPC](docs/latent-mpc.md) |
| Linear model | 169.3 ± 6.5 | [LeWM](docs/lewm.md) |
| Random actions | 106.0 ± 2.0 | [LeWM](docs/lewm.md) |

Each row comes from its own collection and planner settings, one seed each, so read
them as separate experiments; [Results at a glance](docs/results.md) has the full
tables and what is missing. What they show:

- Near upright, learned models plan nearly as well as the true dynamics, and LeWM
  tracks the pose 5 to 7 times more accurately than a linear model from 5 to 50 steps.
- SIGReg is what makes LeWM usable: without it the encoder keeps the state, but
  rollouts are useless.
- How a goal is scored matters as much as the model. Planning on the Euclidean distance
  to an upright embedding is no better than random actions (98.6); measuring it through
  a linear read-out of the state recovers 163.2.
- **Limitation:** no learned model swings the pole up from a hanging start.

## Pipeline

```
expert  ->  data  ->  train  ->  evaluate  ->  control
 PPO        collect    probes     rollout       closed-loop
 policy     + embed    + world    error,        MPC with the
                       models     action        learned model
                                  response
```

## Quickstart

Needs Python 3.13, [uv](https://docs.astral.sh/uv/), a CUDA 13 GPU and Docker
([requirements](docs/getting-started.md)).

```bash
uv sync
./docker/up.sh && export MLFLOW_TRACKING_URI=http://127.0.0.1:5000

# Expert policy, then episodes rolled out from it (64x64 frames for LeWM)
uv run python scripts/cartpole/expert/train_ppo.py
uv run python scripts/cartpole/data/collect_obs.py \
  --num-episodes 500 --no-full-circle-reset \
  --deterministic-fraction 0.6 --stochastic-fraction 0.3 --source-hold 10 \
  --perturb-probability 0.05 --perturb-steps 10 30 --resolutions 64x64 256x256

# Train LeWM, then score it
uv run python scripts/cartpole/train/train_lewm.py \
  --collection-dir data/cartpole_trajectories/<collection_id> --iterations 30000
```

The [LeWM page](docs/lewm.md) continues from there with the probe, rollout evaluation,
action response and closed-loop control. Every script takes `--help`.

## Documentation

| Page | What it covers |
| --- | --- |
| [Getting started](docs/getting-started.md) | Requirements, MLflow, sharing a GPU, tests, package map |
| [Expert policy](docs/expert-policy.md) | The random agent and Brax PPO |
| [Data collection](docs/data-collection.md) | Episode format, perturbation recipe, DreamerV3 export |
| [Encoders and probes](docs/encoders-and-probes.md) | Frozen DINOv2 and V-JEPA 2, state probes |
| [Latent world model](docs/latent-world-model.md) | Predictor on pooled features |
| [Latent MPC](docs/latent-mpc.md) | Closed-loop CEM, action response, perturbation data |
| [V-JEPA 2-AC](docs/vjepa2-ac.md) | Post-training Meta's action-conditioned predictor |
| [LeWM](docs/lewm.md) | End-to-end JEPA, SIGReg, ablations, 224x224 frames |
| [Patch-token world model](docs/patch-token-wm.md) | DINO-WM-style predictor on token grids |
| [Linear baseline](docs/linear-baseline.md) | The baseline every model is compared with |
| [Results at a glance](docs/results.md) | All results side by side, and what is missing |
| [Roadmap](docs/roadmap.md) | The levels of working with world models, and where this repo stands |
| [References](docs/references.md) | Papers and code this builds on |

## Repository layout

```
src/world_model_playground/   the library: environment, data, encoders, models, planning
scripts/cartpole/             command-line entry points by stage (see its README)
tests/                        pytest suite
docs/                         the pages above
docker/                       MLflow server
```

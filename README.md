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

#### Latent MPC in closed loop

Control the real environment with the latent world model. Every step renders
the frames, encodes them with the frozen encoder, plans with CEM through the
predictor, executes the first action, and replans:

```bash
uv run python scripts/cartpole/world_model/latent_mpc.py \
  --embeddings-dir data/embeddings/<collection_id>/vjepa2-vitl_ctx2
uv run marimo edit scripts/cartpole/world_model/latent_mpc_viewer.py
```

The same episodes (same reset keys) run under every controller: `ppo`,
`random`, and CEM through the true MJX dynamics (`mpc-true`, the planner's
ceiling), the linear model (`mpc-linear`), and the latent model, scoring the
CartPole reward of probe-decoded latents (`mpc-latent`) or the distance to the
latent of an upright, centered, still pole (`mpc-latent-goal`). All planners
share the CEM settings (`--horizon 30`, 256 candidates, 4 iterations, knots
held for 2 steps) and optimize the environment's own dense reward
(`wm.planning.cartpole_reward`). Starts are near upright (`balance`) or at any
pole angle (`full-circle`). Trajectories, chosen plans, a summary, and figures
go to `artifacts/cartpole_world_model/mpc/<collection_id>/<embeddings>/`; the
viewer replays the real pole against each step's plan.

Planning asks a model a question its training metrics never checked: what
happens under actions other than the logged ones. `action_response.py`
applies constant +1 and -1 forces from test states in MuJoCo and in each model
and reports the gain, the share of the true response a model predicts:

```bash
uv run python scripts/cartpole/world_model/action_response.py \
  --embeddings-dir data/embeddings/<collection_id>/vjepa2-vitl_ctx2 --upright-only
```

Results over 10 episodes of 200 steps (2 s), V-JEPA 2 ViT-L features, return
± standard error (maximum 200):

| Controller | balance (100 ep. data) | balance (held-action data) | full-circle (100 ep.) | full-circle (held) |
| --- | --- | --- | --- | --- |
| PPO | 199.7 ± 0.1 | 199.7 ± 0.1 | 44.2 ± 6.2 | 44.2 ± 6.2 |
| CEM, true dynamics | 194.1 ± 1.2 | 194.1 ± 1.2 | 61.1 ± 8.9 | 66.7 ± 8.1 |
| CEM, latent model (reward) | 118.1 ± 1.4 | 141.7 ± 7.3 | 52.0 ± 7.9 | 40.3 ± 6.4 |
| CEM, latent model (goal latent) | 96.5 ± 10.6 | 127.3 ± 7.4 | 26.8 ± 9.7 | 21.1 ± 6.8 |
| CEM, linear model | 115.3 ± 3.2 | 121.4 ± 2.9 | 52.9 ± 4.5 | 42.2 ± 5.9 |
| Random | 106.0 ± 2.0 | 106.0 ± 2.0 | 23.5 ± 8.1 | 23.5 ± 8.1 |

With true dynamics the planner balances (194), so the gap is the models'. The
latent model reads the state well (probe pose RMSE 0.025 to 0.06 in closed
loop) but barely knows what actions do. Near upright, after 30 steps of
force, it predicts this share of the true response:

| Training data | latent: cart | latent: pole angle | linear: cart | linear: pole angle |
| --- | --- | --- | --- | --- |
| 100 episodes, sources mixed per step | 0.38 | -0.03 | 0.97 | 0.15 |
| 200 episodes, `--source-hold 10` | 0.66 | 0.16 | 0.90 | 0.09 |

Mixed per step, random forces never last long enough to show their effect, and
logged expert actions are predictable from the state, so a predictor can track
logged trajectories (Phase 3) while ignoring its actions. Holding each action
source for 10 steps (`collect_obs.py --source-hold 10`) raises the latent
model's response, and balance return from 118 to 142, but the pole-angle
response near upright stays far below the physics: random forces quickly
topple the pole, so the data still rarely shows sustained forces while
balancing. The linear model, fitted across every pole orientation, averages
away the action-to-angle coupling whose sign flips between upright and
hanging. On full-circle starts, no controller reliably swings up within 2 s
(PPO included): true-dynamics MPC does best, upright 33 to 41% of the second
half, and the latent model plans at the level of the linear model.

Planning through the latent model costs about 1.1 s per step for 10 episodes
(256 candidates x 30 steps x 4 iterations), against 0.04 s with MJX.

##### Perturbations around balance

To show the model sustained forces where balancing happens, collect episodes
that start near upright and, while the pole is upright (cos θ > 0.9), start a
perturbation with 5% chance per step: one uniform random force held for 10 to
30 steps over the scheduled actions, after which the expert recovers the pole:

```bash
uv run python scripts/cartpole/world_model/collect_obs.py \
  --num-episodes 200 --no-full-circle-reset \
  --deterministic-fraction 0.6 --stochastic-fraction 0.3 --source-hold 10 \
  --perturb-probability 0.05 --perturb-steps 10 30 --resolutions 256x256
```

Perturbations make up 26% of the steps and are recorded as a fourth action
source; 49% of states are near upright (cos θ > 0.9), against 15% with held
actions alone. Embedding, probe, and predictor then train as before. Near
upright, after 10 to 30 steps of force, the latent model now predicts most of
the true response:

| Training data | latent: cart | latent: pole angle | linear: cart | linear: pole angle |
| --- | --- | --- | --- | --- |
| 100 episodes, sources mixed per step | 0.38 | -0.03 | 0.97 | 0.15 |
| 200 episodes, `--source-hold 10` | 0.66 | 0.16 | 0.90 | 0.09 |
| 200 episodes, perturbations around balance | 0.92 | 0.81 | 0.86 | -0.10 |

and closed-loop balance return (10 episodes, same settings as above) reaches:

| Controller | held-action data | perturbation data |
| --- | --- | --- |
| CEM, latent model (goal latent) | 127.3 ± 7.4 | 183.8 ± 2.1 |
| CEM, latent model (reward) | 141.7 ± 7.3 | 175.5 ± 5.3 |
| CEM, linear model | 121.4 ± 2.9 | 174.9 ± 5.4 |
| CEM, true dynamics | 194.1 ± 1.2 | 194.1 ± 1.2 |

With the goal latent, the planner keeps the pole upright 97% of the second
half of each episode. Planning on the probe-decoded reward is held back by the
decoded angular velocity: those plans push twice as hard as true-dynamics MPC
and let the pole oscillate until it falls. Full-circle starts get worse (latent
33 to 39), since this collection rarely shows a hanging pole; mix start
distributions when swing-up matters.

Training the predictor with a 24-step instead of an 8-step open-loop loss
(`train_latent_wm.py --rollout-steps 24`) improves rollouts beyond 50 steps but
degrades the first 10 to 25, which the 30-step planner relies on, and lowers
balance return to 152.6 (reward) and 141.5 (goal latent).

#### V-JEPA 2-AC

Post-train Meta's action-conditioned predictor (V-JEPA 2-AC: ViT-g/16 encoder,
300M-parameter predictor pretrained on DROID robot video) on CartPole and plan
with it. It is released only as a `torch.hub` checkpoint
(`dl.fbaipublicfiles.com/vjepa2/vjepa2-ac-vitg.pt`, 11.8 GB, downloaded to the
torch hub cache on first use). The predictor code is vendored at a pinned
commit in `wm/vendor/vjepa2`; the encoder runs as the Hugging Face V-JEPA 2
architecture carrying the checkpoint's weights, which differ from the Hugging
Face ViT-g's (`wm.ac_checkpoint`).

As in its training, every frame is encoded on its own (repeated to fill a
tubelet) into 256 layer-normalized tokens. One model step spans
`--step-stride` control steps (default 5, i.e. 0.05 s, against 0.25 s on
DROID); their forces feed a new action layer initialized from the pretrained
one, and the state token is constant, so the model sees pixels and forces only:

```bash
# Tokens of the AC encoder (about 145 MB per 200-step episode)
uv run python scripts/cartpole/world_model/embed.py \
  --collection-dir data/cartpole_trajectories/<collection_id> \
  --encoder vjepa2-ac-vitg --store-tokens --batch-size 16
EMB=data/embeddings/<collection_id>/vjepa2-ac-vitg_ctx1
# A probe stacking steps one model step apart decodes predicted tokens
uv run python scripts/cartpole/world_model/train_probe.py \
  --embeddings-dir $EMB --history 2 --history-stride 5
# Zero-shot, post-trained, and from-scratch predictors
uv run python scripts/cartpole/world_model/train_ac_wm.py --embeddings-dir $EMB --iterations 0
uv run python scripts/cartpole/world_model/train_ac_wm.py --embeddings-dir $EMB
uv run python scripts/cartpole/world_model/train_ac_wm.py --embeddings-dir $EMB --init random
# Counterfactual response and closed-loop control, planning every model step
AC=artifacts/cartpole_world_model/ac_wm/<collection_id>/vjepa2-ac-vitg_ctx1/ac_pretrained_k5_h3_r2_i3000.pt
uv run python scripts/cartpole/world_model/action_response.py \
  --embeddings-dir $EMB --predictor $AC --upright-only
uv run python scripts/cartpole/world_model/latent_mpc.py \
  --embeddings-dir $EMB --predictor $AC --resets balance --episodes 5 \
  --candidates 64 --action-repeat 5 --replan-every 5
```

`train_ac_wm.py` trains with V-JEPA 2-AC's L1 teacher-forced and open-loop
losses in bfloat16 with activation checkpointing (`--freeze-blocks` keeps the
first blocks frozen if memory runs short), and reports decoded and pooled-latent
rollout RMSE at horizons that are multiples of the stride, against the linear
model, the probe on true features, and persistence. `latent_mpc.py` and
`action_response.py` load either latent model (`wm.world_models`); every
planner, the baselines included, then replans every `--replan-every` steps.
The goal objective scores the L1 distance of predicted tokens to those of the
upright frame, the energy V-JEPA 2-AC plans with.

#### LeWM: a JEPA trained end to end

Train the encoder and the predictor together from pixels, as in
[LeWM](https://arxiv.org/abs/2603.19312): the loss is the mean squared error
between predicted and actual next embeddings plus `--sigreg-weight` times
SIGReg (`wm.sigreg`), which pulls the embeddings of each step toward an isotropic
Gaussian. There is no stop-gradient, target network, or reconstruction. The
architecture and optimizer follow the released configuration (ViT-tiny encoder
with a batch-normalized projector, 6-block AdaLN-zero causal predictor, AdamW at
5e-5, bfloat16, weight 0.09, 1024 projections); what changes for CartPole is the
input, 64x64 frames with 8x8 patches instead of 224x224 with 14x14, and the
frame skip, 5 control steps per model step, with a history of 3.

```bash
# The perturbation recipe above, with 64x64 frames
uv run python scripts/cartpole/world_model/collect_obs.py \
  --num-episodes 500 --no-full-circle-reset \
  --deterministic-fraction 0.6 --stochastic-fraction 0.3 --source-hold 10 \
  --perturb-probability 0.05 --perturb-steps 10 30 --resolutions 64x64 256x256
uv run python scripts/cartpole/world_model/train_lewm.py \
  --collection-dir data/cartpole_trajectories/<collection_id> --iterations 30000
# The trained encoder is an embed.py encoder, named by its artifact
ENC=lewm:$PWD/artifacts/cartpole_world_model/lewm/<collection_id>/<name>.pt
uv run python scripts/cartpole/world_model/embed.py \
  --collection-dir data/cartpole_trajectories/<collection_id> --encoder $ENC
EMB=data/embeddings/<collection_id>/lewm-<name>_ctx1
uv run python scripts/cartpole/world_model/train_probe.py \
  --embeddings-dir $EMB --history 2 --history-stride 5
uv run python scripts/cartpole/world_model/eval_lewm.py --embeddings-dir $EMB
uv run python scripts/cartpole/world_model/action_response.py \
  --embeddings-dir $EMB --predictor <artifact>.pt --upright-only
uv run python scripts/cartpole/world_model/latent_mpc.py \
  --embeddings-dir $EMB --predictor <artifact>.pt --resets balance full-circle \
  --action-repeat 5 --replan-every 5
```

`train_lewm.py` logs the losses and collapse monitors (embedding spread,
effective rank, a ridge read-out of the state) every `--eval-every` iterations,
and saves the model, its record, and a copy in MLflow when `MLFLOW_TRACKING_URI`
is set. `eval_lewm.py` records the probe in the model, which is how
`action_response.py` and `latent_mpc.py` find it. `--gradient-checkpointing`
cuts the peak memory when the GPU is shared.

Results on a 500-episode collection of the perturbation recipe (100,000
transitions, 64x64 frames; 80/10/10 episode split), for the model trained for
30,000 iterations at batch 128 (about 52 passes over 74,400 windows). It has
17.9M parameters (6.2M encoder and projector, 11.7M predictor) and trained in
116 minutes on a shared RTX 3070. Without gradient checkpointing, batch 128 peaks
at 3.7 GB, so it fits an 8 GB card; with it, 0.9 GB. Test R² of state probes on
its embeddings, stacking 2 steps 5 control steps apart:

| Probe | cart pos | cos θ | sin θ | cart vel | ang vel |
| --- | --- | --- | --- | --- | --- |
| linear | 0.977 | 0.995 | 0.985 | 0.727 | 0.860 |
| MLP | 1.000 | 1.000 | 1.000 | 0.974 | 0.996 |

Open-loop test rollout RMSE of the probe-decoded state from true context and true
actions, every 20 steps of each test episode:

| Horizon | Pose: LeWM | linear | persistence | probe on true embeddings | Velocity: LeWM | linear | persistence | probe on true embeddings |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5 | 0.016 | 0.107 | 0.123 | 0.008 | 0.347 | 0.186 | 0.431 | 0.198 |
| 10 | 0.030 | 0.212 | 0.243 | 0.010 | 0.401 | 0.342 | 0.830 | 0.221 |
| 25 | 0.070 | 0.499 | 0.552 | 0.008 | 0.505 | 0.852 | 1.790 | 0.208 |
| 50 | 0.142 | 0.725 | 0.817 | 0.011 | 0.721 | 2.446 | 2.946 | 0.243 |
| 100 | 0.283 | 0.708 | 0.833 | 0.010 | 1.237 | 3.186 | 3.908 | 0.261 |

LeWM tracks the pose 5 to 7 times more accurately than the linear model from 5
to 50 steps. The linear model predicts velocity better over the first 5 to 10
steps (0.186 against 0.347 at 5), and LeWM beats it from 25 steps on. The model
is trained on one-step predictions only, and beyond about 100 steps its pose
error reaches that of persistence (0.611 against 0.624 at 200 steps). The
response to sustained forces, measured by `action_response.py` as the share of
the true response to +1 against -1 force from 210 near-upright test states that
each model predicts (1 is exact):

| Horizon | LeWM: cart | LeWM: pole angle | linear: cart | linear: pole angle |
| --- | --- | --- | --- | --- |
| 5 | 1.20 | 1.26 | 1.01 | 0.23 |
| 10 | 1.09 | 1.08 | 1.02 | 0.34 |
| 20 | 1.05 | 1.04 | 1.02 | 0.39 |
| 30 | 0.98 | 0.99 | 1.03 | 0.41 |

Closed-loop control over 10 episodes of 200 steps (return ± standard error,
maximum 200), all controllers on the same reset keys and CEM settings as
above (horizon 30, 256 candidates, 4 iterations, a plan every 5 steps):

| Controller | balance | full-circle | planning s/step (10 episodes) |
| --- | --- | --- | --- |
| PPO | 199.6 ± 0.1 | 96.7 ± 8.8 | |
| CEM, true dynamics | 193.7 ± 1.5 | 61.4 ± 7.9 | 0.018 |
| CEM, LeWM (probe-decoded reward) | 187.5 ± 6.6 | 36.1 ± 9.2 | 0.208 |
| CEM, linear model | 169.3 ± 6.5 | 41.6 ± 6.6 | 0.006 |
| CEM, LeWM (goal embedding) | 98.6 ± 10.5 | 27.4 ± 6.6 | 0.216 |
| CEM, LeWM (goal embedding, read-out metric, summed) | 142.9 ± 9.9 | 28.0 ± 8.4 | 0.226 |
| CEM, LeWM (goal embedding, read-out metric, last step) | 163.2 ± 9.3 | 21.4 ± 7.5 | 0.244 |
| Random | 106.0 ± 2.0 | 23.5 ± 8.1 | |

What the experiments show:

- Planning on the reward of probe-decoded embeddings works near upright (187.5
  against 193.7 with the true dynamics) but, as with the other latent models,
  does not swing the pole up from hanging starts.
- Planning on the Euclidean distance to the embedding of an upright frame, which is
  how LeWM plans, does not work here: 98.6, no better than random, with the
  distance summed over the horizon (default) or at its last step only
  (`latent_mpc.py --goal-cost final`, 100.2 ± 3.3 on the 10,000-iteration
  model, against 98.0 ± 10.0). The embeddings carry the state; the Euclidean
  distance does not use it well. On the 10,000-iteration model, across all test
  steps the distance to the goal embedding tracks the state cost (Spearman 0.81),
  but among near-upright steps, where balancing happens, it does not (0.10),
  while the distance after a linear read-out of the state does (0.97 over all
  steps). SIGReg spreads variance over many directions (the top 5 principal
  directions hold 16% of it), and the few that carry the state are drowned out
  in a Euclidean distance.
- Measuring the same goal distance through a read-out of the state fixes most
  of it. `latent_mpc.py --goal-metric readout` fits a ridge regression from the
  training embeddings to the standardized state and scores the squared distance
  after projecting the latent difference through it
  (`wm.planning.fit_readout_metric`). On the 30,000-iteration model it ranks
  which of two near-upright steps is closer to upright correctly 87% of the time
  (Euclidean: 49%, chance), and planning with it reaches 142.9 ± 9.9 on balance
  summed over the horizon and 163.2 ± 9.3 with the last step only, the cost LeWM
  uses, against 98.6 for the Euclidean distance and 106.0 for random actions.
  That is on par with the linear model (169.3) but still short of the
  probe-decoded reward (187.5), and it does not help from hanging starts.
- SIGReg is what makes the embeddings usable for prediction. With
  `--sigreg-weight 0` the encoder still keeps the state, in a tiny
  spread: a probe on its embeddings passes the gate (MLP pose R² 0.996, 0.998, 0.994),
  and the prediction loss falls to 0.0002 simply because the scale shrinks. But
  rollouts are useless (pose RMSE 4.9 at 5 steps, against 0.12 for persistence),
  because small errors in that geometry decode to large ones. The in-training
  monitors run in float32 for this reason: in bfloat16 the shrunken spread
  rounds away and reads as a loss of the state.
- The history matters more than the regularizer weight. Ablations of 10,000
  iterations each (one seed, so differences of a few percent are noise), test
  pose and velocity rollout RMSE:

| Variant | pose h25 | pose h50 | pose h100 | velocity h25 | MLP probe cart vel / ang vel |
| --- | --- | --- | --- | --- | --- |
| history 1 | 0.207 | 0.366 | 0.606 | 1.437 | 0.964 / 0.993 |
| history 2 | 0.110 | 0.211 | 0.398 | 0.774 | 0.973 / 0.995 |
| history 3, weight 0.09 (default) | 0.084 | 0.161 | 0.334 | 0.547 | 0.967 / 0.995 |
| weight 0.02 | 0.081 | 0.159 | 0.304 | 0.548 | 0.973 / 0.994 |
| weight 0.3 | 0.091 | 0.164 | 0.345 | 0.609 | 0.962 / 0.992 |
| weight 0 | 9.417 | 9.468 | 9.525 | 22.344 | 0.937 / 0.985 |

  A single frame carries no velocity, so a history of 1 cannot predict it.
  Weights from 0.02 to 0.3 behave alike; the paper reports the same
  insensitivity on Push-T for 0.01 to 0.2.

##### Frames of 224x224

LeWM trains on 224x224 frames with 14x14 patches. To match what the controller
renders, the same 500 episodes were collected again at that size (same seed and
policy: all 500 episodes have identical states, actions, and action sources, so
the split is the same) and the model was trained for 10,000 iterations, to compare
with the 10,000-iteration model on 64x64 frames:

```bash
uv run python scripts/cartpole/world_model/collect_obs.py \
  --num-episodes 500 --no-full-circle-reset \
  --deterministic-fraction 0.6 --stochastic-fraction 0.3 --source-hold 10 \
  --perturb-probability 0.05 --perturb-steps 10 30 --resolutions 224x224
uv run python scripts/cartpole/world_model/train_lewm.py \
  --collection-dir data/cartpole_trajectories/<collection_id> \
  --resolution 224 --patch-size 14 --iterations 10000 --gradient-checkpointing
```

The frames of 400 training episodes take 12 GB of memory. One seed each:

| | 64x64 | 224x224 |
| --- | --- | --- |
| training time, 10,000 iterations | 1,236 s | 6,095 s |
| peak GPU memory, batch 128 | 3.7 GB | 3.0 GB with checkpointing (not measured without) |
| MLP probe R², cart vel / ang vel | 0.967 / 0.995 | 0.975 / 0.997 |
| pose RMSE at 5 / 25 / 100 steps | 0.022 / 0.084 / 0.334 | 0.021 / 0.077 / 0.295 |
| velocity RMSE at 5 / 25 / 100 steps | 0.357 / 0.547 / 1.334 | 0.380 / 0.480 / 1.162 |
| action response at 30 steps, cart / pole angle | 1.03 / 1.00 | 1.02 / 0.97 |
| balance return, probe-decoded reward | 183.4 ± 5.1 | 190.8 ± 4.4 |
| balance return, Euclidean goal | 98.0 ± 10.0 | 116.2 ± 8.6 |
| balance return, read-out goal, last step | 163.2 ± 9.3 (30,000 iterations) | 159.2 ± 10.9 |
| full-circle return, probe-decoded reward | 37.4 ± 8.4 | 27.0 ± 5.7 |
| planning s/step (10 episodes) | 0.208 | 0.229 |

At 5 times the training cost, larger frames improve the rollouts by 7 to 13% from
25 steps on and leave the closed-loop results within their standard errors, so
on CartPole 64x64 frames are the better trade (with one seed each, the rollout
gain itself is also near the noise). The Level 1 results above come from
other collections, so they are context and not a controlled comparison; running
`embed.py --encoder vjepa2-vitl` and `train_latent_wm.py` on this collection gives
one.

#### Patch-token world model (DINO-WM style)

The latent world model above predicts one pooled feature per step. This one
predicts the whole grid of patch tokens of a frozen encoder, as DINO-WM does: a
block-causal transformer (about 11M parameters) reads the token grids and
grouped actions of the last 3 model steps and predicts the next grid, with
teacher-forced and open-loop losses. One model step spans 5 control steps, and
tokens can be average-pooled to a coarser grid (`--pool-side`), which sets the
cost: attention grows with the square of the tokens per sequence.

```bash
E=data/embeddings/perturb200/vjepa2-vitl_ctx2
uv run python scripts/cartpole/world_model/embed.py \
  --collection-dir data/cartpole_trajectories/perturb200 --encoder vjepa2-vitl --store-tokens
# One flat float16 memmap, so training can sample windows across episodes
uv run python scripts/cartpole/world_model/consolidate_tokens.py --embeddings-dir $E
# The probe decodes the grid pooled to --summary-side, stacking steps 5 apart
uv run python scripts/cartpole/world_model/train_patch_probe.py \
  --embeddings-dir $E --summary-side 2
uv run python scripts/cartpole/world_model/train_patch_wm.py \
  --embeddings-dir $E --pool-side 8 --summary-side 2 --epochs 30 --steps-per-epoch 300
uv run python scripts/cartpole/world_model/eval_patch_wm.py \
  --embeddings-dir $E --predictor artifacts/cartpole_world_model/patch_wm/<collection_id>/<embeddings>/patch_p8_k5_h3_r4.pt
```

`latent_mpc.py` and `action_response.py` load it through `wm.world_models`.
For models with many tokens, `latent_mpc.py --rollout-chunk 128` rolls the
candidates out in chunks so that 10 episodes x 64 candidates fit an 8 GB card
(V-JEPA 2 tokens have 1024 channels). Closed-loop control renders the collected
frame size and lets the encoder resize it, as `embed.py` does; rendering
DINOv2's 224 pixels directly shifts its features by about 15% and a probe's pose
RMSE on the same states from 0.017 to 0.094.

Trained on 200 episodes with perturbations around balance (the data recipe of
the perturbation section above), planning with the same CEM settings for every
model, except that the planner replans every 5 steps with 64 candidates because
a model step spans 5 control steps. Single seed, 10 episodes, return +- standard
error (maximum 200), starts near upright; "gain" is the share of the true
response to 30 steps of constant force that the model predicts near upright
(cart position, pole angle):

| Encoder | Grid | Peak GB | Epoch s | Plan s | Pose RMSE, 25 steps | Gain, 30 steps | Return, reward | Return, goal |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| V-JEPA 2 ViT-L | 1x1 (mean token) | 0.46 | 13 | 0.07 | 0.113 | 0.24, 0.30 | 172.4 +- 6.2 | 186.8 +- 4.3 |
| V-JEPA 2 ViT-L | 2x2 | 0.85 | 14 | 0.07 | 0.115 | 0.39, 0.34 | 175.0 +- 4.7 | 190.3 +- 2.9 |
| V-JEPA 2 ViT-L | 4x4 | 2.39 | 14 | 0.19 | 0.114 | 0.53, 0.45 | 172.1 +- 5.0 | 189.3 +- 1.8 |
| V-JEPA 2 ViT-L | 8x8 | 4.36 | 45 | 0.69 | 0.110 | 0.78, 0.61 | 179.7 +- 4.7 | 183.8 +- 4.1 |
| DINOv2-S | 1x1 (mean token) | 0.39 | 13 | 0.07 | 0.214 | | 169.5 +- 7.1 | 167.3 +- 6.9 |
| DINOv2-S | 4x4 | 1.51 | 13 | 0.17 | 0.173 | | 166.0 +- 8.2 | 149.8 +- 10.7 |
| DINOv2-S | 8x8 | 5.04 | 100 | 0.49 | 0.193 | | 170.8 +- 7.0 | 157.2 +- 8.0 |
| Linear model | | | | 0.006 | 0.472 | 1.03, 0.41 | 170.8 +- 5.2 | |
| True dynamics | | | | 0.019 | | | 193.1 +- 1.3 | |

Plan s is one CEM plan for 10 episodes. Epoch s is 300 optimizer steps of batch 32;
the DINOv2 8x8 epoch shared the GPU with another job. The DINOv2-S 2x2 grid and
the per-horizon errors are in the `cartpole-wm` MLflow experiment.

- Every patch model tracks the pose 1.6 to 4.3 times better than the linear model
  over 25 to 100 steps, and V-JEPA 2 beats DINOv2-S at every horizon. That
  comparison mixes encoder size (ViT-L against ViT-S) with the 2-frame context
  that lets V-JEPA 2 see motion.
- Spatial tokens do not help closed-loop balancing here: every V-JEPA 2 grid,
  including the single mean token, returns 172 to 180 (reward) and 184 to 190
  (goal), within the standard errors. The 8x8 model costs 10 times the memory and
  10 times the planning time of the 1x1 model for no return gain.
- They do change what the model learned about actions: the response gain
  climbs steadily with the number of tokens, from 0.24 (cart) and 0.30 (pole)
  to 0.78 and 0.61. The 5-step replanning masks this at balance. It is also
  below the 0.92 and 0.81 of the stride-1 pooled model in the perturbation
  section, so the table does not show patch tokens beating pooled features in
  general: stride, history, and training length differ.
- The 1x1 model is the mean patch token, not DINO-WM's CLS comparison: the
  CLS-against-patch claim itself is untested here.
- Full-circle starts stay near random (24 to 33, DINOv2-S and V-JEPA 2 8x8). The
  data shows few hanging poses, and the V-JEPA 2 probe reads them badly (pose
  RMSE 0.46 to 0.53 in closed loop, against 0.12 to 0.14 for DINOv2-S), so
  swing-up is out of reach of these models.

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

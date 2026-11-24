# LeWM: a JEPA trained end to end

Train the encoder and predictor together from pixels with a prediction loss and the SIGReg regularizer, then compare it with frozen-encoder models. Covers ablations of the regularizer and history, why planning on goal-embedding distance fails, and the 224x224 comparison.

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
# The perturbation recipe of data-collection.md, with 64x64 frames
uv run python scripts/cartpole/data/collect_obs.py \
  --num-episodes 500 --no-full-circle-reset \
  --deterministic-fraction 0.6 --stochastic-fraction 0.3 --source-hold 10 \
  --perturb-probability 0.05 --perturb-steps 10 30 --resolutions 64x64 256x256
uv run python scripts/cartpole/train/train_lewm.py \
  --collection-dir data/cartpole_trajectories/<collection_id> --iterations 30000
# The trained encoder is an embed.py encoder, named by its artifact
ENC=lewm:$PWD/artifacts/cartpole_world_model/lewm/<collection_id>/<name>.pt
uv run python scripts/cartpole/data/embed.py \
  --collection-dir data/cartpole_trajectories/<collection_id> --encoder $ENC
EMB=data/embeddings/<collection_id>/lewm-<name>_ctx1
uv run python scripts/cartpole/train/train_probe.py \
  --embeddings-dir $EMB --history 2 --history-stride 5
uv run python scripts/cartpole/evaluate/eval_lewm.py --embeddings-dir $EMB
uv run python scripts/cartpole/evaluate/action_response.py \
  --embeddings-dir $EMB --predictor <artifact>.pt --upright-only
uv run python scripts/cartpole/control/latent_mpc.py \
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
maximum 200), all controllers on the same reset keys and CEM settings as in
[latent MPC](latent-mpc.md) (horizon 30, 256 candidates, 4 iterations, a plan every 5 steps):

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

## Frames of 224x224

LeWM trains on 224x224 frames with 14x14 patches. To match what the controller
renders, the same 500 episodes were collected again at that size (same seed and
policy: all 500 episodes have identical states, actions, and action sources, so
the split is the same) and the model was trained for 10,000 iterations, to compare
with the 10,000-iteration model on 64x64 frames:

```bash
uv run python scripts/cartpole/data/collect_obs.py \
  --num-episodes 500 --no-full-circle-reset \
  --deterministic-fraction 0.6 --stochastic-fraction 0.3 --source-hold 10 \
  --perturb-probability 0.05 --perturb-steps 10 30 --resolutions 224x224
uv run python scripts/cartpole/train/train_lewm.py \
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
gain itself is also near the noise). The Level 1 results ([latent world model](latent-world-model.md),
[latent MPC](latent-mpc.md)) come from other collections, so they are context and not a controlled comparison; running
`embed.py --encoder vjepa2-vitl` and `train_latent_wm.py` on this collection gives
one.

---

Previous: [V-JEPA 2-AC](vjepa2-ac.md) · Next: [Patch-token world model](patch-token-wm.md) · [README](../README.md)

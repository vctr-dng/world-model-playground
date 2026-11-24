# Results at a glance

The numbers from the other pages, side by side. **They are not a controlled
comparison:** the models were trained on different collections, with different
planner settings, one seed each, and returns are over 10 episodes of 200 steps
(maximum 200, ± standard error). Each table names its collection and links to
the page that explains it. The unified experiment that would make the models
directly comparable is listed under [what is missing](#what-is-missing).

## Closed-loop control: LeWM collection

500 episodes of the perturbation recipe, 64x64 frames. All controllers share the
reset keys and CEM settings (horizon 30, 256 candidates, 4 iterations, a plan every
5 steps). Source: [LeWM](lewm.md).

| Controller | Balance | Full-circle | Planning s/step (10 episodes) |
| --- | --- | --- | --- |
| PPO | 199.6 ± 0.1 | 96.7 ± 8.8 | |
| CEM, true dynamics | 193.7 ± 1.5 | 61.4 ± 7.9 | 0.018 |
| CEM, LeWM, probe-decoded reward | 187.5 ± 6.6 | 36.1 ± 9.2 | 0.208 |
| CEM, linear model | 169.3 ± 6.5 | 41.6 ± 6.6 | 0.006 |
| CEM, LeWM, goal embedding, read-out metric, last step | 163.2 ± 9.3 | 21.4 ± 7.5 | 0.244 |
| CEM, LeWM, goal embedding, Euclidean | 98.6 ± 10.5 | 27.4 ± 6.6 | 0.216 |
| Random | 106.0 ± 2.0 | 23.5 ± 8.1 | |

## Closed-loop control: patch-token collection

200 episodes of the perturbation recipe, frozen encoders, a plan every 5 steps with
64 candidates. Source: [Patch-token world model](patch-token-wm.md).

| Encoder | Grid | Peak GB | Plan s | Pose RMSE, 25 steps | Balance, reward | Balance, goal |
| --- | --- | --- | --- | --- | --- | --- |
| V-JEPA 2 ViT-L | 1x1 (mean token) | 0.46 | 0.07 | 0.113 | 172.4 ± 6.2 | 186.8 ± 4.3 |
| V-JEPA 2 ViT-L | 2x2 | 0.85 | 0.07 | 0.115 | 175.0 ± 4.7 | 190.3 ± 2.9 |
| V-JEPA 2 ViT-L | 4x4 | 2.39 | 0.19 | 0.114 | 172.1 ± 5.0 | 189.3 ± 1.8 |
| V-JEPA 2 ViT-L | 8x8 | 4.36 | 0.69 | 0.110 | 179.7 ± 4.7 | 183.8 ± 4.1 |
| DINOv2-S | 1x1 (mean token) | 0.39 | 0.07 | 0.214 | 169.5 ± 7.1 | 167.3 ± 6.9 |
| DINOv2-S | 4x4 | 1.51 | 0.17 | 0.173 | 166.0 ± 8.2 | 149.8 ± 10.7 |
| DINOv2-S | 8x8 | 5.04 | 0.49 | 0.193 | 170.8 ± 7.0 | 157.2 ± 8.0 |
| Linear model | | | 0.006 | 0.472 | 170.8 ± 5.2 | |
| True dynamics | | | 0.019 | | 193.1 ± 1.3 | |

## Closed-loop control: pooled features

V-JEPA 2 ViT-L pooled features, 200 episodes of the perturbation recipe, the CEM
settings of the other latent MPC runs (horizon 30, 256 candidates, 4 iterations).
Source: [Latent MPC](latent-mpc.md#perturbations-around-balance).

| Controller | Balance |
| --- | --- |
| CEM, true dynamics | 194.1 ± 1.2 |
| CEM, latent model, goal latent | 183.8 ± 2.1 |
| CEM, latent model, reward | 175.5 ± 5.3 |
| CEM, linear model | 174.9 ± 5.4 |

## Open-loop accuracy of the decoded pose

Root-mean-square error of the pose (cart position, cos θ, sin θ) decoded from the
model's predicted latents, from true context and true actions.

| Model | Collection | 25 steps | 50 steps | Linear model at the same horizons | Source |
| --- | --- | --- | --- | --- | --- |
| Pooled latent, V-JEPA 2 ViT-L | 100 episodes, mixed sources | 0.144 | 0.351 | 0.544, 0.812 | [Latent world model](latent-world-model.md) |
| Patch tokens, V-JEPA 2 ViT-L | 200 episodes, perturbations | 0.110–0.115 | | 0.472 | [Patch-token world model](patch-token-wm.md) |
| LeWM, 30,000 iterations | 500 episodes, perturbations | 0.070 | 0.142 | 0.499, 0.725 | [LeWM](lewm.md) |

Collections and evaluation settings differ between pages, so compare each model
with the linear model on its own row rather than across rows.

## Training cost

| Model | Parameters | Training | Peak GPU memory |
| --- | --- | --- | --- |
| LeWM, 64x64, 30,000 iterations, batch 128 | 17.9M (6.2M encoder, 11.7M predictor) | 116 min on a shared RTX 3070 | 3.7 GB, or 0.9 GB with gradient checkpointing |
| LeWM, 64x64, 10,000 iterations | | 1,236 s | 3.7 GB |
| LeWM, 224x224, 10,000 iterations | | 6,095 s | 3.0 GB with gradient checkpointing |
| Patch-token predictor, encoder frozen | about 11M | 13–100 s per 300 steps of batch 32 (the 100 s run shared the GPU) | 0.39–5.04 GB, by grid size |

## What the tables show

- **Near upright, learned models come close to true dynamics** once the data shows
  sustained forces: 187.5 for LeWM against 193.7, and up to 190.3 for a V-JEPA 2
  patch model against 193.1. A linear model reaches 169–175 on the same task.
- **No learned model swings the pole up from hanging starts.** Learned
  full-circle returns stay at 52 or below (36.1 for LeWM), against 61 to 67 for MPC
  on the true dynamics; random actions score 23.5. The collections rarely show a
  hanging pole.
- **How the goal is scored matters as much as the model.** LeWM planning on the
  Euclidean distance to an upright embedding scores 98.6, no better than random;
  measuring the same distance through a linear read-out of the state gives 163.2
  ([LeWM](lewm.md)).
- **Spatial tokens did not improve balancing.** Grids from 1x1 to 8x8 return within
  their standard errors, while memory grows 10x and planning time 10x
  ([Patch-token world model](patch-token-wm.md)).
- **SIGReg is what makes LeWM predictable.** Trained without it, the encoder still
  keeps the state but rollouts are unusable (pose RMSE 4.9 at 5 steps against 0.12
  for persistence).

## What is missing

- **A controlled comparison.** The models share no collection, planner setting, or
  seed count. Running pooled-feature, patch-token and LeWM models on one collection
  with the same CEM settings, and several seeds, would turn these tables into
  conclusions.
- **V-JEPA 2-AC results.** [The page](vjepa2-ac.md) documents the commands for
  zero-shot, post-trained and from-scratch predictors, but no results are recorded.
- **A CLS-token comparison.** The 1x1 patch-token model uses the mean patch token,
  so DINO-WM's claim that patch features beat the CLS token is untested here.

---

[README](../README.md) · [Roadmap](roadmap.md)

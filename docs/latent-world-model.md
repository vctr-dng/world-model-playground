# Latent world model

An action-conditioned predictor trained on the cached pooled features of a frozen encoder, scored by open-loop rollout error against a linear baseline.

Train an action-conditioned predictor on the cached features of an encoder. A
causal transformer reads the features and actions of the last `--history`
steps and predicts the next feature, with the encoder frozen. Training adds a
teacher-forced one-step loss to an open-loop loss over `--rollout-steps`
predictions fed back as inputs:

```bash
uv run python scripts/cartpole/train/train_latent_wm.py \
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

---

Previous: [Frozen encoders and state probes](encoders-and-probes.md) · Next: [Latent MPC in closed loop](latent-mpc.md) · [README](../README.md)

# Patch-token world model (DINO-WM style)

Predict the whole grid of a frozen encoder's patch tokens instead of one pooled feature, and measure what the spatial detail buys in rollout error, response to actions, and closed-loop return.

The [latent world model](latent-world-model.md) predicts one pooled feature per step. This one
predicts the whole grid of patch tokens of a frozen encoder, as DINO-WM does: a
block-causal transformer (about 11M parameters) reads the token grids and
grouped actions of the last 3 model steps and predicts the next grid, with
teacher-forced and open-loop losses. One model step spans 5 control steps, and
tokens can be average-pooled to a coarser grid (`--pool-side`), which sets the
cost: attention grows with the square of the tokens per sequence.

```bash
E=data/embeddings/perturb200/vjepa2-vitl_ctx2
uv run python scripts/cartpole/data/embed.py \
  --collection-dir data/cartpole_trajectories/perturb200 --encoder vjepa2-vitl --store-tokens
# One flat float16 memmap, so training can sample windows across episodes
uv run python scripts/cartpole/data/consolidate_tokens.py --embeddings-dir $E
# The probe decodes the grid pooled to --summary-side, stacking steps 5 apart
uv run python scripts/cartpole/train/train_patch_probe.py \
  --embeddings-dir $E --summary-side 2
uv run python scripts/cartpole/train/train_patch_wm.py \
  --embeddings-dir $E --pool-side 8 --summary-side 2 --epochs 30 --steps-per-epoch 300
uv run python scripts/cartpole/evaluate/eval_patch_wm.py \
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
[perturbations around balance](data-collection.md#perturbations-around-balance)), planning with the same CEM settings for every
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
  below the 0.92 and 0.81 of the stride-1 pooled model in the
  [perturbation results](latent-mpc.md#perturbations-around-balance), so the table does not show patch tokens beating pooled features in
  general: stride, history, and training length differ.
- The 1x1 model is the mean patch token, not DINO-WM's CLS comparison: the
  CLS-against-patch claim itself is untested here.
- Full-circle starts stay near random (24 to 33, DINOv2-S and V-JEPA 2 8x8). The
  data shows few hanging poses, and the V-JEPA 2 probe reads them badly (pose
  RMSE 0.46 to 0.53 in closed loop, against 0.12 to 0.14 for DINOv2-S), so
  swing-up is out of reach of these models.

---

Previous: [LeWM: a JEPA trained end to end](lewm.md) · Next: [Linear baseline](linear-baseline.md) · [README](../README.md)

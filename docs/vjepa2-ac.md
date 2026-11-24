# V-JEPA 2-AC

Post-train Meta's released action-conditioned predictor (pretrained on DROID robot video) on CartPole, and plan with it.

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
uv run python scripts/cartpole/data/embed.py \
  --collection-dir data/cartpole_trajectories/<collection_id> \
  --encoder vjepa2-ac-vitg --store-tokens --batch-size 16
EMB=data/embeddings/<collection_id>/vjepa2-ac-vitg_ctx1
# A probe stacking steps one model step apart decodes predicted tokens
uv run python scripts/cartpole/train/train_probe.py \
  --embeddings-dir $EMB --history 2 --history-stride 5
# Zero-shot, post-trained, and from-scratch predictors
uv run python scripts/cartpole/train/train_ac_wm.py --embeddings-dir $EMB --iterations 0
uv run python scripts/cartpole/train/train_ac_wm.py --embeddings-dir $EMB
uv run python scripts/cartpole/train/train_ac_wm.py --embeddings-dir $EMB --init random
# Counterfactual response and closed-loop control, planning every model step
AC=artifacts/cartpole_world_model/ac_wm/<collection_id>/vjepa2-ac-vitg_ctx1/ac_pretrained_k5_h3_r2_i3000.pt
uv run python scripts/cartpole/evaluate/action_response.py \
  --embeddings-dir $EMB --predictor $AC --upright-only
uv run python scripts/cartpole/control/latent_mpc.py \
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

---

Previous: [Latent MPC in closed loop](latent-mpc.md) · Next: [LeWM: a JEPA trained end to end](lewm.md) · [README](../README.md)

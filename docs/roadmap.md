# Levels of Working with World Models

Ordered by effort. Each level builds on the previous one.

**Hardware assumed:** 2× RTX 3070 (8 GB each), Threadripper 3960X (48 threads), 62 GB RAM, ~206 GB free disk.
Time estimates are rough guesses, not measurements.

## Status in this repo

This repository implements the levels on **CartPole** (MJX `CartpoleBalance`) with a
PPO expert, not on Push-T, so the Push-T success rates quoted below are reference
points from the papers and do not compare with the results here.

| Level | Status | Where |
|---|---|---|
| 0 | Partly, in a different form. Meta's released V-JEPA 2-AC predictor is loaded and adapted to CartPole (`--iterations 0` evaluates it without post-training); the `lewm-pusht` checkpoint is not used. | [V-JEPA 2-AC](vjepa2-ac.md) |
| 1 | Done. Frozen DINOv2-S and V-JEPA 2 ViT-L encoders with a pooled-feature predictor and a patch-token predictor (grids from 1x1 to 8x8), planned with CEM. | [Latent world model](latent-world-model.md), [Latent MPC](latent-mpc.md), [Patch-token world model](patch-token-wm.md) |
| 2 | Done. LeWM trained end to end with SIGReg, with regularizer and history ablations and a 224x224 comparison. | [LeWM](lewm.md) |
| 3 | Partial. Many comparisons exist, but on different collections and planner settings, with one seed each. The patch-token model has a DINOv2-S against V-JEPA 2 ViT-L comparison, which mixes encoder size with the 2-frame context. | [Results at a glance](results.md) |
| 4 | Not started. | [References](references.md) |
| 5 | Out of scope. | |

## Summary

| Level | What you do | Time (rough) |
|---|---|---|
| 0 | Run a pretrained world model as a black box | 1–2 evenings |
| 1 | Frozen pretrained encoder + trained predictor (DINO-WM style) | Days to a week |
| 2 | Train a JEPA end-to-end from pixels (LeWM style) | About a week |
| 3 | Compare Levels 1 and 2, add an encoder ablation | 1–2 weeks on top |
| 4 | Add your own research contribution | 2–4 weeks, risk of negative result |
| 5 | Other families (generative, Dreamer-style, large-scale pretraining) | n/a |

## Level 0: Use a pretrained world model as a black box

- **What:** Load a released checkpoint (e.g. `quentinll/lewm-pusht`) and run planning in an environment.
- **Skills shown:** Setup, evaluation, reading the literature.

## Level 1: Frozen pretrained encoder, train only the dynamics (DINO-WM style)

- **What:** Freeze DINOv2 or V-JEPA 2. Train a small action-conditioned predictor on its features, then plan with CEM/MPC toward a goal image.
- **Skills shown:** Working with foundation models, latent-space dynamics, planning.
- **Cost:** Cheap in trainable parameters (DINO-WM's predictor is about 19M). Patch features are too large to cache for the full Push-T set (my estimate is roughly 500 GB), so run the frozen encoder on the fly or cache a subset.
- **Note:** Use patch tokens, not CLS. In the DINO-WM paper, patch features reach 96% on Wall against 58% for CLS.

## Level 2: Train a JEPA end-to-end from pixels (LeWM style)

- **What:** Train the encoder and predictor together, using a prediction loss plus the SIGReg regularizer to prevent collapse.
- **Skills shown:** Self-supervised learning, representation collapse, regularization.
- **Cost:** About 15M parameters (ViT-tiny encoder about 5M, predictor about 10M). The paper reports a few hours on one GPU, an L40S with 48 GB. I haven't tested whether it fits in 8 GB; a smaller batch or gradient accumulation may be needed.

## Level 3: Compare and analyze

- **What:** Put Levels 1 and 2 side by side on accuracy, planning time, VRAM, training time and failure cases. Add V-JEPA 2 ViT-L as a third encoder (patch tokens) and run an ablation.
- **Skills shown:** Experimental design, ablations, honest reporting of trade-offs.

## Level 4: Add a research contribution

Pick one question and answer it with experiments. Candidates:

- Few-shot adaptation to dynamics shift, using stable-worldmodel's factors of variation (see arXiv 2611.13592 and 2609.21740).
- Why latent planners fail, and a fix (see 2611.25347 and 2611.25415).
- The counterfactual correction from *Counterfactual World Models* (2611.24778), applied to Push-T.

**Skills shown:** Research taste, extending published work.
**Caveat:** I have only read the abstracts of the adaptation and planner-failure papers, so their reproducibility is unchecked.

## Level 5: Other families of world models

Mostly out of reach on this hardware. Useful for understanding the landscape.

- Generative or pixel-space models (DIAMOND, IRIS, Genie), which predict future frames.
- Dreamer-style models, where RL policies train in imagination.
- Scaling up: V-JEPA 2 pretraining, or V-JEPA 2-AC post-training (reported at under 62 hours of Droid robot video, on top of a large pretrained encoder).

## Recommendation

Do Levels 1–3 as the core. They are feasible and give you a results table. Choose the Level 4 question after you have baselines and have seen where they break.

## Reference points from the papers

| | DINO-WM | LeWM | V-JEPA 2 |
|---|---|---|---|
| Idea | Frozen DINOv2 patch features, trained predictor | End-to-end from pixels, prediction loss + SIGReg | Large frozen video encoder, action-conditioned predictor on top |
| Size | Predictor about 19M | About 15M total | 300M (ViT-L) to 1B (ViT-g) encoders |
| Environments | PointMaze, Wall, Reacher, Push-T, Rope, Granular | Two-Room, Push-T, OGBench-Cube, Reacher | Real Franka arms |

Push-T planning success reported in the LeWM paper: LeWM 96%, DINO-WM 74% (92% with proprioception).

## Repositories

- `galilai-group/stable-worldmodel`: data collection, training and planning in one library, with LeWM and DINO-WM implementations (`scripts/train/lewm.py`, `prejepa.py`) and 30+ environments with factors of variation. Installs on Python 3.10.
- `lucas-maes/le-wm`: LeWM training and evaluation, with pretrained checkpoints on Hugging Face.
- `facebookresearch/vjepa2`: V-JEPA 2 encoders via `torch.hub` or Hugging Face.
- `galilai-group/counterfactual-WM`: Counterfactual World Models. Needs Python 3.11 and a CUDA 13 driver (R580+), and the full paper used about 150 H100 GPU-hours, so reproduce single environments only.
- The original `gaoyuezhou/dino_wm` repo needs an old stack (Python 3.9, mujoco-py 2.1.2, d4rl, gym 0.23), so prefer the stable-worldmodel reimplementation.

## Open items

- ~~Check that LeWM training fits on an 8 GB GPU.~~ Done: batch 128 peaks at 3.7 GB,
  or 0.9 GB with gradient checkpointing ([LeWM](lewm.md)).
- Check the NVIDIA driver version (only relevant to the counterfactual repo).
- Read the Level 4 candidate papers in full before choosing one.
- Level 3: compare the pooled-feature, patch-token and LeWM models on one collection
  with the same planner settings and several seeds ([what is missing](results.md#what-is-missing)).
- Record the V-JEPA 2-AC results; the commands are in [V-JEPA 2-AC](vjepa2-ac.md).

---

[README](../README.md) · [Results at a glance](results.md) · [References](references.md)

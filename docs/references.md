# References

The papers, models and code this repository builds on, with where each is used.
Links and identifiers are the ones the code and pages cite.

## Models and methods

| Work | Used for | Where |
| --- | --- | --- |
| **LeWM** (Maes et al., [arXiv 2603.19312](https://arxiv.org/abs/2603.19312)) | The end-to-end JEPA: ViT-tiny encoder, AdaLN-zero causal predictor, prediction loss plus SIGReg. Code: [`lucas-maes/le-wm`](https://github.com/lucas-maes/le-wm). | [LeWM](lewm.md), `wm.lewm` |
| **SIGReg**, the sketched isotropic Gaussian regularizer of LeJEPA and LeWM | Keeps embeddings from collapsing without a stop-gradient or target network. | `wm.sigreg` |
| **DINO-WM**, frozen DINOv2 patch features with a trained predictor ([`gaoyuezhou/dino_wm`](https://github.com/gaoyuezhou/dino_wm)) | The patch-token predictor and the frozen-encoder setup. | [Patch-token world model](patch-token-wm.md), [Latent world model](latent-world-model.md) |
| **V-JEPA 2** and **V-JEPA 2-AC** ([`facebookresearch/vjepa2`](https://github.com/facebookresearch/vjepa2)) | Frozen video encoders (ViT-L, ViT-H, ViT-g) and the released action-conditioned predictor. The predictor code is vendored at commit `204698b45b3712590f06245fbfba32d3be539812` under its MIT license. | [Encoders and probes](encoders-and-probes.md), [V-JEPA 2-AC](vjepa2-ac.md), `wm.vendor` |
| **DINOv2** (`facebook/dinov2-small`, `-base`, `-large` on the Hugging Face Hub) | Frozen single-frame encoders. | [Encoders and probes](encoders-and-probes.md) |
| **DreamerV3** (`dreamerv3` and `dreamerv3-torch` episode formats) | The export target for training Dreamer-style agents on the collections. | [Data collection](data-collection.md) |

## Simulation and training

| Work | Used for |
| --- | --- |
| [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground) `CartpoleBalance`, on MJX | The environment, rendering, and the dynamics that planners use as ground truth. |
| [Brax](https://github.com/google/brax) PPO, with Playground's tuned `CartpoleBalance` hyperparameters | The expert policy. Pinned to a commit in `pyproject.toml`. |
| [MLflow](https://mlflow.org) | PPO checkpoints, training runs and evaluation results. |

## Candidate directions

The [roadmap](roadmap.md) lists research questions for Level 4. Only their
abstracts have been read, so their reproducibility is unchecked:

- Few-shot adaptation to dynamics shift with stable-worldmodel's factors of
  variation: arXiv 2611.13592 and 2609.21740. Library:
  [`galilai-group/stable-worldmodel`](https://github.com/galilai-group/stable-worldmodel).
- Why latent planners fail, and a fix: arXiv 2611.25347 and 2611.25415.
- The counterfactual correction of *Counterfactual World Models*: arXiv 2611.24778.
  Code: [`galilai-group/counterfactual-WM`](https://github.com/galilai-group/counterfactual-WM).

---

[README](../README.md) · [Roadmap](roadmap.md)

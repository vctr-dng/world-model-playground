# Frozen encoders and state probes

Encode a collection once with a frozen pretrained encoder, then measure how much of the CartPole state its features keep. The probes also decode the latents of every later world model back to observations.

Encode a collection once with a frozen pretrained encoder (DINOv2 or V-JEPA 2
from the Hugging Face Hub). Each step gets the pooled feature of the frame
window ending at it, stored with the observations and actions under
`data/embeddings/<collection_id>/<encoder>_ctx<frames>/`:

```bash
uv run python scripts/cartpole/data/embed.py \
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
uv run python scripts/cartpole/train/train_probe.py \
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

---

Previous: [Data collection](data-collection.md) · Next: [Latent world model](latent-world-model.md) · [README](../README.md)

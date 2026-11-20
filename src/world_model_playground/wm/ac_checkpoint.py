"""The released V-JEPA 2-AC checkpoint: download, loading, and weight mapping.

Meta publishes V-JEPA 2-AC (a ViT-g/16 encoder and a 300M-parameter
action-conditioned predictor post-trained on DROID robot video) only as a
``torch.hub`` checkpoint, not on the Hugging Face Hub. Its encoder differs from
the Hugging Face ``facebook/vjepa2-vitg-fpc64-256`` weights (max abs difference
about 1e-2), so features for the AC predictor must come from this encoder: its
weights are mapped onto the Hugging Face ``VJEPA2Model`` encoder architecture.
"""

from collections.abc import Mapping
from pathlib import Path

import torch

CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/vjepa2/vjepa2-ac-vitg.pt"
DEFAULT_CHECKPOINT = Path(torch.hub.get_dir()) / "checkpoints" / "vjepa2-ac-vitg.pt"
# Hugging Face model providing the encoder architecture (same ViT-g/16 at 256 px).
ENCODER_CONFIG = "facebook/vjepa2-vitg-fpc64-256"
# Settings of the released predictor (upstream src/hub/backbones.py).
PREDICTOR_KWARGS = {
    "img_size": (256, 256),
    "patch_size": 16,
    "num_frames": 64,
    "tubelet_size": 2,
    "embed_dim": 1408,
}


def ensure_checkpoint(path: Path | None = None) -> Path:
    """Returns the local checkpoint, downloading the 11.8 GB file if it is missing."""
    path = Path(path) if path is not None else DEFAULT_CHECKPOINT
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.hub.download_url_to_file(CHECKPOINT_URL, str(path))
    return path


def load_checkpoint(path: Path | None = None) -> dict:
    """Memory-maps the checkpoint, so only the tensors that are used are read."""
    return torch.load(
        ensure_checkpoint(path), map_location="cpu", weights_only=True, mmap=True
    )


def clean_keys(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Strips the ``module.`` and ``backbone.`` prefixes of distributed training."""
    return {
        key.replace("module.", "").replace("backbone.", ""): value
        for key, value in state.items()
    }


def hf_encoder_state_dict(
    encoder_state: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Maps upstream ViT encoder weights onto Hugging Face ``VJEPA2Model.encoder``.

    Upstream fuses the attention projections into ``attn.qkv``; Hugging Face
    keeps separate ``query``, ``key``, and ``value`` layers.
    """
    renames = {
        "patch_embed.proj.": "embeddings.patch_embeddings.proj.",
        "norm.": "layernorm.",
    }
    block_renames = {
        "attn.proj.": "attention.proj.",
        "norm1.": "norm1.",
        "norm2.": "norm2.",
        "mlp.fc1.": "mlp.fc1.",
        "mlp.fc2.": "mlp.fc2.",
    }
    mapped = {}
    for key, value in clean_keys(encoder_state).items():
        if key.startswith("blocks."):
            _, index, rest = key.split(".", 2)
            prefix = f"layer.{index}."
            if rest.startswith("attn.qkv."):
                kind = rest.removeprefix("attn.qkv.")
                for name, part in zip(
                    ("query", "key", "value"), value.chunk(3), strict=True
                ):
                    mapped[f"{prefix}attention.{name}.{kind}"] = part
                continue
            for old, new in block_renames.items():
                if rest.startswith(old):
                    mapped[prefix + new + rest.removeprefix(old)] = value
                    break
            else:
                raise KeyError(f"unexpected encoder weight {key}")
            continue
        for old, new in renames.items():
            if key.startswith(old):
                mapped[new + key.removeprefix(old)] = value
                break
        else:
            raise KeyError(f"unexpected encoder weight {key}")
    return mapped


def load_ac_encoder_backbone(
    checkpoint: Mapping | None = None, *, dtype: torch.dtype = torch.float32
) -> torch.nn.Module:
    """Hugging Face ``VJEPA2Model`` encoder carrying the AC checkpoint's weights."""
    from transformers import VJEPA2Config, VJEPA2Model

    checkpoint = load_checkpoint() if checkpoint is None else checkpoint
    config = VJEPA2Config.from_pretrained(ENCODER_CONFIG)
    with torch.device("meta"):
        encoder = VJEPA2Model(config).encoder
    state = {
        key: value.to(dtype)
        for key, value in hf_encoder_state_dict(checkpoint["encoder"]).items()
    }
    encoder.load_state_dict(state, strict=True, assign=True)
    return encoder

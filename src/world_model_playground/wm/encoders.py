"""Frozen pretrained image and video encoders behind one interface.

Every encoder maps a window of ``context_frames`` uint8 frames ``(B, L, 3, H, W)``
ending at a step to a pooled feature vector ``(B, D)`` and its patch tokens
``(B, N, C)``. Frames are resized to the encoder's input size without the center
crop of the Hugging Face processors: collections render square frames with the
horizontal view of the 320x240 camera, and cropping would hide the rail ends.

- DINOv2 encodes single images (``context_frames=1``); its pooled feature is the
  CLS token concatenated with the mean patch token. Velocities are not visible
  in one frame, so probes stack the features of consecutive steps.
- V-JEPA 2 encodes video clips in tubelets of two frames, so a window of two or
  more frames already carries motion; its pooled feature is the mean token.
"""

from dataclasses import dataclass
from typing import Literal, NamedTuple

import torch
import torch.nn.functional as F
from torch import nn

from world_model_playground.wm.runtime import torch_device

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class EncoderSpec:
    """A pretrained checkpoint and how frames must be fed to it."""

    name: str
    family: Literal["dinov2", "vjepa2"]
    checkpoint: str
    input_size: int
    default_context_frames: int


ENCODERS = {
    spec.name: spec
    for spec in (
        EncoderSpec("dinov2-small", "dinov2", "facebook/dinov2-small", 224, 1),
        EncoderSpec("dinov2-base", "dinov2", "facebook/dinov2-base", 224, 1),
        EncoderSpec("dinov2-large", "dinov2", "facebook/dinov2-large", 224, 1),
        EncoderSpec("vjepa2-vitl", "vjepa2", "facebook/vjepa2-vitl-fpc64-256", 256, 2),
        EncoderSpec("vjepa2-vith", "vjepa2", "facebook/vjepa2-vith-fpc64-256", 256, 2),
        EncoderSpec("vjepa2-vitg", "vjepa2", "facebook/vjepa2-vitg-fpc64-256", 256, 2),
    )
}


class EncoderOutput(NamedTuple):
    pooled: torch.Tensor
    tokens: torch.Tensor


def preprocess_frames(frames: torch.Tensor, size: int) -> torch.Tensor:
    """Scales uint8 ``(..., 3, H, W)`` frames to ImageNet-normalized ``size`` squares."""
    leading = frames.shape[:-3]
    pixels = frames.reshape(-1, *frames.shape[-3:]).float().div_(255.0)
    if pixels.shape[-2:] != (size, size):
        pixels = F.interpolate(
            pixels, size=(size, size), mode="bilinear", antialias=True
        )
    mean = pixels.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = pixels.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return ((pixels - mean) / std).reshape(*leading, 3, size, size)


class FrozenEncoder(nn.Module):
    """Pretrained backbone in eval mode with gradients disabled."""

    feature_dim: int
    token_dim: int

    def __init__(self, spec: EncoderSpec, backbone: nn.Module, context_frames: int):
        super().__init__()
        if context_frames < 1:
            raise ValueError("context_frames must be at least 1")
        self.spec = spec
        self.context_frames = context_frames
        self.backbone = backbone.eval().requires_grad_(False)

    @property
    def device(self) -> torch.device:
        return next(self.backbone.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.backbone.parameters()).dtype

    @torch.inference_mode()
    def forward(self, frames: torch.Tensor) -> EncoderOutput:
        """Encodes uint8 windows ``(B, context_frames, 3, H, W)``."""
        if frames.ndim != 5 or frames.shape[1] != self.context_frames:
            raise ValueError(
                f"frames must have shape (B, {self.context_frames}, 3, H, W)"
            )
        pixels = preprocess_frames(frames.to(self.device), self.spec.input_size)
        pooled, tokens = self._encode(pixels.to(self.dtype))
        return EncoderOutput(pooled.float(), tokens.float())

    def _encode(self, pixels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class Dinov2Encoder(FrozenEncoder):
    def __init__(self, spec: EncoderSpec, backbone: nn.Module, context_frames: int = 1):
        if context_frames != 1:
            raise ValueError("DINOv2 encodes single frames; stack steps in the probe")
        super().__init__(spec, backbone, context_frames)
        self.token_dim = backbone.config.hidden_size
        self.feature_dim = 2 * self.token_dim

    def _encode(self, pixels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.backbone(pixel_values=pixels[:, 0]).last_hidden_state
        cls_token, patches = hidden[:, 0], hidden[:, 1:]
        return torch.cat((cls_token, patches.mean(dim=1)), dim=-1), patches


class VJepa2Encoder(FrozenEncoder):
    def __init__(self, spec: EncoderSpec, backbone: nn.Module, context_frames: int = 2):
        tubelet = backbone.config.tubelet_size
        if context_frames % tubelet:
            # The tubelet convolution would silently drop the newest frames.
            raise ValueError(f"V-JEPA 2 context_frames must be a multiple of {tubelet}")
        super().__init__(spec, backbone, context_frames)
        self.token_dim = backbone.config.hidden_size
        self.feature_dim = self.token_dim

    def _encode(self, pixels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.backbone(pixel_values_videos=pixels).last_hidden_state
        return tokens.mean(dim=1), tokens


def load_encoder(
    name: str,
    *,
    context_frames: int | None = None,
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> FrozenEncoder:
    """Downloads (or reuses the Hugging Face cache of) a registered encoder.

    Weights default to bfloat16 on CUDA and float32 on CPU. Only V-JEPA 2's
    encoder is kept; its masked-prediction head is not needed for features.
    """
    from transformers import Dinov2Model, VJEPA2Model

    if name not in ENCODERS:
        raise ValueError(f"unknown encoder {name!r}; choose from {sorted(ENCODERS)}")
    spec = ENCODERS[name]
    device = torch_device(None if device is None else str(device))
    if dtype is None:
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    context_frames = context_frames or spec.default_context_frames
    if spec.family == "dinov2":
        backbone = Dinov2Model.from_pretrained(spec.checkpoint, dtype=dtype)
        encoder = Dinov2Encoder(spec, backbone, context_frames)
    else:
        model = VJEPA2Model.from_pretrained(spec.checkpoint, dtype=dtype)
        encoder = VJepa2Encoder(spec, model.encoder, context_frames)
    return encoder.to(device)

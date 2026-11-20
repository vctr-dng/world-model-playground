"""V-JEPA 2-AC predictor from facebookresearch/vjepa2 (MIT, see ``LICENSE``).

Vendored from commit 204698b45b3712590f06245fbfba32d3be539812, so the model
code is pinned and tests run offline without ``torch.hub``.
"""

from world_model_playground.wm.vendor.vjepa2.ac_predictor import (
    VisionTransformerPredictorAC,
    build_action_block_causal_attention_mask,
    vit_ac_predictor,
)

UPSTREAM_COMMIT = "204698b45b3712590f06245fbfba32d3be539812"

__all__ = [
    "UPSTREAM_COMMIT",
    "VisionTransformerPredictorAC",
    "build_action_block_causal_attention_mask",
    "vit_ac_predictor",
]

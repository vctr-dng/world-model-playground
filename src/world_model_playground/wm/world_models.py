"""Loading any saved latent world model behind the ``wm.planning`` interface."""

from pathlib import Path

import torch

from world_model_playground.wm.ac_world_model import is_ac_artifact, load_ac_model
from world_model_playground.wm.lewm import is_lewm_artifact, load_lewm
from world_model_playground.wm.predictors import load_predictor


def load_world_model(
    path: Path, device: str | torch.device = "cpu"
) -> tuple[torch.nn.Module, dict]:
    """The Phase 3 latent predictor, a V-JEPA 2-AC or a LeWM model, by artifact kind.

    Both return the model and the metadata their training script saved, with
    the same ``params`` (encoder, context frames, collection, probe) and
    ``episode_splits`` keys.
    """
    if is_ac_artifact(path):
        return load_ac_model(path, device)
    if is_lewm_artifact(path):
        return load_lewm(path, device)
    return load_predictor(path, device)

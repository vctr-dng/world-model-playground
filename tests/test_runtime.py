"""Tests that PyTorch world models can run next to JAX on the same GPU."""

import subprocess
import sys

import pytest
import torch

from world_model_playground.wm.runtime import torch_device

_SHARED_GPU_SCRIPT = """
from world_model_playground.wm.runtime import share_gpu_with_jax, torch_device

share_gpu_with_jax()

import jax
import jax.numpy as jnp
import torch

assert jax.devices()[0].platform == "gpu", jax.devices()
device = torch_device()
assert device.type == "cuda", device

jax_matrix = jnp.full((256, 256), 2.0)
shared = torch.from_dlpack(jax_matrix @ jax_matrix)
assert shared.device.type == "cuda", shared.device
torch_matrix = torch.full((256, 256), 2.0, device=device)
torch.testing.assert_close(shared, torch_matrix @ torch_matrix)
"""


def test_torch_device_honours_explicit_choice():
    assert torch_device("cpu") == torch.device("cpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")
def test_jax_and_torch_share_the_gpu():
    # A fresh interpreter, because JAX reads its allocator settings only once.
    subprocess.run([sys.executable, "-c", _SHARED_GPU_SCRIPT], check=True)

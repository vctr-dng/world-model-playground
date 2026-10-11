"""Running PyTorch world models alongside JAX/MJX on a single GPU."""

import os

import torch


def share_gpu_with_jax() -> None:
    """Stops JAX from reserving most of the GPU so PyTorch can allocate memory too.

    Must run before JAX creates its GPU backend, i.e. before the first JAX
    computation in the process.
    """
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


def torch_device(device: str | None = None) -> torch.device:
    """Returns the requested device, defaulting to CUDA when it is available."""
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

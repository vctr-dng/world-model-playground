"""SIGReg, the sketched isotropic Gaussian regularizer of LeJEPA and LeWM.

Embeddings ``Z`` of shape ``(T, B, D)`` (``B`` samples at each of ``T`` steps)
are pushed toward an isotropic Gaussian, which is what keeps a JEPA trained
without a stop-gradient or a target network from collapsing. By the
Cramer-Wold theorem a distribution matches ``N(0, I)`` when every
one-dimensional projection does, so SIGReg draws ``M`` random unit directions,
projects the embeddings onto them, and averages the Epps-Pulley statistic of
each projection:

    T_m = B * integral w(t) |phi_B(t; Z u_m) - exp(-t^2 / 2)|^2 dt

``phi_B`` is the empirical characteristic function of the ``B`` projections,
``exp(-t^2 / 2)`` that of a standard Gaussian, and ``w(t) = exp(-t^2 / 2)``.
The integral is a trapezoid rule over ``knots`` nodes in ``[0, 3]``, doubled
because the integrand is even. This follows the single-GPU implementation
released with LeWM (https://arxiv.org/abs/2603.19312); the statistic is
computed per step and averaged over steps and directions.
"""

import torch
from torch import nn


class SIGReg(nn.Module):
    """Epps-Pulley distance of embeddings to ``N(0, I)`` along random directions."""

    def __init__(self, knots: int = 17, num_projections: int = 1024):
        super().__init__()
        if knots < 2 or num_projections < 1:
            raise ValueError("knots must be at least 2 and num_projections at least 1")
        self.num_projections = num_projections
        nodes = torch.linspace(0, 3, knots, dtype=torch.float32)
        spacing = 3 / (knots - 1)
        trapezoid = torch.full((knots,), 2 * spacing, dtype=torch.float32)
        trapezoid[[0, -1]] = spacing
        window = torch.exp(-nodes.square() / 2)
        self.register_buffer("nodes", nodes)
        self.register_buffer("gaussian", window)  # characteristic function of N(0, 1)
        self.register_buffer("weights", trapezoid * window)

    def forward(
        self, embeddings: torch.Tensor, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        """The mean statistic of ``embeddings`` ``(T, B, D)``, in float32."""
        if embeddings.ndim != 3:
            raise ValueError("embeddings must have shape (T, B, D)")
        embeddings = embeddings.float()
        samples, dims = embeddings.shape[1], embeddings.shape[2]
        directions = torch.randn(
            dims,
            self.num_projections,
            device=embeddings.device,
            generator=generator,
        )
        directions = directions / directions.norm(dim=0)
        angles = (embeddings @ directions).unsqueeze(-1) * self.nodes  # (T, B, M, K)
        error = (angles.cos().mean(-3) - self.gaussian).square() + (
            angles.sin().mean(-3).square()
        )
        return ((error @ self.weights) * samples).mean()

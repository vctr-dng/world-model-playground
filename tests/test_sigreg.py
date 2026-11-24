"""Tests for SIGReg, the Gaussian regularizer that keeps LeWM from collapsing."""

import pytest
import torch

from world_model_playground.wm.sigreg import SIGReg


@pytest.fixture
def embeddings() -> torch.Tensor:
    return torch.randn(3, 256, 48, generator=torch.Generator().manual_seed(0))


def test_gaussian_embeddings_score_far_below_degenerate_ones(embeddings):
    sigreg = SIGReg(num_projections=256)
    collapsed = torch.zeros_like(embeddings) + 0.01 * torch.randn_like(embeddings)
    low_rank = embeddings[..., :4] @ torch.randn(4, 48)
    gaussian = sigreg(embeddings)

    assert gaussian < 3.0  # the finite-sample floor is about 1
    for degenerate in (collapsed, low_rank, embeddings + 1.0, embeddings * 3.0):
        assert sigreg(degenerate) > 10 * gaussian


def test_gradient_pushes_a_collapsed_batch_apart():
    sigreg = SIGReg(num_projections=128)
    z = (0.05 * torch.randn(2, 128, 16)).requires_grad_()
    optimizer = torch.optim.SGD([z], lr=0.01)
    before = sigreg(z).item()
    for _ in range(30):
        optimizer.zero_grad()
        sigreg(z).backward()
        assert torch.isfinite(z.grad).all()
        optimizer.step()

    assert sigreg(z).item() < before
    assert z.std() > 0.05  # the spread grew back toward one


def test_generator_makes_the_projections_reproducible(embeddings):
    sigreg = SIGReg(num_projections=64)

    first = sigreg(embeddings, generator=torch.Generator().manual_seed(3))
    second = sigreg(embeddings, generator=torch.Generator().manual_seed(3))

    assert first == second


def test_statistic_is_computed_in_float32_for_half_precision_input(embeddings):
    sigreg = SIGReg(num_projections=64)

    result = sigreg(embeddings.bfloat16())

    assert result.dtype == torch.float32
    assert torch.isfinite(result)


def test_rejects_wrong_shapes():
    with pytest.raises(ValueError, match="T, B, D"):
        SIGReg()(torch.randn(8, 4))
    with pytest.raises(ValueError, match="knots"):
        SIGReg(knots=1)

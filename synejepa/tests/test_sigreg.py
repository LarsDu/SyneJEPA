"""Tests for SIGReg and composite losses."""

import torch

from synejepa.loss.composite import CompositeLoss
from synejepa.loss.invariance import InvarianceLoss
from synejepa.loss.sigreg import SIGRegLoss


def test_sigreg_loss_finite():
    loss_fn = SIGRegLoss(num_slices=64)
    z = torch.randn(32, 128)
    loss = loss_fn(z)
    assert loss.isfinite()
    assert loss.item() >= 0


def test_sigreg_collapsed_higher_than_spread():
    """Collapsed embeddings (all identical) should have higher loss than spread ones."""
    loss_fn = SIGRegLoss(num_slices=256)
    spread = torch.randn(256, 64)
    collapsed = torch.ones(256, 64) + torch.randn(256, 64) * 0.01
    loss_spread = loss_fn(spread)
    loss_collapsed = loss_fn(collapsed)
    assert loss_collapsed > loss_spread


def test_invariance_loss_identical():
    loss_fn = InvarianceLoss()
    z = torch.randn(16, 64)
    loss = loss_fn(z, z)
    assert abs(loss.item()) < 1e-5


def test_invariance_loss_orthogonal():
    loss_fn = InvarianceLoss()
    z1 = torch.zeros(1, 64)
    z2 = torch.zeros(1, 64)
    z1[0, 0] = 1.0
    z2[0, 1] = 1.0
    loss = loss_fn(z1, z2)
    assert abs(loss.item() - 2.0) < 1e-5


def test_composite_loss_returns_dict():
    loss_fn = CompositeLoss(lambda_sigreg=1.0, num_slices=64)
    z1 = torch.randn(32, 128)
    z2 = torch.randn(32, 128)
    result = loss_fn(z1, z2)
    assert "loss" in result
    assert "invariance" in result
    assert "sigreg" in result
    assert result["loss"].isfinite()

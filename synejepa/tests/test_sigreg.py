"""Tests for SIGReg (Epps-Pulley), invariance, and composite losses."""

import torch

from synejepa.loss.composite import CompositeLoss
from synejepa.loss.invariance import InvarianceLoss
from synejepa.loss.sigreg import EppsPulley, SIGRegLoss


class TestEppsPulley:
    def test_output_finite(self):
        ep = EppsPulley(n_points=17)
        x = torch.randn(64, 32)  # (N, K) = 64 samples, 32 slices
        stats = ep(x)
        assert stats.shape == (32,)
        assert stats.isfinite().all()

    def test_gaussian_lower_than_uniform(self):
        """Standard Gaussian samples should produce lower test statistic."""
        ep = EppsPulley(n_points=17)
        gaussian = torch.randn(512, 16)
        uniform = torch.rand(512, 16) * 6 - 3  # wider, non-Gaussian
        stat_g = ep(gaussian).mean()
        stat_u = ep(uniform).mean()
        assert stat_g < stat_u


class TestSIGRegLoss:
    def test_loss_finite(self):
        loss_fn = SIGRegLoss(num_slices=64, n_points=17)
        z = torch.randn(32, 128)
        loss = loss_fn(z)
        assert loss.isfinite()
        assert loss.item() >= 0

    def test_collapsed_higher_than_spread(self):
        """Collapsed embeddings (near-identical) should have higher loss."""
        loss_fn = SIGRegLoss(num_slices=256, n_points=17)
        spread = torch.randn(256, 64)
        collapsed = torch.ones(256, 64) + torch.randn(256, 64) * 0.01
        loss_spread = loss_fn(spread)
        loss_collapsed = loss_fn(collapsed)
        assert loss_collapsed > loss_spread

    def test_no_normalization_applied(self):
        """SIGReg should NOT normalize embeddings — verify by checking
        that scaling the input changes the loss."""
        loss_fn = SIGRegLoss(num_slices=64, n_points=17)
        z = torch.randn(64, 32)
        loss_normal = loss_fn(z)
        # Reset step counter so same projections are used
        loss_fn.global_step.zero_()
        loss_scaled = loss_fn(z * 10)
        assert not torch.allclose(loss_normal, loss_scaled, atol=1e-3)

    def test_multi_view_input(self):
        """SIGReg should handle (V, B, D) stacked view input."""
        loss_fn = SIGRegLoss(num_slices=64, n_points=17)
        z = torch.randn(2, 32, 128)  # 2 views, batch 32, dim 128
        loss = loss_fn(z)
        assert loss.isfinite()


class TestInvarianceLoss:
    def test_identical_views_zero_loss(self):
        loss_fn = InvarianceLoss()
        z = torch.randn(16, 64)
        stacked = torch.stack([z, z], dim=0)  # (2, 16, 64)
        loss = loss_fn(stacked)
        assert abs(loss.item()) < 1e-6

    def test_different_views_positive_loss(self):
        loss_fn = InvarianceLoss()
        z1 = torch.randn(16, 64)
        z2 = torch.randn(16, 64)
        stacked = torch.stack([z1, z2], dim=0)
        loss = loss_fn(stacked)
        assert loss.item() > 0

    def test_two_views_equals_quarter_mse(self):
        """With 2 views, loss should equal 0.25 * ||z1 - z2||^2 / (B*D)."""
        loss_fn = InvarianceLoss()
        z1 = torch.randn(8, 32)
        z2 = torch.randn(8, 32)
        stacked = torch.stack([z1, z2], dim=0)
        loss = loss_fn(stacked)
        expected = 0.25 * (z1 - z2).square().mean()
        assert torch.allclose(loss, expected, atol=1e-6)


class TestCompositeLoss:
    def test_returns_all_keys(self):
        loss_fn = CompositeLoss(lambda_sigreg=0.5, num_slices=64, n_points=17)
        z1 = torch.randn(32, 128)
        z2 = torch.randn(32, 128)
        result = loss_fn(z1, z2)
        assert "loss" in result
        assert "invariance" in result
        assert "sigreg" in result
        assert result["loss"].isfinite()

    def test_lambda_zero_is_pure_invariance(self):
        """lambda=0 should make loss equal to invariance only."""
        loss_fn = CompositeLoss(lambda_sigreg=0.0, num_slices=64, n_points=17)
        z1 = torch.randn(32, 64)
        z2 = torch.randn(32, 64)
        result = loss_fn(z1, z2)
        assert torch.allclose(result["loss"], result["invariance"], atol=1e-5)

    def test_lambda_one_is_pure_sigreg(self):
        """lambda=1 should make loss equal to sigreg only."""
        loss_fn = CompositeLoss(lambda_sigreg=1.0, num_slices=64, n_points=17)
        z1 = torch.randn(32, 64)
        z2 = torch.randn(32, 64)
        result = loss_fn(z1, z2)
        assert torch.allclose(result["loss"], result["sigreg"], atol=1e-5)

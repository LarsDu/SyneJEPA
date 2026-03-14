"""Composite loss combining invariance and SIGReg."""

import torch
from torch import Tensor

from synejepa.loss.invariance import InvarianceLoss
from synejepa.loss.sigreg import SIGRegLoss


class CompositeLoss(torch.nn.Module):
    """Weighted combination of invariance and regularization losses.

    loss = invariance(z1, z2) + lambda * (sigreg(z1) + sigreg(z2)) / 2
    """

    def __init__(self, lambda_sigreg: float = 1.0, num_slices: int = 512) -> None:
        super().__init__()
        self.invariance = InvarianceLoss()
        self.sigreg = SIGRegLoss(num_slices=num_slices)
        self.lambda_sigreg = lambda_sigreg

    def forward(self, z1: Tensor, z2: Tensor) -> dict[str, Tensor]:
        """Compute composite loss.

        Returns:
            Dict with keys 'loss', 'invariance', 'sigreg'.
        """
        inv_loss = self.invariance(z1, z2)
        sig_loss = (self.sigreg(z1) + self.sigreg(z2)) / 2
        total = inv_loss + self.lambda_sigreg * sig_loss
        return {
            "loss": total,
            "invariance": inv_loss.detach(),
            "sigreg": sig_loss.detach(),
        }

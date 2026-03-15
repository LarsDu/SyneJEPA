"""Composite loss combining invariance and SIGReg.

Follows the leJEPA paper's convex combination:
    loss = lambda * sigreg(views) + (1 - lambda) * invariance(views)
"""

import torch
import torch.nn as nn
from torch import Tensor

from synejepa.loss.invariance import InvarianceLoss
from synejepa.loss.sigreg import SIGRegLoss


class CompositeLoss(nn.Module):
    """Convex combination of SIGReg regularization and invariance losses.

    loss = lambda_sigreg * sigreg(stacked_views) + (1 - lambda_sigreg) * invariance(stacked_views)

    SIGReg is applied to all views jointly (stacked as leading dimension).
    Invariance pulls views of the same sample toward their centroid.

    Args:
        lambda_sigreg: Interpolation weight in [0, 1]. Higher values emphasize
            Gaussianity regularization; lower values emphasize view invariance.
        num_slices: Number of random projection directions for SIGReg.
        n_points: Number of integration points for Epps-Pulley test.
    """

    def __init__(
        self,
        lambda_sigreg: float = 0.5,
        num_slices: int = 512,
        n_points: int = 17,
    ) -> None:
        super().__init__()
        self.invariance = InvarianceLoss()
        self.sigreg = SIGRegLoss(num_slices=num_slices, n_points=n_points)
        self.lambda_sigreg = lambda_sigreg

    def forward(self, z1: Tensor, z2: Tensor) -> dict[str, Tensor]:
        """Compute composite loss from two views.

        Args:
            z1: View 1 embeddings of shape (B, D).
            z2: View 2 embeddings of shape (B, D).

        Returns:
            Dict with keys 'loss', 'invariance', 'sigreg'.
        """
        stacked = torch.stack((z1, z2), dim=0)  # (2, B, D)

        inv_loss = self.invariance(stacked)
        sig_loss = self.sigreg(stacked)

        total = self.lambda_sigreg * sig_loss + (1 - self.lambda_sigreg) * inv_loss

        return {
            "loss": total,
            "invariance": inv_loss.detach(),
            "sigreg": sig_loss.detach(),
        }

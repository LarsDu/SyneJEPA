"""Invariance loss for multi-view SSL pretraining.

Follows the leJEPA paper: MSE between each view's embeddings and the
cross-view mean (centroid). No normalization is applied to embeddings.
"""

import torch.nn as nn
from torch import Tensor


class InvarianceLoss(nn.Module):
    """MSE-to-centroid invariance loss.

    For V views of B samples with D-dimensional embeddings stacked as (V, B, D),
    computes the mean squared distance from each view to the cross-view centroid:

        loss = mean((centroid - views)^2)

    With 2 views this simplifies to 0.25 * ||z1 - z2||^2.
    """

    def forward(self, z: Tensor) -> Tensor:
        """Compute invariance loss.

        Args:
            z: Stacked view embeddings of shape (V, B, D) where V is number of
               views, B is batch size, D is embedding dimension.

        Returns:
            Scalar loss.
        """
        centroid = z.mean(dim=0)  # (B, D)
        return (centroid - z).square().mean()

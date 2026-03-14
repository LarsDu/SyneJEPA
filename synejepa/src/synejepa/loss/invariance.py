"""Invariance loss for dual-view SSL pretraining."""

import torch
import torch.nn.functional as F
from torch import Tensor


class InvarianceLoss(torch.nn.Module):
    """Cosine invariance loss between two embedding views.

    Encourages embeddings of augmented views of the same input to be similar.
    """

    def forward(self, z1: Tensor, z2: Tensor) -> Tensor:
        """Compute invariance loss.

        Args:
            z1: Embeddings from view 1, shape (B, D).
            z2: Embeddings from view 2, shape (B, D).

        Returns:
            Scalar loss in [0, 2].
        """
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        return 2 - 2 * (z1 * z2).sum(dim=-1).mean()

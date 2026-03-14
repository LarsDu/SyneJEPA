"""Sketched Isotropic Gaussian Regularization (SIGReg) loss.

Regularizes embeddings toward an isotropic Gaussian distribution by projecting
onto random directions and comparing sorted projections to expected Gaussian quantiles.
"""

import math

import torch
import torch.nn.functional as F
from torch import Tensor


class SIGRegLoss(torch.nn.Module):
    """SIGReg loss from the leJEPA paper.

    Projects embeddings onto random unit directions and measures the L2 distance
    between sorted 1D projections and expected standard Gaussian quantiles.
    """

    def __init__(self, num_slices: int = 512) -> None:
        super().__init__()
        self.num_slices = num_slices

    def forward(self, z: Tensor) -> Tensor:
        """Compute SIGReg loss.

        Args:
            z: Embedding tensor of shape (B, D).

        Returns:
            Scalar loss.
        """
        z = F.normalize(z, dim=-1)
        batch_size, dim = z.shape

        directions = torch.randn(dim, self.num_slices, device=z.device, dtype=z.dtype)
        directions = F.normalize(directions, dim=0)

        projections = z @ directions  # (B, num_slices)

        projections_sorted, _ = projections.sort(dim=0)

        expected = torch.erfinv(
            torch.linspace(
                1 / (2 * batch_size),
                1 - 1 / (2 * batch_size),
                batch_size,
                device=z.device,
                dtype=z.dtype,
            )
        ) * math.sqrt(2)

        loss = ((projections_sorted - expected.unsqueeze(1)) ** 2).mean()
        return loss

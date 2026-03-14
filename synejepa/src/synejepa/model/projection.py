"""Projection head for mapping encoder outputs to embedding space."""

import torch.nn as nn
from torch import Tensor


class ProjectionHead(nn.Module):
    """2-layer MLP projection head with batch normalization.

    Maps encoder CLS token output to a lower-dimensional embedding space
    where SIGReg and invariance losses are applied.
    """

    def __init__(
        self,
        input_dim: int = 384,
        hidden_dim: int = 2048,
        output_dim: int = 256,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)

"""Sketched Isotropic Gaussian Regularization (SIGReg) loss.

Implements the Epps-Pulley characteristic function test projected onto random
1D directions (slicing), following the leJEPA paper. Tests whether embeddings
follow an isotropic Gaussian distribution.

References:
    - LeJEPA paper: https://arxiv.org/abs/2511.08544
    - LeJEPA repo: https://github.com/galilai-group/lejepa
"""

import torch
import torch.nn as nn
from torch import Tensor
from torch import distributed as dist
from torch.distributed.nn import all_reduce as functional_all_reduce


def _all_reduce(x: Tensor, op: str = "AVG") -> Tensor:
    """Reduce tensor across DDP ranks if distributed training is active."""
    if dist.is_available() and dist.is_initialized():
        reduce_op = getattr(dist.ReduceOp, op.upper())
        return functional_all_reduce(x, reduce_op)
    return x


def _is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def _get_world_size() -> int:
    if _is_distributed():
        return dist.get_world_size()
    return 1


class EppsPulley(nn.Module):
    """Epps-Pulley two-sample test via characteristic functions.

    Tests whether 1D projections match a standard normal distribution by
    comparing the empirical characteristic function to exp(-t^2/2) at a set
    of integration points using trapezoidal quadrature.

    Supports DDP via all_reduce on cos/sin means so the test operates on
    the full global batch.

    Args:
        t_max: Maximum integration point.
        n_points: Number of integration points (must be odd).
    """

    def __init__(self, t_max: float = 3.0, n_points: int = 17) -> None:
        super().__init__()
        assert n_points % 2 == 1, "n_points must be odd"
        self.n_points = n_points

        t = torch.linspace(0, t_max, n_points, dtype=torch.float32)
        self.register_buffer("t", t)

        dt = t_max / (n_points - 1)
        weights = torch.full((n_points,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt  # half-weight at boundaries

        phi = (-0.5 * t.square()).exp()  # Gaussian characteristic function
        self.register_buffer("phi", phi)
        self.register_buffer("weights", weights * phi)

    def forward(self, x: Tensor) -> Tensor:
        """Compute Epps-Pulley test statistic.

        Args:
            x: Projected samples of shape (*, N, K) where N is sample count
               and K is the number of slices.

        Returns:
            Test statistics of shape (*, K).
        """
        n = x.size(-2)

        # Evaluate empirical characteristic function at integration points
        x_t = x.unsqueeze(-1) * self.t  # (*, N, K, n_points)
        cos_vals = torch.cos(x_t)
        sin_vals = torch.sin(x_t)

        # Mean across samples (batch dim = -3)
        cos_mean = cos_vals.mean(-3)  # (*, K, n_points)
        sin_mean = sin_vals.mean(-3)

        # Reduce across DDP ranks for global statistics
        cos_mean = _all_reduce(cos_mean)
        sin_mean = _all_reduce(sin_mean)

        # Squared difference from Gaussian characteristic function
        err = (cos_mean - self.phi).square() + sin_mean.square()

        # Weighted trapezoidal integration
        return (err @ self.weights) * n * _get_world_size()


class SIGRegLoss(nn.Module):
    """SIGReg: Sliced Isotropic Gaussian Regularization.

    Projects D-dimensional embeddings onto random 1D directions and applies
    the Epps-Pulley test to each projection. Random seeds are synchronized
    across DDP ranks to ensure consistent projection directions.

    Args:
        num_slices: Number of random projection directions.
        t_max: Maximum integration point for Epps-Pulley.
        n_points: Number of integration points for Epps-Pulley.
        clip_value: Minimum threshold for test statistics (None = no clipping).
    """

    def __init__(
        self,
        num_slices: int = 512,
        t_max: float = 3.0,
        n_points: int = 17,
        clip_value: float | None = None,
    ) -> None:
        super().__init__()
        self.num_slices = num_slices
        self.clip_value = clip_value
        self.epps_pulley = EppsPulley(t_max=t_max, n_points=n_points)
        self.register_buffer("global_step", torch.zeros((), dtype=torch.long))
        self._generator: torch.Generator | None = None
        self._generator_device: torch.device | None = None

    def _get_generator(self, device: torch.device, seed: int) -> torch.Generator:
        if self._generator is None or self._generator_device != device:
            self._generator = torch.Generator(device=device)
            self._generator_device = device
        self._generator.manual_seed(seed)
        return self._generator

    def forward(self, z: Tensor) -> Tensor:
        """Compute SIGReg loss.

        Args:
            z: Embedding tensor of shape (*, N, D) where N is sample count
               and D is embedding dimension. No normalization is applied —
               SIGReg tests raw embeddings for Gaussianity.

        Returns:
            Scalar loss (mean of test statistics across slices).
        """
        with torch.no_grad():
            # Synchronize seed across DDP ranks
            global_step_sync = _all_reduce(self.global_step.clone(), op="MAX")
            seed = global_step_sync.item()

            g = self._get_generator(z.device, seed)
            proj_matrix = torch.randn(
                z.size(-1), self.num_slices, device=z.device, dtype=z.dtype, generator=g
            )
            proj_matrix /= proj_matrix.norm(p=2, dim=0)
            self.global_step.add_(1)

        # Project embeddings onto random directions: (*, N, D) @ (D, K) -> (*, N, K)
        projected = z @ proj_matrix

        # Apply Epps-Pulley test to each slice
        stats = self.epps_pulley(projected)

        if self.clip_value is not None:
            stats = stats.clamp(min=self.clip_value)

        return stats.mean()

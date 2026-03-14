"""Random patch masking for audio spectrograms."""

import torch
from torch import Tensor


class RandomPatchMasker:
    """Generates random binary masks for spectrogram patches.

    Given a spectrogram of shape (n_mels, n_frames) and a patch size, randomly selects
    patches to mask at a ratio drawn from U(mask_ratio_min, mask_ratio_max).
    """

    def __init__(
        self,
        n_mels: int = 128,
        n_frames: int = 400,
        patch_size: int = 16,
        mask_ratio_min: float = 0.4,
        mask_ratio_max: float = 0.6,
    ) -> None:
        self.grid_h = n_mels // patch_size
        self.grid_w = n_frames // patch_size
        self.n_patches = self.grid_h * self.grid_w
        self.mask_ratio_min = mask_ratio_min
        self.mask_ratio_max = mask_ratio_max

    def __call__(self, rng: torch.Generator | None = None) -> tuple[Tensor, Tensor]:
        """Generate a random mask.

        Returns:
            context_indices: 1D tensor of indices for visible (unmasked) patches.
            target_indices: 1D tensor of indices for masked patches.
        """
        ratio = torch.empty(1).uniform_(self.mask_ratio_min, self.mask_ratio_max, generator=rng)
        n_masked = int(self.n_patches * ratio.item())
        n_masked = max(1, min(n_masked, self.n_patches - 1))

        perm = torch.randperm(self.n_patches, generator=rng)
        target_indices = perm[:n_masked].sort().values
        context_indices = perm[n_masked:].sort().values
        return context_indices, target_indices

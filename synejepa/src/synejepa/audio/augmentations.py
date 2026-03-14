"""Optional SpecAugment-style augmentations for audio spectrograms.

These are NOT used by default (Audio-JEPA found masking-only is best).
Enable only if baseline underperforms.
"""

import torch
from torch import Tensor


class FrequencyMask:
    """Masks a random contiguous band of frequency bins."""

    def __init__(self, max_width: int = 27) -> None:
        self.max_width = max_width

    def __call__(self, spec: Tensor, rng: torch.Generator | None = None) -> Tensor:
        n_mels = spec.shape[-2]
        width = torch.randint(1, self.max_width + 1, (1,), generator=rng).item()
        start = torch.randint(0, max(1, n_mels - width), (1,), generator=rng).item()
        spec = spec.clone()
        spec[..., start : start + width, :] = 0.0
        return spec


class TimeMask:
    """Masks a random contiguous span of time frames."""

    def __init__(self, max_width: int = 40) -> None:
        self.max_width = max_width

    def __call__(self, spec: Tensor, rng: torch.Generator | None = None) -> Tensor:
        n_frames = spec.shape[-1]
        width = torch.randint(1, self.max_width + 1, (1,), generator=rng).item()
        start = torch.randint(0, max(1, n_frames - width), (1,), generator=rng).item()
        spec = spec.clone()
        spec[..., start : start + width] = 0.0
        return spec


class GaussianNoise:
    """Adds Gaussian noise at a random SNR."""

    def __init__(self, snr_min_db: float = 25.0, snr_max_db: float = 30.0) -> None:
        self.snr_min_db = snr_min_db
        self.snr_max_db = snr_max_db

    def __call__(self, spec: Tensor, rng: torch.Generator | None = None) -> Tensor:
        snr_db = torch.empty(1).uniform_(self.snr_min_db, self.snr_max_db, generator=rng).item()
        signal_power = spec.pow(2).mean()
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = torch.randn_like(spec, generator=rng if rng else None) * noise_power.sqrt()
        return spec + noise

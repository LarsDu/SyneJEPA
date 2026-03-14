"""Audio dataset loading and dual-view generation for SSL pretraining."""

from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import Dataset

from synejepa.audio.masking import RandomPatchMasker
from synejepa.audio.preprocessing import AudioProcessor, SpectrogramConfig


class AudioSSLDataset(Dataset):
    """Dataset that produces dual-view spectrograms with independent masks.

    Each sample returns two mel spectrograms from the same audio segment
    (with a small time shift between views) and two independent random masks.
    """

    def __init__(
        self,
        audio_paths: list[str],
        config: SpectrogramConfig | None = None,
        patch_size: int = 16,
        mask_ratio_min: float = 0.4,
        mask_ratio_max: float = 0.6,
        view_shift_sec: float = 0.5,
    ) -> None:
        self.audio_paths = audio_paths
        self.processor = AudioProcessor(config)
        self.config = self.processor.config
        self.view_shift_sec = view_shift_sec
        self.masker = RandomPatchMasker(
            n_mels=self.config.n_mels,
            n_frames=self.config.n_frames,
            patch_size=patch_size,
            mask_ratio_min=mask_ratio_min,
            mask_ratio_max=mask_ratio_max,
        )

    def __len__(self) -> int:
        return len(self.audio_paths)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        waveform = self.processor.load(self.audio_paths[idx])

        crop_len = int(self.config.sample_rate * self.config.window_sec)
        shift_samples = int(self.config.sample_rate * self.view_shift_sec)
        total_needed = crop_len + shift_samples

        if waveform.shape[-1] < total_needed:
            waveform = torch.nn.functional.pad(waveform, (0, total_needed - waveform.shape[-1]))

        max_start = waveform.shape[-1] - total_needed
        start1 = torch.randint(0, max(1, max_start), (1,)).item()
        shift = torch.randint(0, shift_samples + 1, (1,)).item()
        start2 = start1 + shift

        view1_wav = waveform[start1 : start1 + crop_len]
        view2_wav = waveform[start2 : start2 + crop_len]

        spec1 = self.processor.to_mel_spectrogram(view1_wav)
        spec2 = self.processor.to_mel_spectrogram(view2_wav)

        ctx1, tgt1 = self.masker()
        ctx2, tgt2 = self.masker()

        return {
            "spec1": spec1,
            "spec2": spec2,
            "context_indices_1": ctx1,
            "target_indices_1": tgt1,
            "context_indices_2": ctx2,
            "target_indices_2": tgt2,
        }


class CachedSpectrogramDataset(Dataset):
    """Dataset that loads pre-computed spectrograms from .pt files for faster I/O."""

    def __init__(
        self,
        cache_dir: str,
        n_mels: int = 128,
        n_frames: int = 400,
        patch_size: int = 16,
        mask_ratio_min: float = 0.4,
        mask_ratio_max: float = 0.6,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.files = sorted(self.cache_dir.glob("*.pt"))
        self.masker = RandomPatchMasker(
            n_mels=n_mels,
            n_frames=n_frames,
            patch_size=patch_size,
            mask_ratio_min=mask_ratio_min,
            mask_ratio_max=mask_ratio_max,
        )

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        spec = torch.load(self.files[idx], weights_only=True)
        ctx1, tgt1 = self.masker()
        ctx2, tgt2 = self.masker()
        return {
            "spec1": spec,
            "spec2": spec,
            "context_indices_1": ctx1,
            "target_indices_1": tgt1,
            "context_indices_2": ctx2,
            "target_indices_2": tgt2,
        }

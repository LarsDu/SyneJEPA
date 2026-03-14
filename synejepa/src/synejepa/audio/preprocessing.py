"""Audio preprocessing: wav → mel spectrogram."""

from dataclasses import dataclass

import torch
import torchaudio
from torch import Tensor


@dataclass(frozen=True)
class SpectrogramConfig:
    sample_rate: int = 16000
    n_mels: int = 128
    n_fft: int = 1024
    hop_length: int = 160
    window_sec: float = 4.0

    @property
    def n_frames(self) -> int:
        return int(self.sample_rate * self.window_sec / self.hop_length)


class AudioProcessor:
    """Converts raw audio waveforms to normalized log-mel spectrograms."""

    def __init__(self, config: SpectrogramConfig | None = None) -> None:
        self.config = config or SpectrogramConfig()
        self._mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.config.sample_rate,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            n_mels=self.config.n_mels,
            power=2.0,
        )
        self._resample_cache: dict[int, torchaudio.transforms.Resample] = {}

    def load(self, path: str) -> Tensor:
        """Load audio file, resample to target rate, and convert to mono."""
        waveform, sr = torchaudio.load(path)
        return self._to_mono_and_resample(waveform, sr)

    def _to_mono_and_resample(self, waveform: Tensor, source_sr: int) -> Tensor:
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if source_sr != self.config.sample_rate:
            if source_sr not in self._resample_cache:
                self._resample_cache[source_sr] = torchaudio.transforms.Resample(
                    orig_freq=source_sr, new_freq=self.config.sample_rate
                )
            waveform = self._resample_cache[source_sr](waveform)
        return waveform.squeeze(0)

    def to_mel_spectrogram(self, waveform: Tensor) -> Tensor:
        """Convert waveform to normalized log-mel spectrogram.

        Args:
            waveform: 1D tensor of audio samples at config.sample_rate.

        Returns:
            Tensor of shape (1, n_mels, n_frames) — log-scaled, zero-mean, unit-variance.
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        mel = self._mel_transform(waveform)
        mel = torch.log(mel + 1e-8)
        mel = (mel - mel.mean()) / (mel.std() + 1e-8)
        return mel

    def random_crop(self, waveform: Tensor, rng: torch.Generator | None = None) -> Tensor:
        """Extract a random fixed-length crop from a waveform."""
        target_len = int(self.config.sample_rate * self.config.window_sec)
        if waveform.shape[-1] <= target_len:
            return torch.nn.functional.pad(waveform, (0, target_len - waveform.shape[-1]))
        max_start = waveform.shape[-1] - target_len
        start = torch.randint(0, max_start, (1,), generator=rng).item()
        return waveform[..., start : start + target_len]

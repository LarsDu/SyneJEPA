"""Tests for audio preprocessing pipeline."""

import torch

from synejepa.audio.preprocessing import AudioProcessor, SpectrogramConfig


def test_spectrogram_config_n_frames():
    config = SpectrogramConfig(sample_rate=16000, hop_length=160, window_sec=4.0)
    assert config.n_frames == 400


def test_to_mel_spectrogram_shape():
    processor = AudioProcessor()
    waveform = torch.randn(16000 * 4)  # 4 seconds at 16kHz
    spec = processor.to_mel_spectrogram(waveform)
    assert spec.shape[0] == 1
    assert spec.shape[1] == 128
    assert spec.shape[2] == 400 or abs(spec.shape[2] - 400) <= 1


def test_to_mel_spectrogram_normalized():
    processor = AudioProcessor()
    waveform = torch.randn(16000 * 4)
    spec = processor.to_mel_spectrogram(waveform)
    assert abs(spec.mean().item()) < 0.1
    assert abs(spec.std().item() - 1.0) < 0.1


def test_random_crop_correct_length():
    processor = AudioProcessor()
    waveform = torch.randn(16000 * 30)  # 30 seconds
    cropped = processor.random_crop(waveform)
    expected_len = int(16000 * 4.0)
    assert cropped.shape[-1] == expected_len


def test_random_crop_pads_short_audio():
    processor = AudioProcessor()
    waveform = torch.randn(16000)  # 1 second
    cropped = processor.random_crop(waveform)
    expected_len = int(16000 * 4.0)
    assert cropped.shape[-1] == expected_len

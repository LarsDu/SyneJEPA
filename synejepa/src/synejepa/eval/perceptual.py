"""Perceptual feature correlation with learned embeddings."""

import logging

import torch
import torchaudio
from scipy import stats
from sklearn.decomposition import PCA
from torch import Tensor

logger = logging.getLogger(__name__)


def extract_perceptual_features(waveform: Tensor, sample_rate: int = 16000) -> dict[str, float]:
    """Extract pitch, RMS energy, and spectral centroid from a waveform."""
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    rms = waveform.pow(2).mean().sqrt().item()

    spec = torch.stft(
        waveform.squeeze(0),
        n_fft=1024,
        hop_length=160,
        return_complex=True,
    )
    mag = spec.abs()
    freqs = torch.linspace(0, sample_rate / 2, mag.shape[0])
    centroid = (freqs.unsqueeze(1) * mag).sum(dim=0) / (mag.sum(dim=0) + 1e-8)
    mean_centroid = centroid.mean().item()

    pitch = torchaudio.functional.detect_pitch_frequency(waveform, sample_rate)
    mean_pitch = pitch[pitch > 0].mean().item() if (pitch > 0).any() else 0.0

    return {"pitch": mean_pitch, "energy": rms, "spectral_centroid": mean_centroid}


def compute_perceptual_correlation(
    embeddings: Tensor,
    perceptual_features: list[dict[str, float]],
    n_components: int = 3,
) -> dict[str, float]:
    """Compute Spearman correlation between PCA dims and perceptual features."""
    pca = PCA(n_components=n_components)
    pca_embeddings = pca.fit_transform(embeddings.numpy())

    results = {}
    feature_names = ["pitch", "energy", "spectral_centroid"]

    for feat_name in feature_names:
        feat_values = [f[feat_name] for f in perceptual_features]
        max_corr = 0.0
        for dim in range(n_components):
            corr, _ = stats.spearmanr(pca_embeddings[:, dim], feat_values)
            max_corr = max(max_corr, abs(corr))
        results[f"max_corr_{feat_name}"] = max_corr
        logger.info(f"Max Spearman correlation with {feat_name}: {max_corr:.4f}")

    return results

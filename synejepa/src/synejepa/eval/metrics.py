"""Embedding quality metrics: isotropy, effective rank, retrieval."""

import logging
import math

import torch
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)


def isotropy_score(embeddings: Tensor) -> float:
    """Compute isotropy as ratio of min/max eigenvalue of covariance.

    A score close to 1.0 indicates isotropic embeddings.
    """
    centered = embeddings - embeddings.mean(dim=0)
    cov = (centered.T @ centered) / (centered.shape[0] - 1)
    eigenvalues = torch.linalg.eigvalsh(cov)
    eigenvalues = eigenvalues.clamp(min=1e-10)
    score = (eigenvalues.min() / eigenvalues.max()).item()
    logger.info(f"Isotropy score: {score:.6f}")
    return score


def effective_rank(embeddings: Tensor) -> float:
    """Compute effective rank via entropy of normalized singular values.

    Should approach the embedding dimension if no collapse.
    """
    centered = embeddings - embeddings.mean(dim=0)
    singular_values = torch.linalg.svdvals(centered)
    normalized = singular_values / singular_values.sum()
    normalized = normalized.clamp(min=1e-10)
    entropy = -(normalized * normalized.log()).sum().item()
    rank = math.exp(entropy)
    logger.info(f"Effective rank: {rank:.2f} / {embeddings.shape[1]}")
    return rank


def nn_retrieval_recall(
    query_embeddings: Tensor,
    db_embeddings: Tensor,
    query_labels: Tensor,
    db_labels: Tensor,
    k: int = 5,
) -> float:
    """Compute Recall@k for nearest-neighbor retrieval."""
    query_norm = F.normalize(query_embeddings, dim=-1)
    db_norm = F.normalize(db_embeddings, dim=-1)
    similarity = query_norm @ db_norm.T
    _, topk_indices = similarity.topk(k, dim=-1)
    topk_labels = db_labels[topk_indices]
    hits = (topk_labels == query_labels.unsqueeze(1)).any(dim=1)
    recall = hits.float().mean().item()
    logger.info(f"Recall@{k}: {recall:.4f}")
    return recall

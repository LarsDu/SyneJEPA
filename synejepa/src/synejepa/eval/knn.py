"""k-Nearest Neighbor evaluation on frozen embeddings."""

import logging

import torch
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)


def knn_evaluate(
    train_embeddings: Tensor,
    train_labels: Tensor,
    test_embeddings: Tensor,
    test_labels: Tensor,
    k_values: list[int] | None = None,
) -> dict[str, float]:
    """Evaluate frozen embeddings using k-NN classification.

    Args:
        train_embeddings: (N_train, D) normalized embeddings.
        train_labels: (N_train,) integer labels.
        test_embeddings: (N_test, D) normalized embeddings.
        test_labels: (N_test,) integer labels.
        k_values: List of k values to evaluate.

    Returns:
        Dict mapping "accuracy@k" to accuracy value.
    """
    if k_values is None:
        k_values = [5, 20]

    train_embeddings = F.normalize(train_embeddings, dim=-1)
    test_embeddings = F.normalize(test_embeddings, dim=-1)

    similarity = test_embeddings @ train_embeddings.T  # (N_test, N_train)

    results = {}
    for k in k_values:
        _, topk_indices = similarity.topk(k, dim=-1)
        topk_labels = train_labels[topk_indices]  # (N_test, k)
        predicted = topk_labels.mode(dim=-1).values
        accuracy = (predicted == test_labels).float().mean().item()
        results[f"accuracy@{k}"] = accuracy
        logger.info(f"k-NN (k={k}): accuracy={accuracy:.4f}")

    return results

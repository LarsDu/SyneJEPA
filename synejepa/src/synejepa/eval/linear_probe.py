"""Linear probe evaluation on frozen encoder features."""

import logging

import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from torch import Tensor
from torch.utils.data import DataLoader, Subset

logger = logging.getLogger(__name__)


class LinearProbe(nn.Module):
    def __init__(self, input_dim: int, num_classes: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x)


def extract_embeddings(
    encoder: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    """Extract frozen embeddings from an encoder for all samples."""
    encoder.eval()
    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            specs = batch["spec"].to(device)
            labels = batch["label"]
            embeddings = encoder(specs)
            all_embeddings.append(embeddings.cpu())
            all_labels.append(labels)

    return torch.cat(all_embeddings), torch.cat(all_labels)


def train_linear_probe(
    embeddings: Tensor,
    labels: Tensor,
    num_classes: int,
    n_folds: int = 5,
    epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int = 64,
) -> dict[str, float]:
    """Train and evaluate a linear probe with k-fold cross-validation."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    fold_accuracies = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(embeddings, labels)):
        probe = LinearProbe(embeddings.shape[1], num_classes).to(device)
        optimizer = torch.optim.AdamW(probe.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        train_emb = embeddings[train_idx].to(device)
        train_lbl = labels[train_idx].to(device)
        val_emb = embeddings[val_idx].to(device)
        val_lbl = labels[val_idx].to(device)

        for epoch in range(epochs):
            probe.train()
            for i in range(0, len(train_emb), batch_size):
                batch_emb = train_emb[i : i + batch_size]
                batch_lbl = train_lbl[i : i + batch_size]
                logits = probe(batch_emb)
                loss = criterion(logits, batch_lbl)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        probe.eval()
        with torch.no_grad():
            val_logits = probe(val_emb)
            val_preds = val_logits.argmax(dim=-1)
            accuracy = (val_preds == val_lbl).float().mean().item()
            fold_accuracies.append(accuracy)
            logger.info(f"Fold {fold}: accuracy={accuracy:.4f}")

    mean_acc = sum(fold_accuracies) / len(fold_accuracies)
    logger.info(f"Mean accuracy: {mean_acc:.4f}")

    return {
        "mean_accuracy": mean_acc,
        "fold_accuracies": fold_accuracies,
    }

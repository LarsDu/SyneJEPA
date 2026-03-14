"""Training loop for SyneJEPA SSL pretraining."""

import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from synejepa.loss.composite import CompositeLoss

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    epochs: int = 200
    batch_size: int = 8
    gradient_accumulation_steps: int = 16
    lr: float = 1e-4
    weight_decay: float = 0.05
    warmup_epochs: int = 10
    lambda_sigreg: float = 1.0
    num_slices: int = 512
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "tb_logs"
    checkpoint_every: int = 10
    use_amp: bool = True


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def is_main_process() -> bool:
    return get_rank() == 0


class Trainer:
    """Handles the SSL pretraining loop with optional DDP support."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        config: TrainingConfig,
    ) -> None:
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = model.to(self.device)
        if is_distributed():
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            self.model = DDP(self.model, device_ids=[local_rank])

        self.train_loader = train_loader
        self.criterion = CompositeLoss(
            lambda_sigreg=config.lambda_sigreg,
            num_slices=config.num_slices,
        ).to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
        self.scheduler = self._build_scheduler()

        self.writer = None
        if is_main_process():
            Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
            Path(config.log_dir).mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(config.log_dir)

    def _build_scheduler(self) -> torch.optim.lr_scheduler.LRScheduler:
        warmup = torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=self.config.warmup_epochs,
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.epochs - self.config.warmup_epochs,
            eta_min=self.config.lr / 1000,
        )
        return torch.optim.lr_scheduler.SequentialLR(
            self.optimizer,
            schedulers=[warmup, cosine],
            milestones=[self.config.warmup_epochs],
        )

    def train(self) -> None:
        for epoch in range(self.config.epochs):
            if isinstance(self.train_loader.sampler, DistributedSampler):
                self.train_loader.sampler.set_epoch(epoch)

            metrics = self._train_epoch(epoch)
            self.scheduler.step()

            if is_main_process():
                self._log_epoch(epoch, metrics)
                if (epoch + 1) % self.config.checkpoint_every == 0:
                    self._save_checkpoint(epoch, metrics)

    def _train_epoch(self, epoch: int) -> dict[str, float]:
        self.model.train()
        total_loss = 0.0
        total_inv = 0.0
        total_sig = 0.0
        n_steps = 0

        self.optimizer.zero_grad()
        for step, batch in enumerate(self.train_loader):
            spec1 = batch["spec1"].to(self.device)
            spec2 = batch["spec2"].to(self.device)
            ctx1 = batch["context_indices_1"].to(self.device)
            ctx2 = batch["context_indices_2"].to(self.device)

            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.config.use_amp):
                z1, z2 = self.model(spec1, spec2, ctx1, ctx2)
                losses = self.criterion(z1, z2)
                loss = losses["loss"] / self.config.gradient_accumulation_steps

            loss.backward()

            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

            total_loss += losses["loss"].item()
            total_inv += losses["invariance"].item()
            total_sig += losses["sigreg"].item()
            n_steps += 1

        return {
            "loss": total_loss / max(n_steps, 1),
            "invariance": total_inv / max(n_steps, 1),
            "sigreg": total_sig / max(n_steps, 1),
            "lr": self.optimizer.param_groups[0]["lr"],
        }

    def _log_epoch(self, epoch: int, metrics: dict[str, float]) -> None:
        logger.info(
            f"Epoch {epoch}: loss={metrics['loss']:.4f} "
            f"inv={metrics['invariance']:.4f} sig={metrics['sigreg']:.4f} "
            f"lr={metrics['lr']:.6f}"
        )
        if self.writer:
            for key, val in metrics.items():
                self.writer.add_scalar(f"train/{key}", val, epoch)

        log_path = Path(self.config.log_dir) / "metrics.jsonl"
        with open(log_path, "a") as f:
            f.write(json.dumps({"epoch": epoch, **metrics}) + "\n")

    def _save_checkpoint(self, epoch: int, metrics: dict[str, float]) -> None:
        model_state = self.model.module.state_dict() if is_distributed() else self.model.state_dict()
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model_state,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "metrics": metrics,
        }
        path = Path(self.config.checkpoint_dir) / f"checkpoint_epoch_{epoch:04d}.pt"
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint: {path}")

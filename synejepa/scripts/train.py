"""Hydra entrypoint for SyneJEPA pretraining."""

import logging
import os

import hydra
import torch
import torch.distributed as dist
from omegaconf import DictConfig
from torch.utils.data import DataLoader, DistributedSampler

from synejepa.audio.dataset import CachedSpectrogramDataset
from synejepa.audio.preprocessing import SpectrogramConfig
from synejepa.model.encoder import EncoderConfig
from synejepa.model.synejepa import SyneJEPA
from synejepa.training.trainer import Trainer, TrainingConfig

logger = logging.getLogger(__name__)


def setup_distributed() -> None:
    if "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)


@hydra.main(version_base=None, config_path="../configs/pretrain", config_name="fma_small")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO)
    setup_distributed()

    spec_config = SpectrogramConfig(
        sample_rate=cfg.data.sample_rate,
        n_mels=cfg.data.n_mels,
        n_fft=cfg.data.n_fft,
        hop_length=cfg.data.hop_length,
        window_sec=cfg.data.window_sec,
    )

    dataset = CachedSpectrogramDataset(
        cache_dir=cfg.data.cache_dir,
        n_mels=cfg.data.n_mels,
        n_frames=spec_config.n_frames,
        patch_size=cfg.data.patch_size,
        mask_ratio_min=cfg.data.mask_ratio_min,
        mask_ratio_max=cfg.data.mask_ratio_max,
    )

    sampler = DistributedSampler(dataset) if dist.is_initialized() else None
    loader = DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    encoder_config = EncoderConfig(
        img_size=(cfg.data.n_mels, spec_config.n_frames),
        patch_size=cfg.model.patch_size,
        embed_dim=cfg.model.embed_dim,
        depth=cfg.model.depth,
        num_heads=cfg.model.num_heads,
        use_gradient_checkpointing=cfg.model.use_gradient_checkpointing,
    )

    model = SyneJEPA(
        encoder_config=encoder_config,
        proj_hidden_dim=cfg.model.proj_hidden_dim,
        proj_output_dim=cfg.model.proj_output_dim,
    )

    train_config = TrainingConfig(
        epochs=cfg.training.epochs,
        batch_size=cfg.training.batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
        warmup_epochs=cfg.training.warmup_epochs,
        lambda_sigreg=cfg.training.lambda_sigreg,
        num_slices=cfg.training.num_slices,
        checkpoint_dir=cfg.training.checkpoint_dir,
        log_dir=cfg.training.log_dir,
        checkpoint_every=cfg.training.checkpoint_every,
        use_amp=cfg.training.use_amp,
    )

    trainer = Trainer(model=model, train_loader=loader, config=train_config)

    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {param_count:,}")
    logger.info(f"Effective batch size: {cfg.training.batch_size * cfg.training.gradient_accumulation_steps}")
    logger.info(f"Training for {cfg.training.epochs} epochs")

    trainer.train()

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

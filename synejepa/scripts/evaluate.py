"""Hydra entrypoint for SyneJEPA evaluation."""

import logging

import hydra
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from synejepa.eval.knn import knn_evaluate
from synejepa.eval.linear_probe import extract_embeddings, train_linear_probe
from synejepa.eval.metrics import effective_rank, isotropy_score, nn_retrieval_recall
from synejepa.model.encoder import AudioEncoder, EncoderConfig

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs/eval", config_name="esc50")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder_config = EncoderConfig(
        img_size=(cfg.data.n_mels, int(cfg.data.sample_rate * cfg.data.window_sec / cfg.data.hop_length)),
        patch_size=cfg.model.patch_size,
        embed_dim=cfg.model.embed_dim,
        depth=cfg.model.depth,
        num_heads=cfg.model.num_heads,
    )
    encoder = AudioEncoder(encoder_config).to(device)

    checkpoint = torch.load(cfg.eval.checkpoint_path, map_location=device, weights_only=True)
    model_state = checkpoint["model_state_dict"]
    encoder_state = {
        k.replace("encoder.", ""): v
        for k, v in model_state.items()
        if k.startswith("encoder.")
    }
    encoder.load_state_dict(encoder_state)
    encoder.eval()

    # TODO: Load actual ESC-50 dataset here
    logger.info("Evaluation script ready. Provide ESC-50 dataset to run evals.")
    logger.info(f"Loaded checkpoint from: {cfg.eval.checkpoint_path}")


if __name__ == "__main__":
    main()

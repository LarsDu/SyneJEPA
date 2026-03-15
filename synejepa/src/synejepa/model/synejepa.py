"""SyneJEPA model: composes encoder and projection head."""

import torch.nn as nn
from torch import Tensor

from synejepa.model.encoder import AudioEncoder, EncoderConfig
from synejepa.model.projection import ProjectionHead


class SyneJEPA(nn.Module):
    """Full SyneJEPA model for self-supervised audio pretraining.

    Composes an audio ViT encoder with a projection head. Takes two views
    of the same audio and returns projected embeddings for loss computation.
    """

    def __init__(
        self,
        encoder_config: EncoderConfig | None = None,
        proj_hidden_dim: int = 2048,
        proj_output_dim: int = 256,
    ) -> None:
        super().__init__()
        self.encoder = AudioEncoder(encoder_config)
        self.projector = ProjectionHead(
            input_dim=self.encoder.embed_dim,
            hidden_dim=proj_hidden_dim,
            output_dim=proj_output_dim,
        )

    def encode(self, x: Tensor, context_indices: Tensor | None = None) -> Tensor:
        """Encode and project a single view."""
        h = self.encoder(x, context_indices=context_indices)
        return self.projector(h)

    def forward(
        self,
        spec1: Tensor,
        spec2: Tensor,
        context_indices_1: Tensor | None = None,
        context_indices_2: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Encode both views and return projected embeddings.

        Args:
            spec1: View 1 spectrogram, shape (B, 1, n_mels, n_frames).
            spec2: View 2 spectrogram, shape (B, 1, n_mels, n_frames).
            context_indices_1: Optional mask indices for view 1.
            context_indices_2: Optional mask indices for view 2.

        Returns:
            Tuple of (z1, z2), each shape (B, proj_output_dim).
        """
        z1 = self.encode(spec1, context_indices_1)
        z2 = self.encode(spec2, context_indices_2)
        return z1, z2

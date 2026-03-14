"""Audio Vision Transformer encoder for spectrogram patches."""

from dataclasses import dataclass

import timm
import torch
import torch.nn as nn
from torch import Tensor


@dataclass(frozen=True)
class EncoderConfig:
    img_size: tuple[int, int] = (128, 400)
    patch_size: int = 16
    in_chans: int = 1
    embed_dim: int = 384
    depth: int = 12
    num_heads: int = 6
    use_gradient_checkpointing: bool = False


class AudioEncoder(nn.Module):
    """ViT encoder for audio spectrograms.

    Wraps a timm VisionTransformer with audio-specific configuration.
    Supports optional masking: when context_indices are provided, only those
    patch tokens are processed (plus CLS token).
    """

    def __init__(self, config: EncoderConfig | None = None) -> None:
        super().__init__()
        self.config = config or EncoderConfig()
        self.vit = timm.create_model(
            "vit_small_patch16_224",
            pretrained=False,
            img_size=self.config.img_size,
            patch_size=self.config.patch_size,
            in_chans=self.config.in_chans,
            embed_dim=self.config.embed_dim,
            depth=self.config.depth,
            num_heads=self.config.num_heads,
            num_classes=0,
            class_token=True,
            global_pool="",
        )

        if self.config.use_gradient_checkpointing:
            self.vit.set_grad_checkpointing(enable=True)

    @property
    def embed_dim(self) -> int:
        return self.config.embed_dim

    def forward(self, x: Tensor, context_indices: Tensor | None = None) -> Tensor:
        """Encode spectrogram.

        Args:
            x: Spectrogram tensor of shape (B, 1, n_mels, n_frames).
            context_indices: Optional (N,) indices of visible patches. If provided,
                only those patch tokens are kept before transformer blocks.

        Returns:
            CLS token embedding of shape (B, embed_dim).
        """
        x = self.vit.patch_embed(x)
        cls_token = self.vit.cls_token.expand(x.shape[0], -1, -1)

        if context_indices is not None:
            x = x[:, context_indices]
            pos_embed = self.vit.pos_embed[:, 1:][:, context_indices]
            x = x + pos_embed
            x = torch.cat([cls_token + self.vit.pos_embed[:, :1], x], dim=1)
        else:
            x = torch.cat(
                [cls_token + self.vit.pos_embed[:, :1], x + self.vit.pos_embed[:, 1:]],
                dim=1,
            )

        x = self.vit.pos_drop(x)
        x = self.vit.patch_drop(x)
        x = self.vit.norm_pre(x)
        x = self.vit.blocks(x)
        x = self.vit.norm(x)

        return x[:, 0]

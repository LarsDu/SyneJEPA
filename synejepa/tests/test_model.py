"""Tests for SyneJEPA model components."""

import torch

from synejepa.model.encoder import AudioEncoder, EncoderConfig
from synejepa.model.projection import ProjectionHead
from synejepa.model.synejepa import SyneJEPA


def test_encoder_output_shape():
    config = EncoderConfig(img_size=(128, 400), patch_size=16, embed_dim=384, depth=2, num_heads=6)
    encoder = AudioEncoder(config)
    x = torch.randn(2, 1, 128, 400)
    out = encoder(x)
    assert out.shape == (2, 384)


def test_encoder_with_mask():
    config = EncoderConfig(img_size=(128, 400), patch_size=16, embed_dim=384, depth=2, num_heads=6)
    encoder = AudioEncoder(config)
    x = torch.randn(2, 1, 128, 400)
    context_indices = torch.arange(0, 100)  # 50% of 200 patches
    out = encoder(x, context_indices=context_indices)
    assert out.shape == (2, 384)


def test_projection_head_output_shape():
    proj = ProjectionHead(input_dim=384, hidden_dim=2048, output_dim=256)
    x = torch.randn(4, 384)
    out = proj(x)
    assert out.shape == (4, 256)


def test_synejepa_forward():
    config = EncoderConfig(img_size=(128, 400), patch_size=16, embed_dim=384, depth=2, num_heads=6)
    model = SyneJEPA(encoder_config=config, proj_hidden_dim=512, proj_output_dim=128)
    spec1 = torch.randn(2, 1, 128, 400)
    spec2 = torch.randn(2, 1, 128, 400)
    z1, z2 = model(spec1, spec2)
    assert z1.shape == (2, 128)
    assert z2.shape == (2, 128)

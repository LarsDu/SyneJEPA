"""Tests for random patch masking."""

import torch

from synejepa.audio.masking import RandomPatchMasker


def test_masker_output_shapes():
    masker = RandomPatchMasker(n_mels=128, n_frames=400, patch_size=16)
    ctx, tgt = masker()
    total = len(ctx) + len(tgt)
    assert total == masker.n_patches


def test_masker_ratio_range():
    masker = RandomPatchMasker(
        n_mels=128, n_frames=400, patch_size=16,
        mask_ratio_min=0.4, mask_ratio_max=0.6,
    )
    ratios = []
    for _ in range(100):
        _, tgt = masker()
        ratios.append(len(tgt) / masker.n_patches)
    avg_ratio = sum(ratios) / len(ratios)
    assert 0.35 < avg_ratio < 0.65


def test_masker_no_overlap():
    masker = RandomPatchMasker()
    ctx, tgt = masker()
    ctx_set = set(ctx.tolist())
    tgt_set = set(tgt.tolist())
    assert ctx_set.isdisjoint(tgt_set)


def test_masker_deterministic_with_generator():
    masker = RandomPatchMasker()
    rng1 = torch.Generator().manual_seed(42)
    rng2 = torch.Generator().manual_seed(42)
    ctx1, tgt1 = masker(rng=rng1)
    ctx2, tgt2 = masker(rng=rng2)
    assert torch.equal(ctx1, ctx2)
    assert torch.equal(tgt1, tgt2)

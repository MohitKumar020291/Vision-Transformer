"""
Unit tests for vitm.module.PatchEmbed

To run doctest:
    pytest --doctest-modules vitm/models/patch_embed.py
"""

import torch
import pytest
from vitm.models import PatchEmbed



def test_patch_embed_shape():
    """
    Test PatchEmbed output shape with standard config.
    """
    patch_embed = PatchEmbed(patch_size=8, in_channels=3, embed_dim=768)
    x = torch.randn(1, 3, 224, 224)
    y = patch_embed(x)
    assert y.shape == torch.Size([1, 784, 768])



def test_patch_embed_flatten_false():
    """
    Test PatchEmbed with flatten=False returns shape (B, C, H', W').
    """
    model = PatchEmbed(image_size=32, patch_size=8, in_channels=3, embed_dim=32, flatten=False)
    x = torch.randn(1, 3, 32, 32)
    out = model(x)
    assert out.shape == torch.Size([1, 32, 4, 4])  # 32 channels, 4x4 grid


def test_patch_embed_wrong_input_size():
    """
    Test input shape mismatch triggers assertion.
    """
    model = PatchEmbed(image_size=32, patch_size=8)
    x = torch.randn(1, 3, 30, 30)
    with pytest.raises(AssertionError):
        _ = model(x)
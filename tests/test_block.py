"""
Unit tests for vitm.models.models.Block

To run doctest:
    pytest --doctest-modules tests/test_block.py
"""


import torch
from vitm.models.models import Block

def test_block():
    block = Block(dim=768, num_heads=12, mlp_ratio=4.0)
    x = torch.randn(8, 197, 768)
    out = block(x)
    assert(out.shape == x.shape)
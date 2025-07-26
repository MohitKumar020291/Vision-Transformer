"""
Unit tests for vitm.models.classifier.VITC

To run doctest:
    pytest --doctest-modules tests/test_vitc.py
"""


import pytest
import torch
import torch.nn as nn
from vitm.models.classifier import VITC
from vitm.models.models import VisionTransformer
from vitm.data.dataloader import loadData

def test_vitc():
    image_size, num_classes, trainloader = loadData(batch_size=8, resize=32)
    norm_layer = nn.LayerNorm
    vision_transformer = VisionTransformer(
        image_size,
        patch_size=4,
        num_classes=num_classes,
        num_heads=8,
        mlp_ratio=0.8,
        norm_layer=norm_layer,
        embed_norm_layer=norm_layer,
        final_norm_layer=norm_layer)
    vitc = VITC(vit_model=vision_transformer)
    outputs = None
    for i, batch in enumerate(trainloader, 0):
        inputs, label = batch
        outputs = vitc(inputs)
        break
    assert (outputs.shape == torch.Size((inputs.shape[0], num_classes)))
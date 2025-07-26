# VisionTransformer

::: vitm.models.models.VisionTransformer
    options:
      show_source: true
      show_signature: true

```python
>>> norm_layer = nn.LayerNorm
>>> vision_transformer = VisionTransformer(
    ... image_size,
    ... patch_size=4,
    ... num_classes=num_classes,
    ... num_heads=8,
    ... mlp_ratio=0.8,
    ... norm_layer=norm_layer,
    ... embed_norm_layer=norm_layer,
    ... final_norm_layer=norm_layer)
>>> x = torch.randn(8, 3, 224, 224)
>>> out = vision_transformer(x)
>>> out.shape
```
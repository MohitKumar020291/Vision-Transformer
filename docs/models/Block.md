# Block

::: vitm.models.models.Block
    options:
        show_source: True
        show_signature: True

```python
>>> block = Block(dim=768, num_heads=12, mlp_ratio=4.0)
>>> x = torch.randn(8, 197, 768)
>>> out = block(x)
>>> out.shape  # (8, 197, 768)

```
# ParallelBlock

::: vitm.models.parallel_block.ParallelBlock
    options:
        show_source: true
        show_signature: true

```python
>>> import torch
>>> import torch.nn as nn
>>> from vitm.models.ParallelBlock import ParallelBlock

>>> dim = 256
>>> num_heads = 8
>>> seq_len = 196  # e.g., 14x14 patches
>>> parallel_block = ParallelBlock(
...     dim=dim,
...     num_heads=num_heads,
...     mlp_ratio=0.8,
...     act_layer=nn.GELU,
...     mlp_bias=False,
...     vit=True)

>>> x = torch.randn(8, seq_len, dim)  # (B, N, D)
>>> out = parallel_block(x)
>>> out.shape
torch.Size([8, 196, 256])
```
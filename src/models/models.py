"""
Implementation of vision transformer from hugging face image model's vision transformer
"""

import torch
import torch.nn as nn
from torch.profiler import record_function

from typing import Type, Union, Tuple, Optional, Callable
from functools import partial
from itertools import repeat

from .patch_embeddings import PatchEmbed
from utils.helper import to_2tuple


class Attention(nn.Module):
  def __init__(
      self,
      dim: int,
      num_heads: int = 8,
      qkv_bias: bool = False,
      fused_attention: bool = False,
      qk_norm: bool = False,
      proj_bias = True,
      norm_layer = nn.LayerNorm
      ):
    super().__init__()
    dim % num_heads == 0, "dim should be divisible by num_heads"
    self.dim = dim
    self.num_heads = num_heads
    self.head_dim = self.dim // self.num_heads
    self.qkv_bias = qkv_bias
    self.fused_attention = fused_attention
    self.qk_norm = qk_norm
    self.scale = self.head_dim ** -0.5

    # This is a genius idea - rather than creating wq, wk, wv
    # - we still have a learnable weights
    # I learned this from Hugging face
    self.qkv = nn.Linear(dim, dim * 3, bias = self.qkv_bias)

    # nn.Indentity makes it learnable - I hope so
    self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
    self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()

    self.proj = nn.Linear(dim, dim, bias=proj_bias)

    self._init_weights()

  def _init_weights(self):
    # QKV projection
    nn.init.trunc_normal_(self.qkv.weight, std=0.02)
    if self.qkv.bias is not None:
      nn.init.zeros_(self.qkv.bias)
    
    # Output projection
    nn.init.trunc_normal_(self.proj.weight, std=0.02)
    if self.proj.bias is not None:
      nn.init.zeros_(self.proj.bias)

  def forward(self, x: torch.Tensor):
    B, N, C = x.shape
    # B, N, 3, self.num_heads, self.head_dim -> 3, B, self.num_heads, N, self.head_dim
    with record_function("-------LINEAR-------"):
        qkv = self.qkv(x)
        print(x.shape, qkv.shape, self.qkv.weight.shape)
    # with record_function("-------RESHAPE+PERMUTE-------"):
    qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    # with record_function("-------UNBIND+NORM-------"):
    q, k, v = qkv.unbind(0) # B, self.num_heads, N, self.head_dim
    q, k = self.q_norm(q), self.k_norm(k)

    if self.fused_attention:
      ...
    else:
      # with record_function("ATTENTION"):
          q = q * self.scale # another smart way
          attn = q @ k.transpose(-2, -1)
          attn = attn.softmax(dim=-1)
          x = attn @ v

    x = x.transpose(1, 2).reshape(B, N, C)
    x = self.proj(x)
    return x

# This is for Vision Transformers only
class MLP(nn.Module):
  def __init__(
      self,
      in_features,
      hidden_features: int = None,
      out_features: int = None,
      use_conv: bool = False,
      bias: bool = False,
      act_layer: Type[nn.Module] = nn.GELU,
      norm_layer = None,
      vit: bool = False
      ):

    super().__init__()
    self.vit = vit
    self.in_features = in_features
    # self.hidden_features = self.in_features or hidden_features
    self.hidden_features = hidden_features if hidden_features is not None else self.in_features * 4
    self.out_features = self.in_features or out_features

    self.bias = to_2tuple(bias)
    self.linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

    self.linear_layer1 = self.linear_layer(self.in_features, self.hidden_features, bias = self.bias[0])
    self.ac1 = act_layer()
    # self.layer_norm = norm_layer(self.hidden_features) if norm_layer is not None else nn.Identity()
    self.linear_layer2 = self.linear_layer(self.hidden_features, self.out_features, bias = self.bias[1])

  def forward(self, x):
    x = self.linear_layer1(x)
    x = self.ac1(x)
    x = self.linear_layer2(x)
    return x

class LayerScale(nn.Module):
  def __init__(self, dim: int, init_values: float = 1e-5, inplace = False):
    super().__init__()
    self.gamma = nn.Parameter(torch.ones(dim) * init_values)
    self.inplace = inplace

  def forward(self, x: torch.Tensor):
    return x.mul_(self.gamma) if self.inplace else x * self.gamma


class Block(nn.Module):
  """
  A Block is a transformer-block
  """
  def __init__(
      self,
      dim: int,
      num_heads: int,
      mlp_ratio: float,
      norm_layer: Type[nn.Module] = nn.LayerNorm,
      act_layer: Type[nn.Module] = nn.GELU,
      mlp_layer: MLP = MLP,
      vit: bool = False
      ):
    super().__init__()
    self.dim = dim

    self.norm1 = norm_layer(dim)
    self.attn = Attention(
        dim=dim,
        num_heads=num_heads,
        norm_layer=norm_layer
        )
    self.ls1 = LayerScale(dim=dim)
    self.norm2 = norm_layer(dim)
    act_layer = act_layer
    self.mlp = MLP(
        in_features=dim,
        hidden_features=int(dim*mlp_ratio),
        bias=True,
        norm_layer=norm_layer,
        act_layer=act_layer,
        vit=vit
        )
    self.ls2 = LayerScale(dim=dim)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = x + self.ls1(self.attn(self.norm1(x)))  # Residual around attention
    x = x + self.ls2(self.mlp(self.norm2(x)))   # Residual around MLP
    return x

# Not handling cnns for downsampling now - using DViT
class VisionTransformer(nn.Module):
  def __init__(
      self,
      image_size: Union[int, Tuple[int, int]],
      patch_size: int,
      num_classes: int,
      num_heads: int,
      mlp_ratio: float,
      depth: int = 12,
      embed_dim: int = 768,
      block_fn = Block,
      norm_layer: Optional[nn.Module] = None,
      mlp_layer: MLP = MLP,
      pos_embed: str = 'learn',
      embed_norm_layer: Optional[nn.Module] = None,
      embed_layer: Optional[nn.Module] = PatchEmbed,
      class_token: bool = True,
      final_norm_layer: Optional[nn.Module] = None,
      act_layer: Optional[nn.Module] = nn.GELU,
      ):

    super().__init__()
    self.embed_dim = embed_dim
    self.num_classes = num_classes
    embed_args = {} # I liked this idea
    if embed_norm_layer is not None:
      embed_args['norm_layer'] = embed_norm_layer
    self.image_size = image_size
    self.patch_size = patch_size
    self.patch_embed = embed_layer(
        self.image_size,
        self.patch_size,
        **embed_args
    )
    embed_len = self.patch_embed.num_patches

    self.class_token = None
    if class_token is not None:
        self.class_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.class_token, std=0.02)
      
    pos_embed_len = embed_len + 1 if self.class_token is not None else embed_len

    self.pos_embed = None
    if pos_embed == "learn":
        self.pos_embed = nn.Parameter(torch.zeros(1, pos_embed_len, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    self.norm_pre = norm_layer(embed_dim) if norm_layer else nn.Identity()
    self.blocks = nn.Sequential(
        *[
            block_fn(
                embed_dim,
                num_heads,
                mlp_ratio,
                norm_layer,
                act_layer,
                mlp_layer,
                vit=True
              )
            for _ in range(depth)
        ]
    )
    self.norm_final = final_norm_layer(embed_dim) if final_norm_layer else nn.Identity()
    self.feature_info = [
        dict(module=f'block_{i}', num_chs=embed_dim)
        for i in range(depth)
    ]
    self._init_weights()

  # This is for the dynamic tokenizer - will look into it after sometime
  def _pos_embed(self, x: torch.Tensor):
    if self.pos_embed is None:
      return x

  def _init_weights(self):
    for m in self.modules():
      if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
      elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
      elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)

  def forward_features(self, x):
    with profiler.record_function("MASK INDICES"):
      x = self.patch_embed(x)
    B, N, D = x.shape
    if self.class_token is not None:
        class_token = self.class_token.expand(B, -1, -1)
        x = torch.cat((class_token, x), dim=1)
    x = self.norm_pre(x)
    x = x + self.pos_embed
    return x

  def forward_heads(self, x: torch.Tensor):
    x = self.blocks(x)
    x = self.norm_final(x)
    return x

  def forward(self, x: torch.Tensor):
    x = self.forward_features(x)
    x = self.forward_heads(x)
    return x

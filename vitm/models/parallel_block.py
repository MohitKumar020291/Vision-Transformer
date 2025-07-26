import torch
import torch.nn as nn
from typing import Type


class LayerScale(nn.Module):
  """
    Applies a learnable scaling factor to the input tensor.

    Typically used after residual connections in Transformer blocks to stabilize training,
    especially for very deep models (e.g., >24 layers in ViTs).

    Args:
        dim (int): Dimensionality of the input and scaling vector.
        init_values (float): Initial value for scaling parameters (default: 1e-5).
        inplace (bool): If True, performs the operation in-place (default: False).

    Attributes:
        gamma (nn.Parameter): Learnable scaling parameters of shape (dim,).
    """
  def __init__(self, dim: int, init_values: float = 1e-5, inplace = False):
    super().__init__()
    self.gamma = nn.Parameter(torch.ones(dim) * init_values)
    self.inplace = inplace

  def forward(self, x: torch.Tensor):
    return x.mul_(self.gamma) if self.inplace else x * self.gamma


class ParallelBlockMultiHeadAttention(nn.Module):
    """
    A multi-head attention module with QK normalization and parallel processing.

    This module expects Q, K, V to be fused into a single tensor and handles
    Q/K normalization to improve training stability (especially in large ViTs).

    Args:
        dim (int): Total embedding dimension of the input.
        num_heads (int): Number of attention heads.
        norm_layer (Type[nn.Module]): Normalization layer used for Q and K (default: nn.LayerNorm).

    Attributes:
        q_norm (nn.Module): Normalization applied to the query tensor.
        k_norm (nn.Module): Normalization applied to the key tensor.
        proj (nn.Linear): Linear projection applied after attention.

    Forward Input:
        QKV (torch.Tensor): Fused input tensor of shape (B, N, 3 * dim),
                            where Q, K, V are concatenated along the last dimension.

    Forward Output:
        x (torch.Tensor): Output tensor of shape (B, N, dim).
    """
    def __init__(
            self,
            dim,
            num_heads,
            norm_layer: Type[nn.Module] = nn.LayerNorm
            ):
        super().__init__()
        assert dim % num_heads == 0, "dim should be multiple of num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q_norm = norm_layer(self.head_dim)
        self.k_norm = norm_layer(self.head_dim)

        self.proj = nn.Linear(in_features=dim, out_features=dim)

    def forward(self, QKV):
        B, N, C = QKV.shape
        assert C // 3 == self.dim, f"Inner dim of QKV {C // 3} != dim = {self.dim}"
        # B, N, 3, num_heads, head_dim => 3, B, num_heads, N, head_dim
        # Is it bad to use view here?
        QKV = QKV.view(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = QKV.unbind(0)

        # These excessively large logits, after being passed through a softmax function, 
        # led to near-zero entropy attention weights that were "almost one-hot"
        # Prior to QK Normalisation, it was often necessary to reduce the learning rate for larger ViT 
        # models (e.g., from 1e-3 down to 4e-4 for ViT-H) to maintain stability.
        # With QK Normalisation, ViT-22B can maintain a higher and consistent learning rate of 1e-3
        q, k = self.q_norm(q), self.k_norm(k)

        attn = torch.matmul(q, k.permute(0, 1, 3, 2))
        attn = attn * self.head_dim ** -0.5
        attn = attn.softmax(dim=-1)
        x = attn @ v
        x = x.permute(0, 2, 1, 3).flatten(-2)
        x = self.proj(x)
        return x


class MLP(nn.Module):
    """
    A lightweight MLP used in Transformer-style blocks.

    This MLP applies an activation function followed by a linear projection.
    It expects that the input has already been projected to the hidden dimension.

    Args:
        mlp_hidden_features (int): Size of the hidden representation.
        out_features (int): Output dimensionality, typically same as input to the block.
        act_layer (Type[nn.Module]): Activation function (default: nn.GELU).
        bias (bool): Whether to include a bias term in the output projection (default: True).

    Attributes:
        act_layer (nn.Module): Activation function.
        out_proj (nn.Linear): Output projection layer.
    """
    def __init__(
            self,
            mlp_hidden_features: int,
            out_features: int,
            act_layer: Type[nn.Module]=nn.GELU,
            bias: bool=True
            ):
        super().__init__()
        self.act_layer = act_layer()
        self.out_proj = nn.Linear(in_features=mlp_hidden_features, out_features=out_features, bias=bias)

    def forward(self, x):
        x = self.act_layer(x)
        x = self.out_proj(x)
        return x


class ParallelBlock(nn.Module):
    """
    A fused Transformer block with parallel attention and MLP streams.
    Implemented from the paper: https://arxiv.org/abs/2302.05442  
    This block performs:  
      - LayerNorm on input.   
      - A fused projection that creates Q, K, V and MLP hidden features simultaneously.    
      - Multi-head attention with normalized Q/K.    
      - A lightweight MLP.   
      - Concatenation of attention and MLP outputs.   
      - Final projection and optional LayerScale before a residual addition.   

    This is optimized for compute efficiency and convergence in large-scale ViTs.   

    Args:   
        dim (int): Embedding dimension.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Fraction of `dim` to use for the MLP hidden layer (default: 0.7).
        act_layer (Type[nn.Module]): Activation function used in the MLP (default: nn.GELU).
        mlp_bias (bool): Whether to include a bias in the MLP output (default: False).
        vit (bool): If True, assumes ViT-style behavior (currently unused, reserved for extensions).

    Attributes:
        norm (nn.LayerNorm): Normalization applied before the fused projection.
        fused_proj_1 (nn.Linear): Projects input to QKV + MLP hidden features.
        attn (ParallelBlockMultiHeadAttention): Multi-head self-attention module.
        mlp (MLP): Lightweight feed-forward network.
        fused_proj_2 (nn.Linear): Projects concatenated attention + MLP outputs back to `dim`.
        layer_scale (LayerScale): Optional learnable scaling before residual addition.

    Forward Input:
        `x (torch.Tensor)`: Input tensor of shape (B, N, dim)

    Forward Output:
        `output (torch.Tensor)`: Output tensor of shape (B, N, dim)
    """
    def __init__(
            self, 
            dim, 
            num_heads,
            mlp_ratio: float = 0.7,
            act_layer: Type[nn.Module] = nn.GELU, # Will discuss what nn.GELU is
            mlp_bias: bool = False,
            vit: bool = True
            ):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        assert dim % num_heads == 0, "dim should be multiple of num_heads"
        self.head_dim = dim // num_heads
        self.mlp_hidden_dim = int(dim * mlp_ratio)
        out_fusion_1 = dim * 3 + self.mlp_hidden_dim

        # modules
        self.norm = nn.LayerNorm(dim)
        self.fused_proj_1 = nn.Linear(in_features=dim, out_features=out_fusion_1)
        self.attn = ParallelBlockMultiHeadAttention(dim=self.dim, num_heads=num_heads)
        self.mlp = MLP(
            mlp_hidden_features=self.mlp_hidden_dim,
            out_features=dim,
            act_layer=act_layer,
            bias=mlp_bias
        )
        self.fused_proj_2 = nn.Linear(in_features=2*dim, out_features=dim)
        self.layer_scale = LayerScale(dim)

 
    def forward(self, x):
        _, _, C = x.shape
        assert C == self.dim, "inner dimension of x should be equal to the \
           out_features of fused_linear_layer1"

        norm_x = self.norm(x)

        fused_proj_1 = self.fused_proj_1(norm_x)
        
        qkv, mlp_hidden = torch.split(
            fused_proj_1, 
            [3*self.dim, self.mlp_hidden_dim], 
            dim=-1
        )

        attn_out = self.attn(qkv)
        mlp_out = self.mlp(mlp_hidden)

        combined = torch.cat([attn_out, mlp_out], dim=-1)
        fused_proj_2 = self.fused_proj_2(combined)

        output = x + self.layer_scale(fused_proj_2)

        print(f"Output shape: {output.shape}, Expected shape: {(x.shape[0], x.shape[1], self.dim)}")
        return output


if __name__ == "__main__":
    B, N, C = 4, 1024, 4096
    num_heads = 4
    x = torch.randn((B, N, C))
    # block = ParallelBlock(dim=C, num_heads=num_heads, mlp_bias=True)
    # output = block(x)
    # assert output.shape == x.shape
    from vitm.models.models import VisionTransformer
    model = VisionTransformer(
        image_size=224,
        patch_size=16,
        num_classes=1000,
    )
    output = model(x)
    assert output.shape == (B, N, C), f"Output shape {output.shape} does not match expected shape {(B, N, C)}"
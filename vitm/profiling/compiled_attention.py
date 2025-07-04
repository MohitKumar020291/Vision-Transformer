import torch
import torch.nn as nn
from torch.library import triton_op, wrap_triton


# I thinks writing these function individually is best choice for kernelizing
class AttentionCompiled(nn.Module):
    def __init__(
            self, 
            dim,
            num_heads,
            proj_bias=False
        ):
        super().__init__()
        assert dim % num_heads==0, "dim has to be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // self.num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.scale = self.head_dim ** -0.5

        self.proj_bias = proj_bias
        self.proj = nn.Linear(dim, dim, bias=proj_bias)

    @staticmethod
    def naive_softmax(x: torch.Tensor, dim: int = -1):
        x_max = x.max(dim=dim, keepdims=True).values #for stability: max of each embedding
        x_stable = x - x_max
        exp_x = torch.exp(x_stable)

        sum_exp = exp_x.sum(dim=dim, keepdims=True)
        softmax = exp_x / sum_exp

        return softmax

    @staticmethod
    def naive_layer_norm(x: torch.Tensor):
        """
            Similar to, 
            >>> norm_layer = nn.LayerNorm(head_dim, dtype=x.dtype, device=x.device)
            >>> x_nnln = norm_layer(x)
            >>> x_nnln.allclose(x_normalized_wlp, atol=1e-7)
            True

            Just for my revision:
            abs(input - other) <= atol + rtol * other
        """
        # --NAIVE IMPLEMENTATION--
        eps = 1e-5
        mean = x.mean(-1, keepdims=True) #mean of each embedding
        var = x.var(-1, keepdims=True, unbiased=False) #unbiased=False => population variance
        x_normalized = (x - mean) / torch.sqrt(var + eps)

        # Learnable parameters
        gamma = torch.ones(x.size(-1), dtype=x.dtype, device=x.device)
        beta = torch.zeros(x.size(-1), dtype=x.dtype, device=x.device)

        x_normalized_wlp = gamma * x_normalized + beta #Right now we can see the shifted version of mean!=0, mean = 1
        return x_normalized_wlp

    def forward(self, x):
        B, N, C = x.shape
        qkv = x @ self.qkv.weight.T + (self.qkv.bias if self.qkv.bias is not None else 0)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0) #B, num_heads, N, head_dim
        q, k = self.naive_layer_norm(q), self.naive_layer_norm(k) #B, num_heads, N, head_dim
        q = q * self.scale #B, num_heads, N, head_dim
        attn = q @ k.transpose(-2, -1) #(B, num_heads, N, head_dim) @ (B, num_heads, head_dim, N) = (B, num_heads, N, N)
        attn = self.naive_softmax(attn) #B, num_heads, N, N => each embedding interacting will all the embeddings of it's head, the embedding_dim = head_dim
        x = attn @ v #B, num_heads, N, head_dim
        x = x.transpose(1, 2).reshape(B, N, C) #B, N, num_heads, head_dim => B, N, C, C = num_heads * head_dim
        x = x @ self.proj.weight.T + (self.proj.bias if self.proj_bias else 0)
        return x
    

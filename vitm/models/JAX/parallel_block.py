import jax
import jax.numpy as jnp
import time

from jax import random
from flax import linen as nn
from typing import Any

# --- Timing Decorator
def benchmark_fn(fn):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = fn(*args, **kwargs)
        jax.block_until_ready(result)  # Ensure sync before timing ends
        end = time.time()
        print(f"⏱️ {fn.__name__} took {end - start:.6f} seconds")
        return result
    return wrapper

# --- Per-head normalization
class HeadNorm(nn.Module):
    head_dim: int
    def setup(self):
        self.ln = nn.LayerNorm()

    def __call__(self, x):
        return self.ln(x)

# --- Multi-head Attention branch
class ParallelBlockMultiHeadAttention(nn.Module):
    dim: int
    num_heads: int

    def setup(self):
        self.head_dim = self.dim // self.num_heads
        self.q_norm = HeadNorm(self.head_dim)
        self.k_norm = HeadNorm(self.head_dim)

    def __call__(self, QKV):
        B, N, C = QKV.shape
        QKV = QKV.reshape(B, N, 3, self.num_heads, self.head_dim)
        QKV = jnp.transpose(QKV, (2, 0, 3, 1, 4))  # (3, B, H, N, D)
        q, k, v = QKV[0], QKV[1], QKV[2]
        q = self.q_norm(q)
        k = self.k_norm(k)

        attn = jnp.einsum('bhqd,bhkd->bhqk', q, k) * (self.head_dim ** -0.5)
        attn = nn.softmax(attn, axis=-1)
        x = jnp.einsum('bhqk,bhkd->bhqd', attn, v)
        return x

# --- Main Block
class ParallelBlock(nn.Module):
    dim: int
    num_heads: int
    mlp_ratio: float = 0.7

    def setup(self):
        self.head_dim = self.dim // self.num_heads
        self.mlp_hidden_dim = int(self.dim * self.mlp_ratio)

        self.norm = nn.LayerNorm()
        self.fused_proj1 = nn.Dense(self.dim * 3 + self.mlp_hidden_dim)
        self.attn = ParallelBlockMultiHeadAttention(self.dim, self.num_heads)
        self.attn_out_proj = nn.Dense(self.dim)
        self.mlp_out_proj = nn.Dense(self.dim)
        self.fused_proj2 = nn.Dense(self.dim)

    @benchmark_fn
    def compute_attn(self, qkv):
        attn_out = self.attn(qkv)
        attn_out = jnp.transpose(attn_out, (0, 2, 1, 3)).reshape(qkv.shape[0], qkv.shape[1], -1)
        return self.attn_out_proj(attn_out)

    @benchmark_fn
    def compute_mlp(self, mlp_hidden):
        return self.mlp_out_proj(nn.gelu(mlp_hidden))

    def __call__(self, x):
        norm_x = self.norm(x)
        fused = self.fused_proj1(norm_x)
        qkv, mlp_hidden = jnp.split(fused, [3*self.dim], axis=-1)

        # attn_out = self.compute_attn(qkv)
        # mlp_out = self.compute_mlp(mlp_hidden)
        attn_out, mlp_out = jax.jit(lambda q, m: (self.compute_attn(q), self.compute_mlp(m)))(qkv, mlp_hidden)

        combined = jnp.concatenate([attn_out, mlp_out], axis=-1)
        fused_out = self.fused_proj2(combined)
        return x + fused_out


@jax.jit
def fused_compute(qkv, mlp_hidden):
    # --- Compute Attention ---
    B, T, _ = qkv.shape
    D = qkv.shape[-1] // 3

    q, k, v = jnp.split(qkv, 3, axis=-1)

    scale = D ** -0.5
    attn_logits = jnp.einsum('btd,bTd->btT', q, k) * scale
    attn_weights = jax.nn.softmax(attn_logits, axis=-1)
    attn_out = jnp.einsum('btT,bTd->btd', attn_weights, v)

    # --- Compute MLP ---
    mlp_out = jax.nn.gelu(mlp_hidden)  # assuming 1-layer GELU
    mlp_out = jnp.dot(mlp_out, jax.random.normal(jax.random.PRNGKey(0), (mlp_hidden.shape[-1], D)))

    # Combine results
    return attn_out, mlp_out


if __name__ == "__main__":
    for i in range(5):
        print(f"Run {i+1}:")
        key1, key2 = random.split(random.PRNGKey(0))
        x = random.normal(key1, (16, 128, 192))  # (Batch, Tokens, Dim)

        # start = time.time()
        model = ParallelBlock(dim=192, num_heads=6)
        # jax.block_until_ready(model)
        # end = time.time()
        # print(f"Model initialization took {end - start:.6f} seconds")

        params = model.init(key2, x)
        out = model.apply(params, x)
        print("✅ Output shape:", out.shape)

import torch
import torch.nn as nn

from Kernels import LayerNormModule

from utils import compare, make_input, getSetDevice

def layer_norm_triton_implementation():
    is_cuda, device = getSetDevice()
    B, num_heads, N, dim, input = make_input(device)
    print(input.shape)
    # models = [
    #             torch.compile(AttentionCompiled(dim, num_heads).to(device)),
    #             Attention(dim, num_heads)
    #         ]

    N, dim, input = \
        make_input(
                device, 
                batch=False, 
                multihead=False
                )
    print(input.shape)

    # Comparing only on N, D
    w_shape = (dim,)
    weight = torch.randn(w_shape, dtype=input.dtype, device=device, requires_grad=True)
    bias = torch.randn(w_shape, dtype=input.dtype, device=device, requires_grad=True)
    input_wg = torch.clone(input)
    input_wg.requires_grad_(True)
    dy = .1 * torch.randn_like(input_wg)
    eps = 1e-5 # Same as naive_layer_norm
    compare(
        input=input_wg,
        models=[\
            LayerNormModule(
                w_shape,
                weight,
                bias,
                eps)
            ],
        compile=[False],
        is_cuda=is_cuda,
        device=device
    )
    
    compare(
        input,
        models = [nn.LayerNorm(dim)],
        compile = [False],
        is_cuda = is_cuda,
        device = device
    )

    compare(
        input,
        models = [nn.LayerNorm(dim)],
        compile = [True],
        is_cuda = is_cuda,
        device = device
    )
import torch
import torch.nn as nn

import triton
import triton.language as tl

# @triton.autotune()
@triton.jit
def Softmaxfwd(
    x, # Input pointer
    y, # output pinter
    stride: tl.constexpr,
    D, # Number of columns, might change BLOCK_SIZE_N, that's why having passing it separately
    BLOCK_SIZE_N: tl.constexpr, # It is just number of columns (768, possible)
):
    row_idx = tl.program_id(0) # For each row
    cols_offs = tl.arange(0, BLOCK_SIZE_N)
    mask = cols_offs < D

    x_row_ptrs = x + row_idx * D + cols_offs
    y_row_ptrs = y + row_idx * D + cols_offs
    x_vals = tl.load(x_row_ptrs, mask=mask).to(tl.float32)

    x_softmax = tl.softmax(x_vals, 0)

    # Pytorch also truncates it
    tl.store(y_row_ptrs, x_softmax.to(tl.float16), mask=mask)

def Softmaxbwd(
        x,
        x_grads_ptr,
):
        x_grads = tl.softmax.backward(x, dim=1) # I am not sure if this is possible
        tl.store(x_grads_ptr, x_grads)

class TritonSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        N, D = x.shape
        y = torch.empty_like(x)
        num_blocks = N
        # Considers a row have the size of 64kb
        MAX_BLOCKS = 65536 // x.element_size() # Handle each element individually
        BLOCK_SIZE_N = min(MAX_BLOCKS, triton.next_power_of_2(D))
        assert D <= BLOCK_SIZE_N
        Softmaxfwd[(num_blocks,)](
            x, y,
            x.stride(0),
            D,
            BLOCK_SIZE_N
            )
        return y

    @staticmethod
    def backward(ctx, dy):
        raise NotImplemented("Not implemented yet")

class SoftmaxModule(nn.Module): 
    def __init__(self):
        super().__init__()
    
    def forward(self, x: torch.Tensor):
        return TritonSoftmax.apply(x)
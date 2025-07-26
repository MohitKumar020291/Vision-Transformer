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
    D, # Number of columns, might change BLOCK_SIZE_N, that's why have to pass it separately
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


@triton.jit
def Softmaxbwd(
    dy, # input stream of gradients
    x, # All rows of softmax
    dx, # output stream of gradients
    D, # Number of columns
    BLOCK_SIZE_N: tl.constexpr,
):
    row_id = tl.program_id(0)
    cols_offset = tl.arange(0, BLOCK_SIZE_N)
    mask = cols_offset < D  

    # pointer arithematic
    dy = dy + row_id * D + cols_offset
    x = x + row_id * D + cols_offset
    dx = dx + row_id * D + cols_offset

    # load
    dy = tl.load(dy + cols_offset, mask=mask).to(tl.float32)
    x = tl.load(x + cols_offset, mask=mask).to(tl.float32)

    # Dot product: sum_j (dy_j * y_j)
    dot = tl.sum(dy * x)

    # Compute gradient: dx_i = y_i * (dy_i - dot)
    dx = x * (dy - dot)

    # Store
    tl.store(dx, dx.to(dy.dtype), mask=mask)


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
        if not x.is_contiguous():
            return "====================THE TENSOR IS NOT CONTIGUOS=========================="
        return TritonSoftmax.apply(x)
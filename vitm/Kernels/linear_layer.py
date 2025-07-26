import triton
import triton.language as tl
import torch
import torch.nn as nn
import numpy as np
import warnings
import os

os.environ['TRITON_INTERPRET'] = 1


@triton.jit
def _linear_layer_fwd(
        x, weights, y,
        BLOCK_SIZE_N: tl.constexpr,
        stride_x: tl.constexpr,
        stride_weights: tl.constexpr,
        stride_y: tl.constexpr,
        current_batch_idx: tl.constexpr,
        num_batch_ele: tl.constexpr,
):

    row_idx = tl.program_id(0)

    cols_offs_x = tl.arange(0, BLOCK_SIZE_N)
    cols_offs_y = tl.arange(0, stride_y)
    cols_offs_weights = tl.arange(0, stride_weights)
    weight_rows = tl.arange(0, BLOCK_SIZE_N)

    mask_x = cols_offs_x < stride_x
    mask_y = cols_offs_y < stride_y
    mask_weights = weight_rows[:, None] < stride_x

    x_row_ptr = x + row_idx * stride_x + cols_offs_x
    y_row_ptr = y + current_batch_idx * num_batch_ele + row_idx * stride_y + cols_offs_y
    weights_ptr = weights + weight_rows[:, None] * stride_weights + cols_offs_weights[None, :]

    x_load = tl.load(x_row_ptr, mask=mask_x)
    weights_load = tl.load(weights_ptr, mask=mask_weights) # all weights have been loaded
   
    x_load = x_load[None, :]
    output = tl.dot(x_load, weights_load)

    y_existing = tl.load(y_row_ptr, mask=mask_y, other=0.0)[None, :]

    tl.store(y_row_ptr, tl.reshape(y_existing + output, [16]), mask=mask_y)


class LinearLayerTriton(torch.autograd.Function):
    @staticmethod
    def forward(
            ctx, 
            x, 
            weights,
            test=False
        ):
        # other specs will be added
        B, N, C = x.shape
        _, out_features = weights.shape

        BLOCK_SIZE_N = triton.next_power_of_2(C)
        num_blocks = N
        stride_x = C
        stride_y = out_features
        num_batch_ele = N * stride_y
        stride_weights = weights.stride(0)

        y = torch.zeros((B, N, out_features), device=x.device, dtype=x.dtype)

        for b in range(B):
            _linear_layer_fwd[(num_blocks,)](
                x[b],
                weights,
                y,
                BLOCK_SIZE_N,
                stride_x,
                stride_weights,
                stride_y,
                b,
                num_batch_ele
            )

        if test:
            assert y.shape == (x @ weights).shape
            print(torch.allclose(y, x @ weights))
        return y
            
    @staticmethod
    def backward(ctx):
        raise NotImplementedError("Backward pass for Linear layer is not implemented")


class LinearLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        # created as in_features, out_features because passing a transpose will lead to 
        # problem in striding
        self.weights = nn.Parameter(torch.randn(in_features, out_features).to(torch.float32))

    def forward(self, x, test=False):
        if not isinstance(x, torch.Tensor):
            warnings.warn(f"input, x should be a torch.Tensor got {type(x)}")
            try:
                if isinstance(x, np.ndarray):
                    x = torch.from_numpy(x).to(torch.float32)
                elif isinstance(x, list):
                    x = torch.stack(x).to(torch.float32)
                # other checks could be done
            except Exception as e:
                raise e
        LinearLayerTriton.apply(
            x,
            self.weights,
            test
        )


if __name__ == "__main__":
    B, N, C = 4, 100, 8
    x = torch.randn(B, N, C)
    l = LinearLayer(in_features=C, out_features=16)
    output = l(x, test=True)

def benchmark_model(model, x, runs=10, name="Model"):
    # Warm-up
    for _ in range(10):
        _ = model(x)

    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(runs):
        _ = model(x)
    end_event.record()

    torch.cuda.synchronize()
    avg_time = start_event.elapsed_time(end_event) / runs  # in milliseconds
    print(f"[Benchmark] {name}: Avg time over {runs} runs: {avg_time:.4f} ms")


if __name__ == "__main__":
    B, N, C = 4, 100, 8
    OUT_FEATURES = 16

    x = torch.randn(B, N, C).cuda()

    # Triton Linear
    triton_linear = LinearLayer(in_features=C, out_features=OUT_FEATURES).cuda()
    output = triton_linear(x, test=True)

    # Torch Linear (for comparison)
    torch_linear = torch.nn.Linear(C, OUT_FEATURES).cuda()
    with torch.no_grad():
        torch_linear.weight.copy_(triton_linear.weights.T)  # match weights
        torch_linear.bias.zero_()  # Triton impl has no bias

    # Benchmark both
    benchmark_model(triton_linear, x, name="Triton Linear")
    benchmark_model(torch_linear, x, name="Torch nn.Linear")
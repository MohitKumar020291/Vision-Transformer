import torch
import torch.profiler as profiler
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler

import os

from vitm.models.parallel_block import ParallelBlockMultiHeadAttention


def cpu_profile_parallel_attention(output_dir="/mnt/c/Users/HP/Desktop/DeepLearning/profiler_output"):
    # Dummy input and model
    torch.set_default_device("cuda")

    B, N, dim = 4, 1024, 4096
    QKV = torch.randn(B, N, dim * 3)
    num_heads = 8
    attn = ParallelBlockMultiHeadAttention(dim, num_heads)

    os.makedirs(output_dir, exist_ok=True)
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]
    steps = 5
    wait=1
    warmup=1
    active=steps-wait-warmup
    schedule=profiler.schedule(
        wait=wait, # for step=1, do nothing
        warmup=warmup, # for step=2, warm-up => record, but do nothing
        active=active # for step=3 (steps=5-2), record + save trace
    )
    with profile(
        activities=activities,
        schedule=schedule,
        on_trace_ready=tensorboard_trace_handler(output_dir),
        with_stack=True,
        record_shapes=False,
        profile_memory=False, # Disable to reduce overhead
        with_flops=True
        ) as prof:
        for i in range(steps):
            with torch.profiler.record_function(f"ATTN_FWD_STEP_{i}"):
                _ = attn(QKV)
            torch.cuda.synchronize()
            prof.step()

    print(prof.key_averages(group_by_stack_n=5).table(sort_by="cuda_time_total", row_limit=20))


if __name__ == "__main__":
    cpu_profile_parallel_attention()
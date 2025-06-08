# File contains benchmarkig functions
# Will add more functions (if needed)

import torch
import torch.nn as nn
import torch.profiler as profiler
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler

from typing import Optional, Dict

from utils.helper import getSetDevice


def benchmark(
    input: torch.Tensor,
    model: nn.Module,
    pass_model_args: Dict = None, # for benchmarking the triton kernels
    record_func_name: str = "BENCH",
    compile: bool = True,
    save_model: bool = True,
    output_dir="/mnt/c/Users/HP/Desktop/DeepLearning/profiler_output",
    file_name=None, # Use this to change the output file name 
    on_cuda=True,
    steps=3,
    wait=1,
    warmup=1,
    profiling_config: Optional[Dict] = None
    ) -> profile:
    """
        model: A initialized subclass of nn.Module model=NN(dim)
        input: Pass the input according to the model
        pass_model_args: if using then pass the input tensor for the forward here
    """
    is_cuda, device = getSetDevice()
    if not is_cuda and on_cuda:
        print("PROFILING STOPPED NO CUDA DEVICE FOUND")
        return

    print("PROFILING ON CUDA") if on_cuda else print("PROFILING ON CPU")

    model = model.to(device) #model should be on device
    if compile:
        print("----USING torch.compile----")
        model = torch.compile(model)

    activities=[ProfilerActivity.CPU]
    if on_cuda:
        activities.extend([ProfilerActivity.CUDA])

    active=steps - (wait + warmup)
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
            with profiler.record_function(f"{record_func_name}_{i}"):
                try:
                    _ = model(*pass_model_args) if pass_model_args else model(input)
                    torch.cuda.synchronize()
                except Exception as e:
                    raise ValueError(f"ERROR during profiling from model: \n {e}") #pass the model number also (only for multiple models)
            prof.step()
    return prof


def compare_models(
        models: list[nn.Module],
        config_benchmark: dict,
        print_bench: bool= True
    ) -> None:
    """
        model passed in config_benchmark will not be considered
    """
    for idx, model in enumerate(models):
        print(model)
        # Some times we may just want to benchmark model1 or model2
        if model:
            config_benchmark["model"] = model
            config_benchmark_ = config_benchmark.copy()
            config_benchmark_["compile"] = config_benchmark_["compile"][idx]
            print(config_benchmark_["compile"])
            prof = benchmark(**config_benchmark_)
            if print_bench:
                print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10)) if print_bench else print()
        else:
            print(f"-------------models[{idx}] is None, not a problem if did intentionally!-------------")
    return
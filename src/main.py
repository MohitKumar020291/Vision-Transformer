import torch
import torch.nn as nn

import yaml 
import os
import argparse
from typing import Dict

from training.train import train_model
from profiling.attention import cpu_profile_attention
from models import Attention
from profiling import AttentionCompiled
from profiling import LayerNormModule
from profiling import compare_models
from utils import getSetDevice


def load_config(path="config/config.yaml"):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, path)
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# pass this into the helper.py
def CLI_args():
    parser = argparse.ArgumentParser()
    # provide_full_yaml 
    parser.add_argument('--pfy', type=str, required=False)
    parser.add_argument('--pathpfy', type=str, required=False)
    args = vars(parser.parse_args())
    if args.get("pfy") == 'True':
        if args.get("pathpfy") is None:
            raise ValueError(f"--pfy is True, but --pathpfy is missing. Received: pfy={args.get('pfy')}, pathpfy={args.get('pathpfy')}") #Is value error a correct error for absence of the path?
    return args


def make_input(
        device: torch.cuda.device, 
        batch = True,
        multihead = True
        ):
    """Has to be deleted"""
    B, N, dim = 32, 100, 768

    input = torch.randn(B, N, dim).to(device)
    num_heads = 8
    head_dim = dim // 8
    shape = []

    if batch:
        shape.append(B) #[B]
    shape.append(N) #[B,N]
    if multihead:
        shape.extend([num_heads, head_dim]) #[B,N,num_heads,head_dim]
    else:
        shape.append(dim) #[B,N,dim]

    shape = tuple(shape)
    if len(shape)==4:
        input = input.reshape(shape)
        input = input.permute(0, 2, 1, 3)
    return *shape, input


def compare(
        input: torch.Tensor,
        models: list[nn.Module],
        compile: list,
        is_cuda: bool,
        device: torch.cuda.device,
        pass_model_args: Dict = None,
):
    """The function to appropriately call compare_models"""
    output_dir = "/mnt/c/Users/HP/Desktop/DeepLearning/profiler_output"

    for model in models:
        # Read the reason in src/profiling/benchmark_fns/compare_models
        if model:
            model.to(device)
    config_benchmark = {
        "pass_model_args": pass_model_args if pass_model_args else None,
        "input": input,
        "model": None, #to be skipped: just pass it here
        "record_func_name": "BENCH_LAYER_NORM",
        "compile": compile or [True] * len(models)
    }
    compare_models(
        models=models,
        config_benchmark=config_benchmark
    )

if __name__ == "__main__":
    # Define a standard config
    # Use this to generate a new standard config
    # print(CLI_args())

    # config = load_config()
    # train_model(config)

    # cpu_profile_attention()

    benchmark = True
    if benchmark:
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
            models=[LayerNormModule(
                w_shape,
                weight,
                bias,
                eps
                )],
            compile=[False],
            is_cuda=is_cuda,
            device=device
        )
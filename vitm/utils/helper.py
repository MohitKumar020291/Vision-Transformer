import torch
import torch.nn as nn

import collections.abc
from itertools import repeat
from enum import Enum
from typing import Union, List, Tuple, Dict
import yaml 
import os

from vitm.profiling import compare_models

#Any x will be repeted n times if it is not iterable
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse

to_2tuple = _ntuple(2)
# to_2tuple(4)


class Format(str, Enum):
    NCHW = 'NCHW'
    NHWC = 'NHWC'
    NCL = 'NCL'
    NLC = 'NLC'

def nchw_to(x: torch.Tensor, fmt: Format) -> torch.Tensor:
    if fmt == "NHWC":
        x = x.permute(0, 2, 3, 1)
    elif fmt == Format.NLC:
        x = x.flatten(2).transpose(1, 2)
    elif fmt == Format.NCL:
        x = x.flatten(2)
    return x


# functions to show an image
def imshow(img):
    import numpy as np
    import matplotlib.pyplot as plt
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


def getParam(config, keys: Union[str, List[str]]):
    if not isinstance(keys, str) and not (isinstance(keys, list) and isinstance(keys[0], str)): #though this does not checks everything
        print(keys)
        return ValueError("key is supposed to be a str or a list of strings")
    if isinstance(keys, str):
        keys = [keys]
    else:
        keys = keys
    
    return_obj = config
    config_str = "config"
    error_msg = None
    for idx, key_ in enumerate(keys):
        config_str += "[" + key_ + "]"
        return_obj = return_obj.get(key_, None)
        if return_obj == None:
            error_msg = f"{config_str} do not exists - perhaps {key_} is a problem!"
            return None, config_str, error_msg
    return return_obj, config_str, error_msg


def getSetDevice() -> Tuple[bool, torch.device]:
    cuda_is_available = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_is_available else "cpu")

    return cuda_is_available, device


def make_input(
        device: torch.cuda.device, 
        batch = True,
        multihead = True
        ):
    """Has to be deleted"""
    B, N, dim = 32, 100, 768

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
    input = torch.randn(shape, dtype=torch.float16, device=device)
    if len(shape)==4:
        input = input.reshape(shape)
        # This makes the tensor non-contiguous and then reshape have to create a whole new tensor, copying it.
        input = input.permute(0, 2, 1, 3)
        while not input.is_contiguous():
            input = input.contiguous()
    return *shape, input


# pass this into the helper.py
# def CLI_args():
#     parser = argparse.ArgumentParser()
#     # provide_full_yaml 
#     parser.add_argument('--pfy', type=str, required=False)
#     parser.add_argument('--pathpfy', type=str, required=False)
#     args = vars(parser.parse_args())
#     if args.get("pfy") == 'True':
#         if args.get("pathpfy") is None:
#             raise ValueError(f"--pfy is True, but --pathpfy is missing. Received: pfy={args.get('pfy')}, pathpfy={args.get('pathpfy')}") #Is value error a correct error for absence of the path?
#     return args



def load_config(path="config/config.yaml"):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, path)
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

return_type_compare = Union[Tuple[List[torch.Tensor], torch.profiler.profile], torch.profiler.profile]
def compare(
        input: torch.Tensor,
        models: list[nn.Module],
        compile: list,
        is_cuda: bool,
        device: torch.cuda.device,
        pass_model_args: Dict = None,
        record_func_name: str = None,
        return_tensor: bool = False
) -> return_type_compare:
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
        "record_func_name":  record_func_name or "BENCH",
        "compile": compile or [True] * len(models),
        "return_tensor": return_tensor
    }
    output = compare_models(
        models=models,
        config_benchmark=config_benchmark
    )
    return output
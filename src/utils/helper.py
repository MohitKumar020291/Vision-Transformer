import torch

import collections.abc
from itertools import repeat
from enum import Enum
from typing import Union, List, Tuple

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
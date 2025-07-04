import torch

from typing import Tuple

def getSetDevice() -> Tuple[bool, torch.device]:
    cuda_is_available = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_is_available else "cpu")

    return cuda_is_available, device

# These are not proper tests - will improve them
import torch
import torch.nn as nn

from vitm.Kernels import SoftmaxModule

from vitm.utils import compare, make_input, getSetDevice

def softmax_triton_implementation():
    """The Batch and heads are not considered"""
    is_cuda, device = getSetDevice()
    N, dim, input = \
        make_input(
                device, 
                batch=False, 
                multihead=False
                )
    
    # Triton
    triton_softmax_output = compare(
        input=input,
        models=[\
            SoftmaxModule()
            ],
        compile=[False],
        is_cuda=is_cuda,
        device=device,
        return_tensor=True
        )
    
    # pytorch
    pytorch_softmax_output = compare(
        input=input,
        models=[\
            nn.Softmax(dim=1)
            ],
        compile=[True],
        is_cuda=is_cuda,
        device=device,
        return_tensor=True
    )
    print(torch.allclose(triton_softmax_output[0][-1], pytorch_softmax_output[0][-1], atol=1e-3))

    print("Max abs diff:")
    diff = torch.abs(triton_softmax_output[0][-1] - pytorch_softmax_output[0][-1])
    print(torch.max(torch.abs(triton_softmax_output[0][-1] - pytorch_softmax_output[0][-1])))

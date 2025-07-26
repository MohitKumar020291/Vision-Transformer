import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import distribute_tensor, Shard, Replicate
from torch.multiprocessing import spawn, Queue
from torch.distributed.tensor.parallel import (
                                            parallelize_module, 
                                            ColwiseParallel, 
                                            RowwiseParallel)
from vitm.models.ParallelBlock import ParallelBlock
import numpy as np
from typing import Union, List
from enum import Enum
import os


def setup(device_id, world_size) -> int:
    """Initialize distributed process group and set CUDA device.
    
    Args:
        device_id: Global process rank (0 to world_size-1)
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    if not dist.is_initialized():
        dist.init_process_group(
            "nccl" if torch.cuda.is_available() else "gloo",
            rank=device_id, world_size=world_size
        )

    avail_device_id = device_id % world_size
    torch.cuda.set_device(avail_device_id)

    return avail_device_id


def create_dims(mesh_shape: tuple):
    mesh_dim_names = tuple(f"dim_{i}" for i in range(len(mesh_shape)))
    return mesh_dim_names


def create_mesh_shape(
        mesh_shape: Union[int, tuple] = None, 
        shard_along_dim: tuple = None) -> tuple:
    if isinstance(mesh_shape, int):
        mesh_shape = (1, mesh_shape)
    elif isinstance(mesh_shape, tuple):
        if len(mesh_shape) == 1:
            mesh_shape = (1, mesh_shape[0])
        else:
            return mesh_shape
    elif mesh_shape is None:
        if shard_along_dim is None:
            mesh_shape = (1, 1)
        elif isinstance(shard_along_dim, int):
            world_size = dist.get_world_size()
            mesh_shape = (world_size,)
        elif isinstance(shard_along_dim, tuple):
            raise ValueError("Provide mesh_shape if the sharding_along_dim is a tuple")
    return mesh_shape


def create_placements(
        shard_along_dims: tuple, 
        tensor_ndim: int, 
        mesh_ndim: int) -> List[Shard]:
    # check dtype using pydant
    placements = []
    if shard_along_dims is not None:
        if len(shard_along_dims) > mesh_ndim:
            raise ValueError(f"\nshard_along_dims = {shard_along_dims} \
                        cannot be greater than the mesh_ndim = {mesh_ndim}")

        shard_along_dim = list(shard_along_dims)
        shard_along_dim.sort()
        for idx, shard_along_dim in enumerate(shard_along_dims):
            if shard_along_dim < tensor_ndim:
                placements.append(Shard(shard_along_dim))
            else:
                raise ValueError(f"\nshard_along_dim = {shard_along_dim} at \
                    {idx} cannot be greater then the tensor ndim = {tensor_ndim}")

    for _ in range(mesh_ndim - len(placements)):
        placements.append(Replicate())
    return placements


np2torch_dtype = {
    np.int8: torch.int8,
    np.int16: torch.int16,
    np.int32: torch.int32,
    np.int64: torch.int64,
    np.float16: torch.float16,
    np.float32: torch.float32,
    np.float64: torch.float64,
}


def pre_check_and_process(
        mesh_shape,
        device_id, 
        tensor: torch.Tensor,
        device: str = "cuda"
):
    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor).to(np2torch_dtype[tensor.dtype])
    num_devices = np.prod(mesh_shape)
    assert num_devices == dist.get_world_size(), \
        f"Mismatch: mesh shape={mesh_shape}, num_devices={dist.get_world_size()}"

    tensor = tensor.to(f"{device}:{device_id}")

    if isinstance(shard_along_dims, int):
        shard_along_dims = (shard_along_dims,)

    return tensor, shard_along_dims


def create_mesh(
        mesh_shape: Union[int, tuple] = None,
        shard_along_dims: tuple = None,
        device: str = "cuda",
):
    mesh_shape = create_mesh_shape(mesh_shape, shard_along_dims)
    mesh_dim_names = create_dims(mesh_shape)
    # removed f"cuda:{device_id}" because
    # RuntimeError: ('Device type with index is not supported but got cuda:0. ', 
    # "If you maintained a 'torch.device' object, it's recommended to pass in 
    # 'device.type'.")
    mesh = init_device_mesh(device, mesh_shape, mesh_dim_names=mesh_dim_names)
    return mesh


def multiple_device_sharding(
        tensor: torch.Tensor,
        device_id: int,
        mesh: torch.distributed.device_mesh.DeviceMesh,
        mesh_shape: Union[int, tuple] = None,
        shard_along_dims: tuple = None,
        device: str = "cuda",
        visual: bool = False,
):
    """Shard tensor across devices.
    
    Args:
        tensor: Input tensor to shard
        mesh_shape: Device grid (e.g., (2,4))
        shard_dims: Dimensions to split along
        device_id: Current device ID
    """

    tensor, shard_along_dims = pre_check_and_process(
                                    mesh_shape=mesh_shape,
                                    device_id=device_id,
                                    tensor=tensor,
                                    device=device
                                )

    placements = create_placements(shard_along_dims=shard_along_dims, 
                    tensor_ndim=tensor.ndim, mesh_ndim=len(mesh_shape))
    sharded_tensor = distribute_tensor(tensor, mesh, placements=placements)

    if visual:
        print(f"\n[Device id {device_id}] Sharded tensor local part:\n", 
            sharded_tensor.to_local(), "\n")

    return sharded_tensor


def pre_configure_model(path_to_config: str = "models/sv22b.yaml"):
    from vitm.utils.helper import load_config

    path_to_config = os.path.join(os.path.join(os.getcwd(), "vitm"), path_to_config)
    print("loading config from:", path_to_config)
    config = load_config(path_to_config)

    return config["dim"], config["num_heads"], config["mlp_ratio"]


def tk_parallelism_test_function(
        device_id: int,
        world_size: int,
        path_to_config: str = "models/sv22b.yaml",
):
    assert world_size % 1 == 0, f"world_size={world_size} must be divisible by 4"
    rows = world_size // 1
    cols = 1
    mesh_shape = (rows, cols)
    mesh = create_mesh(device="cuda", mesh_shape=mesh_shape, shard_along_dims=None)
    dim, num_heads, mlp_ratio = pre_configure_model(path_to_config)

    model = ParallelBlock(
        dim=dim,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio
    )

    dp_mesh = mesh["dim_0"]
    tp_mesh = mesh["dim_1"]
    
    plan = {"fused_proj_1": ColwiseParallel(),
        "attn.proj": ColwiseParallel(),
        "mlp.out_proj": ColwiseParallel(),
        "fused_proj_2": RowwiseParallel()}
    model_tp = parallelize_module(
        model,
        tp_mesh,
        plan).to(device_id)
    model_2d = fully_shard(model_tp, mesh=dp_mesh)

    input_data = torch.randn(8, 3, dim).to(f"cuda:{device_id}")

    logits = model_2d(input_data)

    dist.destroy_process_group()

    return logits



def main():
    """Main function executed per device"""
    device_id = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    avail_device_id = setup(device_id, world_size)

    tk_parallelism_test_function(
        device_id=avail_device_id,
        world_size=world_size,
        path_to_config="models/sv22b.yaml",
    )


if __name__ == "__main__":
    main()



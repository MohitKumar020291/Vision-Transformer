import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import os
from torch.distributed.tensor.parallel import parallelize_module, ColwiseParallel
from torch.nn.parallel import DistributedDataParallel as DDP


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc = nn.Linear(10, 10)

    def forward(self, x):
        return self.fc(x)


def example(rank, world_size):
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    model = parallelize_module(
                Model(),
                ) # Will provide a parallel model here

    ddp_model = DDP(model, device_ids=[rank])

    loss_fn = nn.MSELoss()

    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    outputs = ddp_model(torch.randn(20, 10).to(rank))

    labels = torch.randn(20, 10).to(rank)

    loss_fn(outputs, labels).backward()

    optimizer.step()


def main():
    world_size = torch.cuda.device_count()
    mp.spawn(example,
        args=(world_size,),
        nprocs=world_size,
        join=True)


if __name__=="__main__":
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    main()
import torch
import torchvision
import torch.nn as nn
import torch.autograd.profiler as profiler
import torch.optim as optim
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, CosineAnnealingLR
import torchvision.transforms as transforms
from torchvision import transforms

from typing import Tuple, Union
import os

from models import VisionTransformer, VITC
from data.dataloader import loadData
from utils.helper import getParam
from .callbacks import PrintLossCallback, ModelCheckpoint

def move_batch_to_device(batch, device):
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    elif isinstance(batch, (tuple, list)):
        return type(batch)(move_batch_to_device(x, device) for x in batch)
    elif isinstance(batch, dict):
        return {k: move_batch_to_device(v, device) for k, v in batch.items()}
    else:
        return batch


def getOptimizerAndSchedular(
      model: nn.Module,
      epochs: int,
      trainloader_length: int,
      optimizer: optim = None,
      scheduler: optim.lr_scheduler.SequentialLR = None
    ) -> Tuple[Optimizer, LRScheduler]:
    """
    trainloader_length: number of batches per epoch or iterations per epoch
    """
    # LinearLR:
    # base_lr = 0.001 and total_iters=500 <- learning rate change for 500 times the step() method
    # for {i} in [0, total_iters-1]
    # lr_{i} = base_lr * (start_factor + {i} * (end_factor - start_factor) / (total_iters - 1))
    # linearly goes from 0.01 × base_lr to 1.0 × base_lr across 500 steps (iterations)
    # CosineAnnealingLR
    # for {i} in [0, T_max]
    # lr_{i} = eta_min + 1 / 2 * (base_lr - eta_min) / (1 + cos( {i} * pi / T_max ))
    # Starts at base_lr and decays to eta_min over T_max steps in a cosine curve. 
    # The decay gets flatter and flatter as it approaches the end — unlike exponential decay which drops fast
    # milestone must match the total_iters in SequentialLR <-- at what iter to change

    if optimizer and optimizer == "adam":
        optimizer = optim.AdamW(
                    model.parameters(), 
                    lr=1e-3, 
                    weight_decay=0.05,
                    betas=(0.9, 0.98))
    else:
        optimizer = optimizer

    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        [
            torch.optim.lr_scheduler.LinearLR(
                optimizer, 
                start_factor=0.01, 
                end_factor=1.0, 
                total_iters=500
            ),
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=(epochs*trainloader_length)-500, 
                eta_min=1e-6
            )
        ],
        milestones=[500]
    )
    return optimizer, scheduler


def modelSpecs(config: dict):
    """
    return model_path, epochs, batch_size, optimizer, scheduler, criterion, use_cuda
    """
    returns = ["model_path", "epochs", "batch_size", "optimizer", "scheduler", "criterion", "use_cuda"]
    parent_key = {
        "model_path": ["experiment", "runs", "save_dir"],
        "epochs": ["training"],
        "batch_size": ["training"],
        "optimizer": ["training"],
        "scheduler": ["training"],
        "criterion": ["training"],
        "use_cuda": ["training"]
    }
    for idx, arg in enumerate(returns):
        print(arg)
        if arg == "model_path":
            keys = ["paths", "save_dir"]
            output, config_str, error_msg = getParam(config, keys)

            import time
            print("creating a new model path...")
            fallback_path = time.strftime("%Y-%m-%d_%H-%M-%S") + ".txt"
            print(f"New model path is: {fallback_path}")

            returns[idx] = "experiment/runs/" or output
            returns[idx] += fallback_path 
            if error_msg:
                f"The model to be saved on {returns[idx]}"
        
        # A dictionary which stores the parent of epochs: like training -> so we don't have to do arg == "epochs"
        else:
            keys = parent_key[arg] + [arg]
            print(keys)
            output, config_str, error_msg = getParam(config, keys)
            if error_msg:
                raise Exception(error_msg)
            returns[idx] = output
    
    print(returns, "\n\n\n\n\n")
    return tuple(returns)


# The function later goes into utils/helper.py
def get_model_args(model_args):
    if model_args["type"] == "ViT":
        image_size = model_args["image_size"]
        patch_size = model_args["patch_size"]
        return "ViT", image_size, patch_size


def train_model(
    config: dict
    ):
    print(config)
    model_path, epochs, batch_size, optimizer, scheduler, criterion, use_cuda = modelSpecs(config)
    cuda_is_available = torch.cuda.is_available()
    use_cuda = cuda_is_available
    device = torch.device("cuda" if cuda_is_available else "cpu")
    if use_cuda and not cuda_is_available:
        # use_gpu -> extends to AMD
        print("WARNING: CUDA requested but not available. Training on CPU.")
    elif use_cuda and cuda_is_available:
        print("Training with CUDA (GPU enabled).")
    else:
        print("Training with CPU.")

    image_size, num_classes, trainloader = loadData(batch_size)
    norm_layer = nn.LayerNorm
    vision_transformer = VisionTransformer(
        image_size,
        patch_size=4,
        num_classes=num_classes,
        num_heads=8,
        mlp_ratio=0.8,
        norm_layer=norm_layer,
        embed_norm_layer=norm_layer,
        final_norm_layer=norm_layer)

    vitc = VITC(vit_model=vision_transformer)
    vitc = vitc.to(device)

    for name, param in vitc.named_parameters():
        if param.requires_grad:
            print(f"{name}: mean={param.data.mean():.5f}, std={param.data.std():.5f}")

    # find alternatives also
    if optimizer == "adam" and scheduler == "sequential-lr":
        optimizer, scheduler = getOptimizerAndSchedular(vitc, epochs, len(trainloader), optimizer, scheduler)
    if criterion and criterion == "cross-entropy":
        criterion = nn.CrossEntropyLoss()
        criterion.to(device)
    else:
        # what to do if criterion is not provided? We will find out
        ...

    try:
        model_args = config.get("model", None)
        if not model_args:
            raise KeyError(f"No 'model' key provided in config")
    except Exception as e:
        print("Error:", e)

    print("starting model training...")
    callbacks = [
        PrintLossCallback(),
        ModelCheckpoint(model=vitc, path=model_path)
    ]

    # start training
    for epoch in range(epochs):
        for cb in callbacks:
            method = getattr(cb, 'on_epoch_start', None)
            if callable(method):
                method(epoch)

        running_loss = 0.0 # total loss of an epoch
        for i, batch in enumerate(trainloader, 0):
            batch = move_batch_to_device(batch, device)
            inputs, labels = batch

            optimizer.zero_grad()
            outputs = vitc(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vitc.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            batch_loss = loss.item()
            running_loss += batch_loss

            logs = {"loss": loss.item()}
            for cb in callbacks:
                method = getattr(cb, 'on_batch_end', None)
                if callable(method):
                    method(i, logs)
            
            print(prof.key_averages(group_by_stack_n=5).table(sort_by='cuda_time_total', row_limit=5))
            break
        
        logs = {"val_loss": running_loss}
        for cb in callbacks:
            method = getattr(cb, 'on_epoch_end', None)
            if callable(method):
                method(i, logs)
        running_loss = 0.0

    print("Finished Training!")

    torch.save({
    'model_state_dict': vitc.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict()},
    model_path)

    print("Saved model! ")
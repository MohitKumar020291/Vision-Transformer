import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision
import torchvision.transforms as transforms
from torchvision import transforms
from torch.utils.data import Subset

import matplotlib.axes as axes
import matplotlib.pyplot as plt
import numpy as np
import random

from typing import Optional
from models import VisionTransformer, VITC


# functions to show an image
def imshow(img, ax: Optional[axes._axes.Axes] = None):
  img = img / 2 + 0.5
  npimg = img.numpy()
  plt.imshow(np.transpose(npimg, (1, 2, 0)))


# Define a transformation to apply to the images
transform = transforms.Compose(
    [
        # transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

batch_size = 32
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
subset_indices = random.sample(range(len(trainset)), 50000)
train_subset = Subset(trainset, subset_indices)
trainloader = torch.utils.data.DataLoader(train_subset, 
                                        batch_size=batch_size,
                                        shuffle=True, 
                                        num_workers=2)
print(len(trainloader.dataset))

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# making a vision transformer model
data_iter = iter(trainloader)
images, labels = next(data_iter)
num_classes = len(classes)
image_shape = images[0].shape
image_size = image_shape[-2], image_shape[-1]
norm_layer = nn.LayerNorm
vision_transformer = VisionTransformer(
    image_size,
    patch_size=4,
    num_classes=num_classes,
    num_heads=8,
    mlp_ratio=0.8,
    norm_layer=norm_layer,
    embed_norm_layer=norm_layer,
    final_norm_layer=norm_layer
    )

# vitc = VITC(
#     vit_model=vision_transformer,
#     hidden_features=768,
#     mlp_ratio=0.8,
#     bias=True
# )
vitc = VITC(vit_model=vision_transformer)

epochs = 16
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(
    vitc.parameters(), 
    lr=1e-3, 
    weight_decay=0.05,
    betas=(0.9, 0.98)
)
# scheduler = CosineAnnealingLR(optimizer, T_max=epochs*len(trainloader), eta_min=1e-6)
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
            T_max=(epochs*len(trainloader))-500, 
            eta_min=1e-6
        )
    ],
    milestones=[500]
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vitc = vitc.to(device)

for name, param in vision_transformer.named_parameters():
    if param.requires_grad:
        print(f"{name}: mean={param.data.mean():.5f}, std={param.data.std():.5f}")

for epoch in range(epochs):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = vitc(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(vitc.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()
        print(f"epoch: {epoch+1}, Batch: {i+1}, loss: {running_loss}")
        running_loss = 0.0

print('Finished Training')

torch.save(vitc.state_dict(), 'vitc_model.pth')
print("Saved model")

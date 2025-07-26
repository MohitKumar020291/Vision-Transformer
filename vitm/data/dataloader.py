# Take the choice of dataset from user
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset

import random

def loadData(
    batch_size: int = 32,
    resize: int = 0
  ):
  """
    Load the CIFAR-10 dataset with optional resizing and return a DataLoader.

    Args:
        batch_size (int): Number of samples per batch.
        resize (int): If non-zero, resizes image to (resize, resize).

    Returns:
        image_size (Tuple[int, int]): Width and height of the images.
        num_classes (int): Number of classes in the dataset.
        trainloader (DataLoader): PyTorch DataLoader with training data.
    """
  compose_arg = [
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
  ]
  if resize != 0:
    compose_arg.insert(0, transforms.Resize(resize),)
  transform = transforms.Compose(compose_arg)
  # Take the dataset name through the CLI and check if the dataset is available or not?
  # If the dataset has to be taken from other resources then - clean it first
  trainset = torchvision.datasets.CIFAR10(
                root='./data', 
                train=True,
                download=True, 
                transform=transform)
  subset_indices = random.sample(range(len(trainset)), 50000)
  train_subset = Subset(trainset, subset_indices)
  trainloader = torch.utils.data.DataLoader(train_subset, 
                                          batch_size=batch_size,
                                          shuffle=True, 
                                          num_workers=2, # num_workers needs to be tuned
                                          pin_memory=True)
  classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


  # making a vision transformer model
  data_iter = iter(trainloader)
  images, labels = next(data_iter)
  num_classes = len(classes)
  image_shape = images[0].shape
  image_size = image_shape[-2], image_shape[-1]

  print("DATA INFO:")
  print("# Samples:", len(trainloader.dataset)) 
  print("# Classes:", len(classes))
  print("Image size:", image_size)

  return image_size, num_classes, trainloader
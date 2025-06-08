import torch
import torch.nn as nn

from models import VisionTransformer

class VITC(nn.Module):
  def __init__(self, vit_model: VisionTransformer):
    super().__init__()
    self.vit_model = vit_model
    self.head = nn.Linear(self.vit_model.embed_dim, self.vit_model.num_classes)

  def forward(self, x: torch.Tensor):
    x = self.vit_model(x)
    x = x[:, 0]
    x = self.head(x)
    return x
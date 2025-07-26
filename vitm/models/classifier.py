import torch
import torch.nn as nn

from vitm.models import VisionTransformer

class VITC(nn.Module):
  """
  A classification wrapper around a Vision Transformer (ViT) model.

  This module takes a VisionTransformer backbone (e.g. ViT encoder),
  extracts the representation of the [CLS] token, and applies a
  linear classification head to produce final logits.

  Args:
      vit_model (VisionTransformer): A ViT model instance that outputs
          token embeddings. Must expose `embed_dim` and `num_classes` attributes.

  Attributes:
      vit_model (VisionTransformer): The backbone transformer model.
      head (nn.Linear): Linear classification head that maps the [CLS] token
          to `num_classes` output classes.

  Forward Input:
      x (torch.Tensor): A batch of input images of shape (B, C, H, W).

  Forward Output:
      logits (torch.Tensor): Output tensor of shape (B, num_classes), where
      each row contains the unnormalized log-probabilities for each class.
  """
  def __init__(self, vit_model: VisionTransformer):
    super().__init__()
    self.vit_model = vit_model
    self.head = nn.Linear(self.vit_model.embed_dim, self.vit_model.num_classes)

  def forward(self, x: torch.Tensor):
    x = self.vit_model(x)
    x = x[:, 0]
    x = self.head(x)
    return x
"""
Generates patch embeddings
Implemented from the: https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/patch_embed.py#L26
"""

import torch
import torch.nn as nn

from type import Optional, Callable, Tuple, Union

from ../helper/helper.py import to_2tuple, nchw_to

class PatchEmbed(nn.Module):
  def __init__(
      self,
      image_size: Union[int, Tuple[int, int]] = 224,
      patch_size: int = 16,
      in_channels: int = 3,
      embed_dim: int = 768,
      norm_layer: Optional[Callable] = None,
      flatten: bool = True,
      bias: bool = False,
      output_fmt: Optional[Format] = None
      ):
    super().__init__()
    self.output_fmt = output_fmt
    self.patch_size = to_2tuple(patch_size)
    self.image_size, self.grid_size, self.num_patches = self._init_img_size(image_size)
    self.flatten = flatten

    # Think of this as:
    # A linear layer with input features = in_channels * kernel_size * kernel_size
    # while the output features being embed_dim
    # N is decided by stride argument
    self.projection = nn.Conv2d(in_channels, embed_dim, kernel_size=self.patch_size, stride=self.patch_size, bias=bias)
    nn.init.kaiming_normal_(self.projection.weight, mode='fan_in', nonlinearity='relu')
    self.norm_layer = norm_layer(embed_dim) if norm_layer else nn.Identity()

  def _init_img_size(self, image_size: Union[int, Tuple[int, int]]):
    image_size = to_2tuple(image_size)
    grid_size = [s//p for s, p in zip(image_size, self.patch_size)]
    num_patches = grid_size[0] * grid_size[1]
    return image_size, grid_size, num_patches

  def forward(self, x: torch.Tensor):
    B, C, H, W = x.shape
    print(B, C, H, W)
    assert(self.image_size[0] == H), "Input tensor's height does not match the model"
    assert(self.image_size[1] == W), "Input tensor's height does not match the model"
    x = self.projection(x)
    if self.flatten:
      x = x.flatten(2).transpose(1, 2) #BCHW -> BNC
    elif self.output_fmt != "NCHW":
      x = nchw_to(x)
    x = self.norm_layer(x)
    return x

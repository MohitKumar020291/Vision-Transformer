"""
Generates patch embeddings
Implemented from the: https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/patch_embed.py#L26
"""

import torch
import torch.nn as nn

from typing import Optional, Callable, Tuple, Union

from vitm.utils.helper import to_2tuple, nchw_to, Format


class PatchEmbed(nn.Module):
  """
  Converts an input image into a sequence of patch embeddings for use in Vision Transformer models.

  This module splits an image into non-overlapping patches, flattens each patch, and projects it
  to a specified embedding dimension via a 2D convolution. Optionally applies a normalization layer.

  Inspired by: https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/patch_embed.py

  Args:
      image_size (int or Tuple[int, int], optional): Size of the input image (height, width). 
          If int, a square image is assumed. Default is 224.
      patch_size (int, optional): Size of each image patch (both height and width). Default is 16.
      in_channels (int, optional): Number of input channels (e.g., 3 for RGB). Default is 3.
      embed_dim (int, optional): Dimension of the output embedding for each patch. Default is 768.
      norm_layer (Callable, optional): Optional normalization layer applied after projection. 
          If None, uses identity (no normalization). Default is None.
      flatten (bool, optional): Whether to flatten the spatial dimensions and return output in (B, N, C) format. 
          If False, keeps (B, C, H, W) or uses custom output format. Default is True.
      bias (bool, optional): If True, adds a learnable bias to the projection layer. Default is False.
      output_fmt (Format, optional): Output format if `flatten=False`. Controls how non-flattened output is represented. 
          Default is None.

  Attributes:
      projection (nn.Conv2d): Convolutional layer that performs patch embedding.
      norm_layer (nn.Module): Normalization layer (or identity if not specified).
      num_patches (int): Total number of patches in the input image.
      image_size (Tuple[int, int]): Resolved input image size.
      grid_size (Tuple[int, int]): Number of patches along (height, width).
  """
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

    # For a patch of image(B, C, H, W), think of this as:
    # A linear layer with input features = in_channels * kernel_size * kernel_size (if type kernel_size=int)
    # while the output features being embed_dim
    # N is decided by stride argument
    # Finally, H//patch_size = H_dash, W//patch_size=W_dash of patches => 1, embed_dim, H_dash, W_dash (1 b/c of 1 image)
    # Then, for each batch we have (B, embed_dim, H_dash, W_dash)
    self.projection = nn.Conv2d(in_channels, embed_dim, kernel_size=self.patch_size, stride=self.patch_size, bias=bias)
    nn.init.kaiming_normal_(self.projection.weight, mode='fan_in', nonlinearity='relu')
    self.norm_layer = norm_layer(embed_dim) if norm_layer else nn.Identity()

  def _init_img_size(self, image_size: Union[int, Tuple[int, int]]):
    """
    Initializes image size and computes grid and patch counts.

    Args:
        image_size (int or Tuple[int, int]): Height and width of the input image.

    Returns:
        Tuple containing:
            - image_size (Tuple[int, int]): Resolved image size.
            - grid_size (Tuple[int, int]): Grid dimensions (height_patches, width_patches).
            - num_patches (int): Total number of patches.
    """
    image_size = to_2tuple(image_size)
    grid_size = [s//p for s, p in zip(image_size, self.patch_size)]
    num_patches = grid_size[0] * grid_size[1]
    return image_size, grid_size, num_patches

  def forward(self, x: torch.Tensor):
    """
    Applies patch embedding to the input tensor.

    Args:
        x (torch.Tensor): Input tensor of shape (B, C, H, W), where:
            - B: batch size
            - C: number of channels
            - H: height of image
            - W: width of image

    Returns:
        torch.Tensor: If `flatten=True`, returns shape (B, N, C) where N is number of patches. 
                      If `flatten=False`, returns (B, C, H', W') or uses `output_fmt` conversion.

    Examples:
      >>> embed = PatchEmbed(image_size=32, patch_size=8, in_channels=3, embed_dim=784)
      >>> x = torch.randn(1, 3, 32 32)
      >>> y = embed(x)
      >>> y.shape
      torch.Size([1, 16, 784])
    """
    B, C, H, W = x.shape
    print(B, C, H, W)
    assert(self.image_size[0] == H), "Input tensor's height does not match the model"
    assert(self.image_size[1] == W), "Input tensor's width does not match the model"
    x = self.projection(x)
    if self.flatten:
      x = x.flatten(2).transpose(1, 2) #(B, embed_dim, H_dash, W_dash) -> B, embed_dim, N -> B, N, embed_dim
    elif self.output_fmt != "NCHW":
      x = nchw_to(x, self.output_fmt)
    x = self.norm_layer(x)
    return x
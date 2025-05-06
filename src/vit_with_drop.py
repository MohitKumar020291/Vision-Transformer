import timm
import torch.nn as nn

import types
import inspect
from PIL import Image
import copy
import requests
from urllib.request import urlopen


functype = types.MethodType

# Import this function and provide your implementation of forward
"""
# Using DynamicViT

class TokenPruner(nn.Module):
  def __init__(self, embed_dim: int, n_blocks: int):
    super().__init__()
    self.reduction_ratios = [0.2 for _ in range(n_blocks)]
    self.predictor = torch.randn(embed_dim, 1) #That's what we are learning

  def forward(self, x, stage):
    scores = (x @ self.predictor).squeeze()
    
    print(1 - self.reduction_ratios[stage], len(scores))
    num_keeps = int((1 - self.reduction_ratios[stage]) * len(scores))
    topk = torch.topk(scores, num_keeps)

    indices = list(topk.indices)
    pruned_x = x[:, indices]
    print(pruned_x.shape)
    return indices, scores, pruned_x

def redifined_forward_dvit(self: block_class, x):
  token_predictor = TokenPruner(x.shape[-1], self.total_idxs)
  x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
  indices, scores, x = token_predictor(x, stage = self.my_index)
  x = x + self.drop_path2(self.ls2(self.mlp(x)))
  return x
"""

def timm_manipulate(new_block_forward) -> None:
  org_model = timm.create_model('vit_base_patch16_224.augreg2_in21k_ft_in1k', pretrained=True)
  model = copy.deepcopy(org_model)

  # I have to make this code correct : get depth from anywhere bro!
  # Right now I am hardcoding it
  num_blocks = len(model._modules["blocks"])

  for idx, block in enumerate(model._modules["blocks"]):
    model._modules["blocks"][idx].forward = functype(new_block_forward, block)
    block.my_index = idx
    block.total_idxs = num_blocks

  for model_ in [model, org_model]:
    img = Image.open(urlopen(
        'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'
    ))

    model_ = model_.eval()

    data_config = timm.data.resolve_model_data_config(model_)
    transforms = timm.data.create_transform(**data_config, is_training=False)

    output = model_(transforms(img).unsqueeze(0))

    top5_probabilities, top5_class_indices = torch.topk(output.softmax(dim=1) * 100, k=5)
    print(top5_probabilities, top5_class_indices)

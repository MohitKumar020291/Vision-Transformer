import timm
import torch.nn as nn

import types
import inspect
from PIL import Image
import copy

functype = types.MethodType

org_model = timm.create_model('vit_base_patch16_224.augreg2_in21k_ft_in1k', pretrained=True)
model = copy.deepcopy(org_model)

for idx, block in enumerate(org_model._modules["blocks"]):
  block.pos_embed_fn = org_model._pos_embed

for idx, block in enumerate(model._modules["blocks"]):
  block.pos_embed_fn = model._pos_embed

block_class = type(model._modules["blocks"][0])
def redifined_forward(self: block_class, x: torch.Tensor) -> torch.Tensor:
  self.token_dropper = TokenDropper(dim=x.shape[-1])
  x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
  print("Block Idx", self.my_index)
  x = self.token_dropper(self.norm2(x))
  x = x + self.drop_path2(self.ls2(self.mlp(x)))
  return x

for idx, block in enumerate(model._modules["blocks"]):
  model._modules["blocks"][idx].forward = functype(redifined_forward, block)
  block.my_index = idx

from PIL import Image
import requests
from urllib.request import urlopen

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

import collections.abc
from itertools import repeat
from enum import Enum

#Any x will be repeted n times if it is not iterable
def _ntuple(n):
  def parse(x):
    if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
      return tuple(x)
    return tuple(repeat(x, n))
  return parse

to_2tuple = _ntuple(2)
# to_2tuple(4)


class Format(str, Enum):
  NCHW = 'NCHW'
  NHWC = 'NHWC'
  NCL = 'NCL'
  NLC = 'NLC'

def nchw_to(x: torch.Tensor, fmt: Format) -> torch.Tensor:
  if fmt == "NHWC":
    x = x.permute(0, 2, 3, 1)
  elif fmt == Format.NLC:
        x = x.flatten(2).transpose(1, 2)
  elif fmt == Format.NCL:
      x = x.flatten(2)
  return x

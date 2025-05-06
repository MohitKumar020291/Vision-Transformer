import torch
import torch.nn as nn
from typing import Optional, Union

class TokenDropper(nn.Module):
    def __init__(self, dim: int, keep_ratio: Optional[float] = 0.7):
        super().__init__()
        self.keep_ratio = keep_ratio
        # convert to orderDict if required
        self.score_func = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 1)
        )
    
    # We are dropping tokens on the basis of the attention score - what could be improved about it
    def forward(self, x):
        B, N, D = x.shape
        # for each of the token we will have scores
        # interesting is that nn.Sequential do not have __call__ function (it have but for RuntimeError)
        scores = self.score_func(x).squeeze(-1) # squeeze(-1) because nn.Linear returns 1 element
        k = int(N * self.keep_ratio)

        topk = scores.topk(k, dim=-1) # we have squeezed already
        lowk = (-1 * scores).topk(N-k, dim=-1).indices
        print("dropped tokens:", lowk, "\n")
        print("#dropped tokens:", N-k)
        print("---------------\n\n")
        indices = topk.indices #B, k

        # dim = 1 is changing
        x_kept = torch.gather(x, dim=1, index=indices.unsqueeze(-1).expand(-1, -1, D))
        return x_kept


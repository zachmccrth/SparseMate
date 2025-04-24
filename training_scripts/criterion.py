from torch.nn import Module

from torch import norm, Tensor

class ResidualNorm(Module):
    def __init__(self):
        super().__init__()

    def forward(self, input: Tensor, *args, **kwargs):
        return norm(input, dim=-1).mean()


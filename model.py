import math
import torch
from torch import nn
from torch.nn import functional as F

class StableDropout(nn.Module):
    def __init__(self, p=0.5) -> None:
        super().__init__()
        self.mask = None
        self.p = p

    def reset(self):
        self.mask = None
    
    def forward(self, x):
        return x
        if self.mask is None:
            self.mask = torch.rand_like(x) < self.p
        return x.mul(self.mask).div(self.p)


class Model(nn.Module):
    def __init__(self, dim_obs=9, dim_action=4) -> None:
        super().__init__()
        self.stem = nn.Linear(12*16, 192, bias=False)
        self.v_proj = nn.Linear(dim_obs, 192)
        # self.drop1 = StableDropout()
        # self.drop2 = StableDropout()

        # i, j = torch.meshgrid(torch.linspace(0, 1, 12), torch.linspace(0, 1, 16))
        # rnd = torch.unbind(torch.rand(4, 192) - 0.5)
        # w = i[..., None] * rnd[0] + (1 - i[..., None]) * rnd[1] \
        #     + j[..., None] * rnd[2] + (1 - j[..., None]) * rnd[3]
        # w = w.flatten(0, 1).t().contiguous()
        # self.stem.weight.data[:] = w / w.std() / math.sqrt(12*16*2)
        # self.v_proj.weight.data.div_(math.sqrt(2))

        self.gru = nn.GRUCell(192, 192)
        self.fc = nn.Linear(192, dim_action, bias=False)
        self.fc.weight.data.mul_(0.01)
        # self.history = []
        self.act = nn.LeakyReLU(0.05)

    def reset(self):
        # self.drop1.reset()
        # self.drop2.reset()
        pass

    def forward(self, x: torch.Tensor, v, hx=None):
        x = self.act(self.stem(x.flatten(1)) + self.v_proj(v))
        hx = self.gru(x, hx)
        return self.fc(self.act(hx)), hx


if __name__ == '__main__':
    Model()

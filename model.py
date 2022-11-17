import torch
from torch import nn
from torch.nn import functional as F


class Model(nn.Module):
    def __init__(self, dim_obs=9, dim_action=4) -> None:
        super().__init__()
        self.stem = nn.Linear(16*9, 256, bias=False)
        self.v_proj = nn.Linear(dim_obs, 256)
        self.gru = nn.GRUCell(256, 256)
        self.fc = nn.Linear(256, dim_action, bias=False)
        self.fc.weight.data.mul_(0.01)
        self.drop = nn.Dropout()
        self.history = []

    def forward(self, x: torch.Tensor, v, hx=None):
        x = F.max_pool2d(x, 10, 10)
        x = (self.stem(x.flatten(1)) + self.v_proj(v)).relu()
        x = self.drop(x)
        hx = self.gru(x, hx)
        return self.fc(self.drop(hx)).tanh(), hx

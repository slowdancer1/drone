import torch
from torch import nn
from torch.nn import functional as F


class Model(nn.Module):
    def __init__(self, dim_obs=9, dim_action=4) -> None:
        super().__init__()
        self.stem = nn.Linear(12*16, 192, bias=False)
        self.v_proj = nn.Linear(dim_obs, 192)
        self.gru = nn.GRUCell(192, 192)
        self.fc = nn.Linear(192, dim_action, bias=False)
        self.fc.weight.data.mul_(0.01)
        self.history = []

    def forward(self, x: torch.Tensor, v, hx=None):
        # x = F.max_pool2d(x, 5, 5)
        x = (self.stem(x.flatten(1)) + self.v_proj(v)).relu_()
        hx = self.gru(x, hx)
        return self.fc(hx.relu()).tanh(), hx

from cmath import isnan
from turtle import forward
from matplotlib import pyplot as plt
import numpy as np
from env import Env, quaternion_to_forward
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import AdamW
from tensorboardX import SummaryWriter


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mem = nn.Parameter(torch.randn(150, 4)*0.001)
        self.stem = nn.Conv2d(3, 64, 7, 2)
        self.gru = nn.GRUCell(64, 64)
        self.fc = nn.Linear(64, 4, bias=False)
        nn.init.constant_(self.fc.weight, 0)

    def forward(self, x: torch.Tensor, hx=None):
        n, c, h, w = x.shape
        pos = torch.stack(torch.meshgrid(
            torch.linspace(-1, 1, h, device=x.device),
            torch.linspace(-1, 1, w, device=x.device)
        ), 0)[None].expand(n, 2, h, w)
        x = torch.cat([x, pos], 1)
        x = self.stem(x)
        x = x.relu().mean((2, 3))
        hx = self.gru(x, hx)
        return self.fc(hx), hx

# model = Model()
model = Model().cuda()
env = Env()
optim = AdamW(model.parameters(), 2e-4)

ctl_dt = 1 / 15

writer = SummaryWriter(flush_secs=1)

for i in range(250):
    env.reset()
    p_history = []
    q_history = []
    v_history = []
    act_history = []
    h = None
    loss_obj_avoidance = 0
    for t in range(i + 2):
        with torch.no_grad():
            _, depth = env.render()
            depth = np.nan_to_num(1 / depth, False, 0, 0, 0)
            x = torch.clamp(torch.as_tensor(depth[None, None]) - 1, -1, 5)
        act, h = model(x.cuda(), h)
        env.step(act[0].cpu(), ctl_dt)

        p = env.quad.p + env.quad.v * torch.linspace(0, ctl_dt * 2, 10)[:, None]
        distance = (p[:, None] - env.obstacles).pow(2).sum(-1).sqrt().add(-1)
        distance = torch.cat([distance, p[:, -1:] + 1], -1)
        if torch.any(distance < 0):
            break
        loss_obj_avoidance += (1 - distance).relu().pow(2).mean()

        p_history.append(env.quad.p)
        q_history.append(env.quad.q)
        v_history.append(env.quad.v)
        act_history.append(act)

    p_history = torch.stack(p_history)
    q_history = torch.stack(q_history)
    v_history = torch.stack(v_history)
    act_history = torch.stack(act_history)

    v_target = torch.zeros_like(v_history)
    v_target[:, 0] = 1
    loss_v_error = F.mse_loss(v_history, v_target, reduction='none').sum(-1).mean()
    loss_p_error = F.mse_loss(p_history[:, 1:], v_target[:, 1:], reduction='none').sum(-1).mean()

    loss_angular_acc = (act_history[1:] - act_history[:-1]).div(ctl_dt).pow(2).sum(-1).mean()
    loss_acc = (v_history[1:] - v_history[:-1]).div(ctl_dt).pow(2).sum(-1).mean()

    distance = (p_history[:, None] - env.obstacles).pow(2).sum(-1).sqrt().add(-1)
    distance = torch.cat([distance, p_history[:, -1:] + 1], -1).relu()

    # v_norm = torch.norm(v_history, dim=-1)
    # loss_look_ahead = (1 - (quaternion_to_forward(q_history) @ v_history.t()) / (v_norm + 0.1)) * v_norm
    # loss_look_ahead = loss_look_ahead.mean()
    loss_look_ahead = 0

    loss_obj_avoidance /= t + 1

    loss = loss_v_error + 0.1 * loss_p_error + 0.1 * loss_angular_acc + 0.001 * loss_acc + 1e3 * loss_obj_avoidance + loss_look_ahead

    nn.utils.clip_grad.clip_grad_norm_(model.parameters(), 0.01)
    print(loss.item())
    optim.zero_grad()
    loss.backward()
    optim.step()
    with torch.no_grad():
        writer.add_scalar('loss', loss, i)
        writer.add_scalar('loss_v_error', loss_v_error, i)
        writer.add_scalar('loss_p_error', loss_p_error, i)
        writer.add_scalar('loss_angular_acc', loss_angular_acc, i)
        writer.add_scalar('loss_acc', loss_acc, i)
        writer.add_scalar('loss_obj_avoidance', loss_obj_avoidance, i)
        writer.add_scalar('loss_look_ahead', loss_look_ahead, i)
        writer.add_scalar('t', t, i)
        if (i + 1) % 50 == 0:
            plt.plot(v_history[:, 0], label=f'{i + 1}')

plt.legend()
plt.savefig('log.png')

with torch.no_grad():
    env.reset()
    p_history = []
    v_history = []
    vid = []
    h = None
    for t in range(250):
        color, depth = env.render()
        depth = np.nan_to_num(1 / depth, False, 0, 0, 0)
        x = torch.clamp(torch.as_tensor(depth[None, None]) - 1, -1, 5)
        act, h = model(x.cuda(), h)
        env.step(act[0].cpu(), ctl_dt)
        if t % 5 == 0:
            color, depth = env.render()
            vid.append(color)
        p_history.append(env.quad.p)
        v_history.append(env.quad.v)
    p_history = torch.stack(p_history)
    v_history = torch.stack(v_history)

    vid = np.stack(vid).transpose(0, 3, 1, 2)[None]
    writer.add_video('color', vid, fps=3)
    fig = plt.figure()
    plt.plot(v_history[:, 0], label='x')
    plt.plot(v_history[:, 1], label='y')
    plt.plot(v_history[:, 2], label='z')
    plt.legend()
    writer.add_figure('v', fig)

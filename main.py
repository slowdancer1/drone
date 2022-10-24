import os
from matplotlib import pyplot as plt
import numpy as np
from env_gl import Env
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import AdamW
from tensorboardX import SummaryWriter

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--resume')
parser.add_argument('--batch_size', default=16)
args = parser.parse_args()

class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Linear(16*9, 256, bias=False)
        self.v_proj = nn.Linear(7, 256)
        self.gru = nn.GRUCell(256, 256)
        self.fc = nn.Linear(256, 4, bias=False)
        self.fc.weight.data.mul_(0.01)
        self.drop = nn.Dropout()
        self.history = []
    
    def forward(self, x: torch.Tensor, v, hx=None):
        x = F.max_pool2d(x, 10, 10)
        x = (self.stem(x.flatten(1)) + self.v_proj(v)).relu()
        x = self.drop(x)
        hx = self.gru(x, hx)
        return self.fc(self.drop(hx)).tanh(), hx

# model = Model()
device = torch.device('cpu')
model = Model().to(device)
if args.resume:
    model.load_state_dict(torch.load(args.resume, map_location=device))
env = Env(args.batch_size, device)
optim = AdamW(model.parameters(), 5e-4)

ctl_dt = 1 / 15

writer = SummaryWriter(flush_secs=1)
p_ctl_pts = torch.linspace(0, ctl_dt, 8, device=device).reshape(-1, 1, 1, 1)

for i in range(10000):
    env.reset()
    p_history = []
    v_history = []
    act_history = []
    vid = []
    h = None
    loss_obj_avoidance = 0
    for t in range(150):
        color, depth = env.render()
        depth = torch.as_tensor(depth[:, None]).to(device)
        x = torch.clamp(1 / depth - 1, -1, 6)
        if (i + 1) % 100 == 0 and t % 3 == 0:
            vid.append(color[0].copy())
        state = torch.cat([env.quad.v, env.quad.q], -1)
        act, h = model(x, state, h)
        env.step(act, ctl_dt)

        p_history.append(env.quad.p)
        v_history.append(env.quad.v)
        act_history.append(act)

    p_history = torch.stack(p_history)
    v_history = torch.stack(v_history)
    act_history = torch.stack(act_history)

    v_target = torch.randn_like(v_history)
    v_target[..., 0] += 4
    v_target[..., 2] = 0
    v_target = F.normalize(v_target, dim=-1)
    v_forward = torch.sum(v_target * v_history, -1)
    v_drift = v_history - v_forward[..., None] * v_target

    loss_v_forward = (4 - v_forward).relu().mean()
    loss_v_drift = v_drift.pow(2).sum(-1).mean()

    loss_d_ctrl = (act_history[1:] - act_history[:-1]).div(ctl_dt).pow(2).sum(-1).mean()
    loss_acc = (v_history[1:] - v_history[:-1]).div(ctl_dt).pow(2).sum(-1).mean()

    distance = (p_history[:, :, None] - env.obstacles).pow(2).sum(-1).sqrt().add(-1)
    distance = torch.cat([distance, p_history[..., -1:] + 1], -1).min(-1).values
    x_l = distance.clamp(0.1, 1)
    loss_obj_avoidance = (x_l - x_l.log()).mean() - 1

    loss = loss_v_forward + loss_v_drift + loss_d_ctrl + 0.01 * loss_acc + 25 * loss_obj_avoidance

    nn.utils.clip_grad.clip_grad_norm_(model.parameters(), 0.01)
    print(loss.item())
    optim.zero_grad()
    loss.backward()
    optim.step()
    with torch.no_grad():
        writer.add_scalar('loss', loss, i)
        writer.add_scalar('loss_v_forward', loss_v_forward, i)
        writer.add_scalar('loss_v_drift', loss_v_drift, i)
        writer.add_scalar('loss_d_ctrl', loss_d_ctrl, i)
        writer.add_scalar('loss_acc', loss_acc, i)
        writer.add_scalar('loss_obj_avoidance', loss_obj_avoidance, i)
        writer.add_scalar('t', t, i)
        if (i + 1) % 500 == 0:
            vid = np.stack(vid).transpose(0, 3, 1, 2)[None]
            writer.add_video('color', vid, i, fps=5)
            fig = plt.figure()
            v_history = v_history.cpu()
            plt.plot(v_history[:, 0, 0], label='x')
            plt.plot(v_history[:, 0, 1], label='y')
            plt.plot(v_history[:, 0, 2], label='z')
            plt.legend()
            writer.add_figure('v', fig, i)
            torch.save(model.state_dict(), os.path.join(writer.logdir, f'checkpoint{i//100:04d}.pth'))

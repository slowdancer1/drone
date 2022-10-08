import os
from matplotlib import pyplot as plt
import numpy as np
from env_gl import Env, quaternion_to_forward
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import AdamW
from tensorboardX import SummaryWriter

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--resume')
args = parser.parse_args()


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Linear(16*9, 256, bias=False)
        self.v_proj = nn.Linear(6, 256)
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
model = Model().cuda()
if args.resume:
    model.load_state_dict(torch.load(args.resume, map_location='cuda'))
env = Env('cuda')
optim = AdamW(model.parameters(), 5e-4)

ctl_dt = 1 / 15

writer = SummaryWriter(flush_secs=1)
p_ctl_pts = torch.linspace(0, ctl_dt * 2, 16, device='cuda')[:, None, None]

for i in range(10000):
    env.reset()
    p_history = []
    q_history = []
    v_history = []
    act_history = []
    vid = []
    h = None
    loss_obj_avoidance = 0
    for t in range(250):
        color, depth = env.render()
        depth = torch.as_tensor(depth[:, None]).cuda()
        x = torch.clamp(1 / depth - 1, -1, 6)
        if (i + 1) % 100 == 0 and t % 3 == 0:
            vid.append(color[0].copy())
        act, h = model(x, torch.cat([env.quad.v, env.quad.w], -1), h)
        env.step(act, ctl_dt)

        p = env.quad.p + env.quad.v * p_ctl_pts
        distance = (p[:, :, None] - env.obstacles).pow(2).sum(-1).sqrt().add(-1)
        distance = torch.cat([distance, p[..., -1:] + 1], -1)
        x_l = distance.clamp(0.1, 1)
        loss_obj_avoidance += (x_l - x_l.log()).mean() - 1

        p_history.append(env.quad.p)
        q_history.append(env.quad.q)
        v_history.append(env.quad.v)
        act_history.append(act)

    p_history = torch.stack(p_history)
    q_history = torch.stack(q_history)
    v_history = torch.stack(v_history)
    act_history = torch.stack(act_history)

    v_target = torch.zeros_like(v_history)
    v_target[..., 0] = 2
    loss_v_error = (2 - v_history[..., 0]).relu().pow(2).mean()
    loss_p_error = F.mse_loss(p_history[..., 1:], v_target[..., 1:], reduction='none').sum(-1).mean()

    loss_d_ctrl = (act_history[1:] - act_history[:-1]).div(ctl_dt).pow(2).sum(-1).mean()
    loss_acc = (v_history[1:] - v_history[:-1]).div(ctl_dt).pow(2).sum(-1).mean()

    # distance = (p_history[:, None] - env.obstacles).pow(2).sum(-1).sqrt().add(-1)
    # distance = torch.cat([distance, p_history[:, -1:] + 1], -1).relu()

    # v_norm = torch.norm(v_history, dim=-1)
    # loss_look_ahead = (1 - (quaternion_to_forward(q_history) @ v_history.t()) / (v_norm + 0.1)) * v_norm
    # loss_look_ahead = loss_look_ahead.mean()
    loss_look_ahead = 0

    loss_obj_avoidance /= t + 1

    loss = loss_v_error + 0.1 * loss_p_error + 0.1 * loss_d_ctrl + 0.01 * loss_acc + 1e3 * loss_obj_avoidance + loss_look_ahead

    nn.utils.clip_grad.clip_grad_norm_(model.parameters(), 0.01)
    print(loss.item())
    optim.zero_grad()
    loss.backward()
    optim.step()
    with torch.no_grad():
        writer.add_scalar('loss', loss, i)
        writer.add_scalar('loss_v_error', loss_v_error, i)
        writer.add_scalar('loss_p_error', loss_p_error, i)
        writer.add_scalar('loss_d_ctrl', loss_d_ctrl, i)
        writer.add_scalar('loss_acc', loss_acc, i)
        writer.add_scalar('loss_obj_avoidance', loss_obj_avoidance, i)
        writer.add_scalar('loss_look_ahead', loss_look_ahead, i)
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

import os
import time
from matplotlib import pyplot as plt
import numpy as np
from env_gl import Env
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import AdamW
from tensorboardX import SummaryWriter

import argparse

from ratation import _axis_angle_rotation

parser = argparse.ArgumentParser()
parser.add_argument('--resume')
parser.add_argument('--batch_size', type=int, default=16)
args = parser.parse_args()

class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Linear(16*9, 256, bias=False)
        self.v_proj = nn.Linear(9, 256)
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

device = torch.device('cuda')
model_device = torch.device('cuda')

env = Env(args.batch_size, device)
model = Model()
model = model.to(model_device)

if args.resume:
    model.load_state_dict(torch.load(args.resume, map_location=model_device))
optim = AdamW(model.parameters(), 5e-4)

ctl_dt = 1 / 15

writer = SummaryWriter(flush_secs=1)

for i in range(20000):
    t0 = time.time()
    env.reset()
    p_history = []
    v_history = []
    act_history = []
    nearest_pt_history = []
    vid = []
    h = None
    loss_obj_avoidance = 0
    p_target = torch.stack([
        torch.rand((args.batch_size,), device=device) * 10 + 20,
        torch.rand((args.batch_size,), device=device) * 10 - 5,
        torch.full((args.batch_size,), 0, device=device)
    ], -1)

    loss_v = 0
    loss_look_ahead = 0

    for t in range(150):
        color, depth, nearest_pt = env.render()
        p_history.append(env.quad.p)
        nearest_pt_history.append(nearest_pt.copy())

        depth = torch.as_tensor(depth[:, None]).to(model_device)
        x = torch.clamp(1 / depth - 1, -1, 6)
        if i == 0 or (i + 1) % 100 == 0 and t % 3 == 0:
            vid.append(color[0].copy())
        target_v = p_target - env.quad.p
        R = _axis_angle_rotation('Z',  env.quad.w[:, -1])
        loss_look_ahead += 1 - F.cosine_similarity(R[:, :2, 0], env.quad.v[:, :2]).mean()
        target_v_norm = torch.norm(target_v, 2, -1, keepdim=True)
        target_v = target_v / target_v_norm * target_v_norm.clamp_max(6)
        local_v = torch.squeeze(env.quad.v[:, None] @ R, 1)
        local_v_target = torch.squeeze(target_v[:, None] @ R, 1)
        state = torch.cat([
            local_v,
            env.quad.w,
            local_v_target
        ], -1).to(model_device)
        act, h = model(x, state, h)
        act = act.to(device)
        env.step(act, ctl_dt)

        # loss
        loss_v += F.smooth_l1_loss(local_v, local_v_target, beta=0.1) * 3

        v_history.append(env.quad.v)
        act_history.append(act)

    p_history = torch.stack(p_history)
    v_history = torch.stack(v_history)
    act_history = torch.stack(act_history)
    nearest_pt_history = torch.as_tensor(np.stack(nearest_pt_history)).to(device)

    loss_v /= t + 1
    loss_look_ahead /= t + 1

    loss_d_ctrl = (act_history[1:] - act_history[:-1]).div(ctl_dt)
    loss_d_ctrl = loss_d_ctrl.pow(2).sum(-1).mean()

    distance = torch.norm(p_history - nearest_pt_history, 2, -1)
    x_l = distance.clamp(0.01, 1)
    loss_obj_avoidance = (x_l - x_l.log()).mean() - 1

    loss = loss_v + loss_d_ctrl + 10 * loss_obj_avoidance + loss_look_ahead

    nn.utils.clip_grad.clip_grad_norm_(model.parameters(), 0.01)
    print(f'{loss.item():.3f}, time: {time.time()-t0:.2f}s')
    optim.zero_grad()
    loss.backward()
    optim.step()
    with torch.no_grad():
        writer.add_scalar('loss', loss, i)
        writer.add_scalar('loss_v', loss_v, i)
        writer.add_scalar('loss_d_ctrl', loss_d_ctrl, i)
        writer.add_scalar('loss_look_ahead', loss_look_ahead, i)
        writer.add_scalar('loss_obj_avoidance', loss_obj_avoidance, i)
        if i == 0 or (i + 1) % 500 == 0:
            vid = np.stack(vid).transpose(0, 3, 1, 2)[None]
            writer.add_video('color', vid, i, fps=5)
            fig = plt.figure()
            v_history = v_history.cpu()
            plt.plot(v_history[:, 0, 0], label='x')
            plt.plot(v_history[:, 0, 1], label='y')
            plt.plot(v_history[:, 0, 2], label='z')
            plt.legend()
            writer.add_figure('v', fig, i)
            fig = plt.figure()
            p_history = p_history.cpu()
            plt.plot(p_history[:, 0, 0], label='x')
            plt.plot(p_history[:, 0, 1], label='y')
            plt.plot(p_history[:, 0, 2], label='z')
            plt.legend()
            writer.add_figure('p', fig, i)
            torch.save(model.state_dict(), os.path.join(writer.logdir, f'checkpoint{i//100:04d}.pth'))

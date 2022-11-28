import os
from collections import defaultdict
import time
from matplotlib import pyplot as plt
import numpy as np
from env_gl import Env
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tensorboardX import SummaryWriter
from tqdm import tqdm

import argparse
from model import Model

from rotation import _axis_angle_rotation

parser = argparse.ArgumentParser()
parser.add_argument('--resume')
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--num_iters', type=int, default=10000)
parser.add_argument('--lr', type=float, default=1e-3)
args = parser.parse_args()

device = torch.device('cuda')
model_device = torch.device('cuda')

env = Env(args.batch_size, 80, 60, device)
model = Model(10)
model = model.to(model_device)

if args.resume:
    model.load_state_dict(torch.load(args.resume, map_location=model_device))
optim = AdamW(model.parameters(), args.lr)
sched = CosineAnnealingLR(optim, args.num_iters, args.lr * 0.01)

ctl_dt = 1 / 15

writer = SummaryWriter('.')
scaler_q = defaultdict(list)
def add_scalar(k, v, i):
    scaler_q[k].append(v)
    if len(scaler_q[k]) >= 20:
        writer.add_scalar(k, sum(scaler_q[k]) / len(scaler_q[k]), i)
        scaler_q[k].clear()

def barrier(x: torch.Tensor):
    x.mul(2).clamp_max(1)
    return torch.where(x > 0.01, x - torch.log(x), -99. * (x - 0.01) + 4.61517).mean() - 1

# def barrier(x: torch.Tensor):
#     return 10 * (1 - x).relu().pow(3).mean()

states_mean = [1.882, 0.0, 0.0, 0.0, 0.0, 0.0, 3.127, 0.0, 0.0, 0.1]
states_mean = torch.tensor([states_mean], device=device)
states_std = [1.555, 0.496, 0.279, 0.073, 0.174, 0.069, 2.814, 0.596, 0.227, 0.057]
states_std = torch.tensor([states_std], device=device)

pbar = tqdm(range(args.num_iters), ncols=80)
for i in pbar:
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
        torch.rand((args.batch_size,), device=device) * 20 + 10,
        torch.rand((args.batch_size,), device=device) * 12 - 6,
        torch.full((args.batch_size,), 0, device=device)
    ], -1)

    loss_v = 0
    loss_look_ahead = 0
    margin = torch.rand((args.batch_size,), device=device) * 0.2
    max_speed = torch.rand((args.batch_size, 1), device=device) * 9 + 1

    act_buffer = [
        torch.randn((args.batch_size, 4), device=device) * 0.1,
        torch.randn((args.batch_size, 4), device=device) * 0.1,
    ]
    for t in range(150):
        color, depth, nearest_pt = env.render(ctl_dt)
        p_history.append(env.quad.p)
        nearest_pt_history.append(nearest_pt.copy())

        depth = torch.as_tensor(depth[:, None], device=model_device)
        if i == 0 or (i + 1) % 500 == 0 and t % 3 == 0:
            vid.append(color[-1].copy())
        target_v = p_target - env.quad.p
        R = _axis_angle_rotation('Z',  env.quad.w[:, -1])
        loss_look_ahead += 1 - F.cosine_similarity(R[:, :2, 0], env.quad.v[:, :2]).mean()
        target_v_norm = torch.norm(target_v, 2, -1, keepdim=True)
        target_v = target_v / target_v_norm * target_v_norm.clamp_max(max_speed)
        local_v = torch.squeeze(env.quad.v[:, None] @ R, 1)
        local_v.add_(torch.randn_like(local_v) * 0.01)
        local_v_target = torch.squeeze(target_v[:, None] @ R, 1)
        state = torch.cat([
            local_v,
            env.quad.w,
            local_v_target,
            margin[:, None]
        ], -1).to(model_device)

        # normalize
        x = 1 / depth.clamp_(0.01, 10) - 0.34
        x = F.max_pool2d(x, 5, 5)
        state = (state - states_mean) / states_std

        act, h = model(x, state, h)
        act = act.to(device)
        act_buffer.append(act)
        env.step(act_buffer.pop(0), ctl_dt)

        # loss
        local_v = torch.squeeze(env.quad.v[:, None] @ R, 1)
        loss_v += F.smooth_l1_loss(local_v, local_v_target) * 3

        v_history.append(env.quad.v)
        act_history.append(act)

    p_history = torch.stack(p_history)
    v_history = torch.stack(v_history)
    act_history = torch.stack(act_history)
    nearest_pt_history = torch.as_tensor(np.stack(nearest_pt_history), device=device)

    loss_v /= t + 1
    loss_look_ahead /= t + 1

    loss_d_ctrl = (act_history[1:] - act_history[:-1]).div(ctl_dt)
    loss_d_ctrl = loss_d_ctrl.pow(2).sum(-1).mean() + loss_d_ctrl.abs().sum(-1).mean()

    distance = torch.norm(p_history - nearest_pt_history, 2, -1)
    loss_obj_avoidance = barrier(distance - margin)

    loss = loss_v + 0.1 * loss_d_ctrl + loss_obj_avoidance + loss_look_ahead

    # nn.utils.clip_grad.clip_grad_norm_(model.parameters(), 0.01)
    pbar.set_description_str(f'loss: {loss.item():.3f}')
    optim.zero_grad()
    loss.backward()
    optim.step()
    sched.step()

    with torch.no_grad():
        add_scalar('loss', loss, i)
        add_scalar('loss_v', loss_v, i)
        add_scalar('loss_d_ctrl', loss_d_ctrl, i)
        add_scalar('loss_look_ahead', loss_look_ahead, i)
        add_scalar('loss_obj_avoidance', loss_obj_avoidance, i)
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
            torch.save(model.state_dict(), f'checkpoint{i//500:04d}.pth')

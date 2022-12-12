import math
import os
from collections import defaultdict
from random import randint
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
# from eval import eval
from rotation import _axis_angle_rotation

parser = argparse.ArgumentParser()
parser.add_argument('--resume', default=None)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--num_iters', type=int, default=20000)
parser.add_argument('--coef_v', type=float, default=1.0)
parser.add_argument('--coef_v_dri', type=float, default=0.2)
parser.add_argument('--coef_d_ctrl', type=float, default=1.0)
parser.add_argument('--coef_obj_avoidance', type=float, default=10.0)
parser.add_argument('--coef_look_ahead', type=float, default=0.01)
parser.add_argument('--coef_tgt', type=float, default=1.0)
parser.add_argument('--coef_d_acc', type=float, default=0.)
parser.add_argument('--coef_d_jerk', type=float, default=0.)
parser.add_argument('--coef_cns', type=float, default=1.0)
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--grad_decay', type=float, default=0.7)
args = parser.parse_args()
writer = SummaryWriter('.', flush_secs=1)
print(args)

device = torch.device('cuda')

env = Env(args.batch_size, 80, 60, device)
env.quad.grad_decay = args.grad_decay
model = Model()
model = model.to(device)

if args.resume:
    model.load_state_dict(torch.load(args.resume, map_location=device))
optim = AdamW(model.parameters(), args.lr)
sched = CosineAnnealingLR(optim, args.num_iters, args.lr * 0.01)

ctl_dt = 1 / 15


scaler_q = defaultdict(list)
def smooth_dict(ori_dict):
    for k, v in ori_dict.items():
        scaler_q[k].append(v)


def barrier(x: torch.Tensor):
    x = x.clamp_max(1)
    return torch.where(x > 0.01, -torch.log(x), -100. * (x - 0.01) + 4.60517).mean()
    clamp_min = 0.02
    val = clamp_min - math.log(clamp_min)
    grad = 1 - 1 / clamp_min
    return torch.where(x > clamp_min,
        x - torch.log(x), grad * (x - clamp_min) + val).mean() - 1

# def barrier(x: torch.Tensor):
#     return 10 * (1 - x).relu().pow(3).mean()

states_mean = [3.62, 0, 0, 0, 0, 4.14, 0, 0, 0.125]
states_mean = torch.tensor([states_mean], device=device)
states_std = [2.770, 0.367, 0.343, 0.080, 0.240, 4.313, 0.396, 0.327, 0.073]
states_std = torch.tensor([states_std], device=device)

pbar = tqdm(range(args.num_iters), ncols=80)
# depths = []
# states = []
for i in pbar:
    t0 = time.time()
    env.reset()
    model.reset()
    p_history = []
    v_history = []
    w_history = []
    nearest_pt_history = []
    vid = []
    h = None
    loss_obj_avoidance = 0
    p_target = torch.stack([
        torch.rand((args.batch_size,), device=device) * 22 + 10,
        torch.rand((args.batch_size,), device=device) * 12 - 6,
        torch.full((args.batch_size,), 0, device=device)
    ], -1)

    loss_v = 0
    loss_v_dri = 0
    loss_look_ahead = 0
    loss_cns = 0
    margin = torch.rand((args.batch_size,), device=device) * 0.25
    max_speed = torch.rand((args.batch_size, 1), device=device) * 9 + 3

    act_buffer = []
    for _ in range(randint(1, 3)):
        act = torch.cat([
            env.quad.w[:, :2],
            torch.zeros((args.batch_size, 2), device=device)
        ], -1)
        act_buffer.append(act)
    speed_ratios = []
    for t in range(150):
        color, depth, nearest_pt = env.render(ctl_dt)
        p_history.append(env.quad.p)
        nearest_pt_history.append(nearest_pt.copy())

        depth = torch.as_tensor(depth[:, None], device=device)
        if (i + 1) % 250 == 0:
            vid.append(color[-1].copy())
        target_v = p_target - env.quad.p.detach()
        R = _axis_angle_rotation('Z',  env.quad.w[:, -1])
        # loss_look_ahead += 1 - F.cosine_similarity(R[:, :2, 0], env.quad.v[:, :2]).mean()
        target_v_norm = torch.norm(target_v, 2, -1, keepdim=True)
        target_v_unit = target_v / target_v_norm
        target_v = target_v_unit * torch.min(target_v_norm, max_speed)
        loss_look_ahead += 1 - torch.sum(R[:, :2, 0] * env.quad.v.detach()[:, :2], -1).mean()
        local_v = torch.squeeze(env.quad.v[:, None] @ R, 1)
        local_v.add_(torch.randn_like(local_v) * 0.01)
        local_v_target = torch.squeeze(target_v[:, None] @ R, 1)
        state = torch.cat([
            local_v,
            env.quad.w[:, :2],
            local_v_target,
            margin[:, None]
        ], -1)

        # normalize
        x = 3 / depth.clamp_(0.01, 10) - 0.6
        x = F.max_pool2d(x, 5, 5)
        # states.append(state.detach())
        mirror_state = state.clone()
        mirror_state[:, [1, 3, 6]] = -mirror_state[:, [1, 3, 6]]
        state = torch.cat([state, mirror_state])
        x = torch.cat([x, torch.flip(x, (-1,))])
        state = (state - states_mean) / states_std
        # depths.append(depth.clamp_(0.01, 10).detach())
        act, h = model(x, state, h)
        act, mirror_act = torch.chunk(act, 2)
        mirror_act = mirror_act.clone()
        mirror_act[:, 0] = -mirror_act[:, 0]
        loss_cns += F.mse_loss(act, mirror_act)

        act_buffer.append(act)
        env.step(act_buffer.pop(0), ctl_dt)

        # loss
        v_forward = torch.sum(target_v_unit * env.quad.v, -1, True)
        with torch.no_grad():
            speed_ratio = v_forward.div(max_speed).clamp(0, 1)
            speed_ratio *= torch.cosine_similarity(target_v, env.quad.v)[:, None]
        speed_ratios.append(speed_ratio)
        v_drift = env.quad.v - v_forward * target_v_unit
        loss_v += F.smooth_l1_loss(v_forward, target_v_norm)
        loss_v_dri += v_drift.pow(2).sum(-1).mean(0)

        v_history.append(env.quad.v)
        w_history.append(act)

    p_history = torch.stack(p_history)
    v_history = torch.stack(v_history)
    w_history = torch.stack(w_history)
    nearest_pt_history = torch.as_tensor(np.stack(nearest_pt_history), device=device)

    loss_v /= t + 1
    loss_v_dri /= t + 1
    loss_look_ahead /= t + 1
    loss_cns /= t + 1

    d_ctrl = (w_history[1:] - w_history[:-1]).div(ctl_dt)
    loss_d_ctrl = F.mse_loss(d_ctrl,
        torch.zeros_like(d_ctrl)).mul(3)

    a_history = (v_history[1:] - v_history[:-1]).div(ctl_dt)
    jerk_history = (a_history[1:] - a_history[:-1]).div(ctl_dt)
    loss_d_acc = a_history.pow(2).sum(-1).mean()
    loss_d_jerk = jerk_history.pow(2).sum(-1).mean()

    distance = torch.norm(p_history - nearest_pt_history, 2, -1) - margin
    loss_obj_avoidance = barrier(distance)
    loss_tgt = F.smooth_l1_loss(p_history, p_target.broadcast_to(p_history.shape), reduction='none')
    loss_tgt, loss_tgt_ind = loss_tgt.sum(-1).min(0)
    loss_tgt[loss_tgt_ind == 149] = loss_tgt.detach()[loss_tgt_ind == 149]
    loss_tgt = loss_tgt.mean()

    loss = args.coef_v * loss_v + \
        args.coef_v_dri * loss_v_dri + \
        args.coef_d_ctrl * loss_d_ctrl + \
        args.coef_obj_avoidance * loss_obj_avoidance + \
        args.coef_look_ahead * loss_look_ahead + \
        args.coef_tgt * loss_tgt + \
        args.coef_d_acc * loss_d_acc + \
        args.coef_d_jerk * loss_d_jerk + \
        args.coef_cns * loss_cns
    
    if torch.isnan(loss):
        print("loss is nan, exiting...")
        exit(1)

    nn.utils.clip_grad.clip_grad_norm_(model.parameters(), 0.01)
    pbar.set_description_str(f'loss: {loss.item():.3f}')
    optim.zero_grad()
    loss.backward()
    optim.step()
    sched.step()

    with torch.no_grad():
        success = torch.all(distance > 0, 0)
        speed = torch.cat(speed_ratios, -1).max(-1).values
        mean_speed = speed.mean().item()
        smooth_dict({
            'loss': loss.item(),
            'loss_v': loss_v.item(),
            'loss_v_dri': loss_v_dri.item(),
            'loss_d_ctrl': loss_d_ctrl.item(),
            'loss_look_ahead': loss_look_ahead.item(),
            'loss_obj_avoidance': loss_obj_avoidance.item(),
            'loss_tgt': loss_tgt.item(),
            'loss_d_acc': loss_d_acc.item(),
            'loss_d_jerk': loss_d_jerk.item(),
            'loss_cns': loss_cns.item(),
            'success': success.sum().item() / args.batch_size,
            'speed': mean_speed,
            'ar': (success * speed).mean().item() * mean_speed})
        log_dict = {}
        if (i + 1) % 250 == 0:
            vid = np.stack(vid).transpose(0, 3, 1, 2)[None]
            fig_v, ax = plt.subplots()
            v_history = v_history.cpu()
            ax.plot(v_history[:, -1, 0], label='x')
            ax.plot(v_history[:, -1, 1], label='y')
            ax.plot(v_history[:, -1, 2], label='z')
            ax.legend()
            fig_p, ax = plt.subplots()
            p_history = p_history.cpu()
            ax.plot(p_history[:, -1, 0], label='x')
            ax.plot(p_history[:, -1, 1], label='y')
            ax.plot(p_history[:, -1, 2], label='z')
            ax.legend()
            fig_a, ax = plt.subplots()
            w_history = w_history.cpu()
            ax.plot(w_history[:, -1, 0], label='x')
            ax.plot(w_history[:, -1, 1], label='y')
            ax.plot(w_history[:, -1, 2], label='z')
            ax.legend()
            writer.add_video('demo', vid, i + 1, 15)
            writer.add_figure('v_history', fig_v, i + 1)
            writer.add_figure('p_history', fig_p, i + 1)
            writer.add_figure('a_history', fig_a, i + 1)
            torch.save(model.state_dict(), f'checkpoint{i//1000:04d}.pth')
        if (i + 1) % 25 == 0:
            for k, v in scaler_q.items():
                writer.add_scalar(k, sum(v) / len(v), i + 1)
            scaler_q.clear()

torch.save(model.state_dict(), 'last.pth')

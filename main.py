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
parser.add_argument('--num_iters', type=int, default=100000)
parser.add_argument('--coef_v', type=float, default=2.0)
parser.add_argument('--coef_ctrl', type=float, default=1.0)
parser.add_argument('--coef_obj_avoidance', type=float, default=5.)
parser.add_argument('--coef_look_ahead', type=float, default=0.)
parser.add_argument('--coef_tgt', type=float, default=1.)
parser.add_argument('--coef_d_acc', type=float, default=0.1)
parser.add_argument('--coef_d_jerk', type=float, default=0.01)
# parser.add_argument('--coef_cns', type=float, default=0.0)
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--grad_decay', type=float, default=0.7)
args = parser.parse_args()
writer = SummaryWriter('./log', flush_secs=1)
print(args)

device = torch.device('cuda')

env = Env(args.batch_size, 80, 60, device)
model = Model(7+9, 3)
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
    x = x.mul(2).clamp_max(1)
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

pbar = tqdm(range(args.num_iters), ncols=120)
# depths = []
# states = []
for i in pbar:
    t0 = time.time()
    model.reset()
    p_history = []
    v_history = []
    target_v_history = []
    a_reals = []
    nearest_pt_history = []
    vid = []
    h = None
    loss_obj_avoidance = 0
    target0 = torch.stack([
        torch.full((args.batch_size//4,), 24, device=device),
        torch.rand((args.batch_size//4,), device=device) * 6 - 6,
        torch.rand((args.batch_size//4,), device=device),
    ], -1)
    target1 = torch.stack([
        torch.full((args.batch_size//4,), 24, device=device),
        torch.rand((args.batch_size//4,), device=device) * 6,
        torch.rand((args.batch_size//4,), device=device),
    ], -1)
    target2 = torch.stack([
        torch.rand((args.batch_size//4,), device=device),
        torch.rand((args.batch_size//4,), device=device) * 6,
        torch.rand((args.batch_size//4,), device=device),
    ], -1)
    target3 = torch.stack([
        torch.rand((args.batch_size//4,), device=device),
        torch.rand((args.batch_size//4,), device=device) * 6 - 6,
        torch.rand((args.batch_size//4,), device=device),
    ], -1)
    p_target = torch.cat([target0,target1,target2,target3], 0)
    drone_p = torch.cat([target2,target3,target0,target1], 0)
    env.reset(p_target, drone_p)
    env.quad.grad_decay = args.grad_decay

    loss_v = 0
    # loss_cns = 0
    margin = torch.rand((args.batch_size,), device=device) * 0.25
    max_speed = torch.rand((args.batch_size, 1), device=device) * 4 + 4

    act_buffer = [torch.zeros_like(env.quad.v)] * randint(1, 2)
    speed_ratios = []
    collide_drone = np.zeros((64))
    for t in range(120):
        color, depth, nearest_pt, obstacle_pt = env.render(ctl_dt, drone_p)
        for j in range(args.batch_size):
            for k in range(4):
                if torch.norm(torch.tensor(obstacle_pt[j][k]).cuda() - drone_p[j], 2, -1) < margin[j]:
                    collide_drone[j] = 1
        drone_p = env.quad.p.clone()
        # print(drone_p[0],drone_p[16],drone_p[32],drone_p[48])
        # print(obstacle_pt[0,:4],obstacle_pt[16,:4],obstacle_pt[32,:4],obstacle_pt[48,:4],)
        # print()
        p_history.append(env.quad.p)
        nearest_pt_history.append(nearest_pt.copy())

        depth = torch.as_tensor(depth[:, None], device=device)
        if (i + 1) % 250 == 0:
            vid.append(color[-1].copy())
        target_v = p_target - env.quad.p.detach()
        R = torch.stack([
            env.quad.forward_vec,
            env.quad.left_vec,
            env.quad.up_vec,
        ], -1)
        target_v_norm = torch.norm(target_v, 2, -1, keepdim=True)
        target_v_unit = target_v / target_v_norm
        target_v = target_v_unit * torch.min(target_v_norm, max_speed)
        with torch.no_grad():
            state = torch.cat([
                torch.squeeze(env.quad.v[:, None] @ R, 1),
                torch.squeeze(target_v[:, None] @ R, 1),
                R.flatten(1),
                margin[:, None]
            ], -1)

        # normalize
        x = 3 / depth.clamp_(0.01, 10) - 0.6
        x = F.max_pool2d(x, 5, 5)
        act, h = model(x, state, h)

        a_pred = (R @ act.unsqueeze(-1)).squeeze(-1)
        act_buffer.append(a_pred - env.quad.v)
        a_pred = act_buffer.pop(0)
        env.step(a_pred, ctl_dt)

        # loss
        with torch.no_grad():
            v_forward = torch.sum(target_v_unit * env.quad.v, -1, True)
            speed_ratio = v_forward.div(max_speed).clamp(0, 1)
            speed_ratio *= torch.cosine_similarity(target_v, env.quad.v)[:, None]
        speed_ratios.append(speed_ratio)

        a_reals.append(env.quad.a)
        v_history.append(env.quad.v)
        target_v_history.append(target_v)

    p_history = torch.stack(p_history)
    a_reals = torch.stack(a_reals)
    nearest_pt_history = torch.as_tensor(np.stack(nearest_pt_history), device=device)

    v_history = torch.stack(v_history)
    target_v_history = torch.stack(target_v_history)
    T, B, _ = v_history.shape
    v_history_cum = torch.cumsum(torch.cat([
        torch.zeros((1, B, 3), device=device),
        v_history,
        v_history.detach()[-1:].repeat(14, 1, 1)]), 0)
    v_history_avg = (v_history_cum[15:] - v_history_cum[:-15]) / 15
    delta_v = torch.norm(v_history_avg - target_v_history, 2, -1)
    loss_v = F.smooth_l1_loss(delta_v, torch.zeros_like(delta_v), beta=0.1)

    jerk_history = (a_reals[1:] - a_reals[:-1]).div(ctl_dt)
    loss_d_acc = a_reals.pow(2).sum(-1).mean()
    loss_d_jerk = jerk_history.pow(2).sum(-1).mean()

    vec_to_pt = nearest_pt_history - p_history
    distance = torch.norm(vec_to_pt, 2, -1)
    dir_to_pt = vec_to_pt.detach() / (distance.detach() + 1e-5)[..., None]
    v_distance = torch.sum(v_history * dir_to_pt, -1, True)
    pts = torch.linspace(0, ctl_dt, 10, device=device)
    distance = distance - margin
    loss_obj_avoidance = barrier(distance[..., None] - v_distance * pts)

    loss_tgt = F.smooth_l1_loss(p_history, p_target.broadcast_to(p_history.shape), beta=0.05, reduction='none')
    loss_tgt, loss_tgt_ind = loss_tgt.sum(-1).min(0)
    invalid_mask = (loss_tgt_ind == 149) | (loss_tgt > 1)
    loss_tgt[invalid_mask] = loss_tgt.detach()[invalid_mask]
    loss_tgt = loss_tgt.mean()

    loss = args.coef_v * loss_v + \
        args.coef_obj_avoidance * loss_obj_avoidance + \
        args.coef_tgt * loss_tgt + \
        args.coef_d_acc * loss_d_acc + \
        args.coef_d_jerk * loss_d_jerk
        # args.coef_cns * loss_cns
    
    if torch.isnan(loss):
        print("loss is nan, exiting...")
        exit(1)


    with torch.no_grad():
        success = torch.all(distance > 0, 0)
        speed = torch.cat(speed_ratios, -1).max(-1).values
        _success = success.sum().item() / args.batch_size
        smooth_dict({
            'loss': loss.item(),
            'loss_v': loss_v.item(),
            'loss_obj_avoidance': loss_obj_avoidance.item(),
            'loss_tgt': loss_tgt.item(),
            'loss_d_acc': loss_d_acc.item(),
            'loss_d_jerk': loss_d_jerk.item(),
            'success': _success,
            'speed': speed.mean().item(),
            'ar': (success * speed).mean().item() * _success})
        log_dict = {}
        if (i + 1) % 250 == 0:
            vid = np.stack(vid).transpose(0, 3, 1, 2)[None]
            fig_p, ax = plt.subplots()
            p_history = p_history.cpu()
            ax.plot(p_history[:, -1, 0], label='x')
            ax.plot(p_history[:, -1, 1], label='y')
            ax.plot(p_history[:, -1, 2], label='z')
            ax.legend()
            fig_a, ax = plt.subplots()
            a_reals = a_reals.cpu()
            ax.plot(a_reals[:, -1, 0], label='x')
            ax.plot(a_reals[:, -1, 1], label='y')
            ax.plot(a_reals[:, -1, 2], label='z')
            ax.legend()
            writer.add_video('demo', vid, i + 1, 15)
            writer.add_figure('p_history', fig_p, i + 1)
            writer.add_figure('a_reals', fig_a, i + 1)
            torch.save(model.state_dict(), f'checkpoint{i//1000:04d}.pth')
        if (i + 1) % 25 == 0:
            for k, v in scaler_q.items():
                writer.add_scalar(k, sum(v) / len(v), i + 1)
            scaler_q.clear()

    nn.utils.clip_grad.clip_grad_norm_(model.parameters(), 0.01)
    collide_drone = collide_drone.sum() / args.batch_size
    pbar.set_description_str(f'loss: {loss.item():.3f} success: {_success:.3f} collide_drone: {collide_drone:.3f}')
    optim.zero_grad()
    loss.backward()
    optim.step()
    sched.step()
torch.save(model.state_dict(), 'last.pth')

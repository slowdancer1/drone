import math
import os
from collections import defaultdict
from random import randint
import time
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from env_gl import Env
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from tqdm import tqdm

import argparse
from model import Model
# from eval import eval
from rotation import _axis_angle_rotation

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=4)
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
print(args)

device = torch.device('cuda')

env = Env(args.batch_size, 80, 60, device, True)
model = Model(7+9, 3).eval()
model = model.to(device)

model.load_state_dict(torch.load('r4.pth', map_location=device))

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

pbar = tqdm(range(args.num_iters), ncols=80)
# depths = []
# states = []
# fig = plt.figure(figsize=(20, 20))
# plt.ion()
# ax = plt.axes(projection='3d')
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

    # plt.cla()
    # last_p = [0,0,0,0]
    # quad_p = [0,0,0,0]
    for t in range(150):
        drone_p = env.quad.p.clone()
        color, depth, nearest_pt, obstacle_pt = env.render(ctl_dt, drone_p)
        # print(drone_p)
        # print(obstacle_pt[:,:4])
        # print()

        # ax.set_xlim(-10,30)
        # ax.set_ylim(-10,10)
        # ax.set_zlim(-5,5)
        # #plt.axis('off')
        # for k in range(4):
        #     target = p_target[k].cpu().detach().numpy()
        #     last_p[k] = quad_p[k]
        #     quad_p[k] = env.quad.p[k].cpu().detach().numpy()
        #     obstacle = obstacle_pt[k][4:]
        #     if t>0:
        #         ax.scatter(target[0], target[1], target[2], linewidths=5, marker='o', label='Target', color='green')
        #         ax.plot([last_p[k][0],quad_p[k][0]], [last_p[k][1],quad_p[k][1]],[last_p[k][2],quad_p[k][2]], label='Drone', color='red')
        #         ax.scatter(obstacle[:,0], obstacle[:,1], obstacle[:,2], linewidths=5, marker='x', label='Obstacle', color='black')


        # plt.cla()
        # ax.set_xlim(-10,30)
        # ax.set_ylim(-10,10)
        # ax.set_zlim(-5,5)
        # #plt.axis('off')
        # for k in range(4):
        #     target = p_target[k].cpu().detach().numpy()
        #     ax.scatter(target[0], target[1], target[2], linewidths=5, marker='o', label='Target', color='green')
        #     quad_p = env.quad.p[k].cpu().detach().numpy()
        #     ax.scatter(quad_p[0], quad_p[1], quad_p[2], linewidths=5, marker='d', label='Drone', color='red')
        #     obstacle = obstacle_pt[k][4:]
        #     ax.scatter(obstacle[:,0], obstacle[:,1], obstacle[:,2], linewidths=5, marker='x', label='Obstacle', color='black')
        
        #plt.pause(0.05)


        p_history.append(env.quad.p)
        nearest_pt_history.append(nearest_pt.copy())

        depth = torch.as_tensor(depth[:, None], device=device)
        
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

    
    #plt.ioff()
    #plt.show()
    plt.pause(3)

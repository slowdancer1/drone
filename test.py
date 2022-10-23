import json
import os
import cv2
from matplotlib import pyplot as plt
import numpy as np
from env_gl import EnvRenderer, run
import torch
from torch import nn
from torch.nn import functional as F

import argparse

# torch.set_grad_enabled(False)

parser = argparse.ArgumentParser()
parser.add_argument('--resume')
parser.add_argument('--demo', action='store_true')
args = parser.parse_args()


batch_size = 1
class QuadState:
    def __init__(self, device) -> None:
        self.p = torch.zeros((batch_size, 3), device=device)
        self.q = torch.zeros((batch_size, 4), device=device)
        self.q[:, 0] = 1
        self.v = -torch.randn((batch_size, 3), device=device) * 0.01
        self.w = torch.zeros((batch_size, 3), device=device)
        self.a = torch.zeros((batch_size, 3), device=device)
        self.g = torch.zeros((batch_size, 3), device=device)
        self.g[:, 2] -= 9.80665
        self.thrust = torch.randn((batch_size, 1), device=device) * 0. + 9.80665

        self.rate_ctl_delay = 0.2

    def run(self, action, ctl_dt=1/15):
        self.p, self.v, self.q, self.w = run(
            self.p, self.v, self.q, self.w, self.g, self.thrust, action, ctl_dt, self.rate_ctl_delay)

    def stat(self):
        print("p:", self.p.tolist())
        print("v:", self.v.tolist())
        print("q:", self.q.tolist())
        print("w:", self.w.tolist())


class Env:
    def __init__(self, device) -> None:
        torch.manual_seed(0)
        self.device = device
        self.r = EnvRenderer(batch_size)
        self.reset()

    def reset(self):
        self.quad = QuadState(self.device)
        self.obstacles = torch.stack([
            torch.rand((batch_size, 40), device=self.device) * 30 + 5,
            torch.rand((batch_size, 40), device=self.device) * 10 - 5,
            torch.rand((batch_size, 40), device=self.device) * 8 - 2
        ], -1)
        self.r.set_obstacles(self.obstacles.cpu().numpy())

    @torch.no_grad()
    def render(self):
        state = torch.cat([self.quad.p, self.quad.q], -1)
        color, depth = self.r.render(state.cpu().numpy())
        return color, depth

    def step(self, action, ctl_dt=1/15):
        self.quad.run(action, ctl_dt)


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
model = Model().eval()
if args.resume:
    model.load_state_dict(torch.load(args.resume, map_location='cpu'))
env = Env('cpu')

ctl_dt = 1 / 15

for i in range(1):
    real_traj = []
    env.reset()
    h = None
    loss_obj_avoidance = 0
    for t in range(30*15):
        color, depth = env.render()
        depth = torch.as_tensor(depth[:, None])
        x = torch.clamp(1 / depth - 1, -1, 6)
        state = torch.cat([env.quad.v, env.quad.w], -1)
        act, h = model(x, state, h)
        env.step(act, ctl_dt)

        time = t / 15
        if env.quad.p[0, 0] > 35:
            print("done", time)
            with open("env_result.txt", 'a') as f:
                f.write(f'{time}\n')
            break
        no_colision = torch.all(torch.norm(env.quad.p - env.obstacles[0], 2, -1) > 1)
        if not no_colision:
            print("fail")
            with open("env_result.txt", 'a') as f:
                f.write(f'30\n')
            break
        if args.demo and t%3==0:
            cv2.imwrite(f'{t//3:03d}.png', color[0])
        x, y, z = env.quad.p[0].tolist()
        real_traj.append([x, y, z + 1])

    with open("log.json", 'w') as f:
        json.dump({
            'real_traj': real_traj
        }, f)
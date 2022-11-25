from matplotlib import pyplot as plt
import torch

import argparse

from env_gl import Env
from model import Model
from rotation import _axis_angle_rotation

# torch.set_grad_enabled(False)

parser = argparse.ArgumentParser()
parser.add_argument('--resume')
parser.add_argument('--demo', action='store_true')
args = parser.parse_args()

# model = Model()
model = Model().eval()
if args.resume:
    model.load_state_dict(torch.load(args.resume, map_location='cpu'))
env = Env(1, 'cpu')

ctl_dt = 1 / 15

real_traj = []
nearest_pts = []
env.reset()
env.quad.v[:] = 0
env.quad.v[:] = 0
h = None
loss_obj_avoidance = 0
p_target = torch.tensor([30, 0, 0])
for t in range(20*15):
    color, depth, nearest_pt = env.render()
    depth = torch.as_tensor(depth[:, None])
    x = torch.clamp(1 / depth - 1, -1, 6)
    target_v = p_target - env.quad.p
    R = _axis_angle_rotation('Z',  env.quad.w[:, -1])
    target_v_norm = torch.norm(target_v, 2, -1, keepdim=True)
    target_v = target_v / target_v_norm * target_v_norm.clamp_max(6)
    local_v = torch.squeeze(env.quad.v[:, None] @ R, 1)
    local_v_target = torch.squeeze(target_v[:, None] @ R, 1)
    state = torch.cat([
        local_v,
        env.quad.w,
        local_v_target
    ], -1)
    act, h = model(x, state, h)
    env.step(act, ctl_dt)
    nearest_pts.append(nearest_pt[0].tolist())
    x, y, z = env.quad.p[0].tolist()
    real_traj.append([x, y, z])

plt.figure(figsize=(3, 10))
plt.scatter(p_target[1], p_target[0], marker='o', label='Target')
x, y, _ = zip(*real_traj)
plt.plot(y, x, label='Trajectory')
x, y, _ = zip(*[p for p in nearest_pts if p[2] > 0])
plt.scatter(y, x, marker='x', label='Obstacle', color='black')
plt.legend()
plt.savefig('demo.png')

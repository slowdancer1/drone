from matplotlib import pyplot as plt
import torch
import torch.nn.functional as F

import argparse
import cv2

from env_gl import Env
from model import Model
from rotation import _axis_angle_rotation

# torch.set_grad_enabled(False)

states_mean = [1.882, 0.0, 0.0, 0.0, 0.0, 3.127, 0.0, 0.0, 0.125]
states_mean = torch.tensor([states_mean])
states_std = [1.555, 0.496, 0.279, 0.073, 0.174, 2.814, 0.596, 0.227, 0.073]
states_std = torch.tensor([states_std])

parser = argparse.ArgumentParser()
parser.add_argument('--resume')
parser.add_argument('--demo', action='store_true')
args = parser.parse_args()

# model = Model()
model = Model().eval()
if args.resume:
    model.load_state_dict(torch.load(args.resume, map_location='cpu'))
env = Env(1, 80, 60, 'cpu')

ctl_dt = 1 / 15

real_traj = []
nearest_pts = []
env.reset()
env.quad.p[:] = 0
env.quad.v[:] = 0
env.quad.w[:] = 0
h = None
loss_obj_avoidance = 0
p_target = torch.tensor([30, 0, 0])
margin = torch.tensor([0.125])
act_buffer = [torch.zeros(1, 4)] * 2
for t in range(20*15):
    color, depth, nearest_pt = env.render(ctl_dt)
    
    cv2.imwrite(f'figs/{t:03d}.jpg', color[0].copy())
    depth = torch.as_tensor(depth[:, None])
    target_v = p_target - env.quad.p
    R = _axis_angle_rotation('Z',  env.quad.w[:, -1])
    target_v_norm = torch.norm(target_v, 2, -1, keepdim=True)
    target_v = target_v / target_v_norm * target_v_norm.clamp_max(6)
    local_v = torch.squeeze(env.quad.v[:, None] @ R, 1)
    local_v_target = torch.squeeze(target_v[:, None] @ R, 1)
    state = torch.cat([
        local_v,
        env.quad.w[:, :2],
        local_v_target,
        margin[:, None]
    ], -1)
    print(*state[0].tolist())

    # normalize
    x = 1 / depth.clamp_(0.01, 10) - 0.34
    x = F.max_pool2d(x, 5, 5)
    state = (state - states_mean) / states_std

    act, h = model(x, state, h)
    act[:, 2] += env.quad.w[:, 2]
    act_buffer.append(act)
    env.step(act_buffer.pop(0), ctl_dt)
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

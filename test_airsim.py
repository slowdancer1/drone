import argparse
from time import sleep, time
import airsim
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_grad_enabled(False)

from ratation import _axis_angle_rotation, matrix_to_euler_angles, quaternion_to_matrix


parser = argparse.ArgumentParser()
parser.add_argument('--resume')
parser.add_argument('--demo', action='store_true')
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
        x = F.max_pool2d(x, 16, 16)
        x = (self.stem(x.flatten(1)) + self.v_proj(v)).relu()
        x = self.drop(x)
        hx = self.gru(x, hx)
        return self.fc(self.drop(hx)).tanh(), hx


# model = Model()
model = Model().eval()
if args.resume:
    model.load_state_dict(torch.load(args.resume, map_location='cpu'))

# connect to the AirSim simulator
client = airsim.MultirotorClient()
client.reset()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

# Async methods returns Future. Call join() to wait for task to complete.
client.takeoffAsync().join()


p_target = torch.as_tensor([24., -6, 2])
h = None
while True:
    t0 = time()
    state = client.getMultirotorState()

    q = state.kinematics_estimated.orientation
    q = torch.as_tensor([q.w_val, q.x_val, -q.y_val, -q.z_val])

    p = state.kinematics_estimated.position
    p = torch.as_tensor([p.x_val, -p.y_val, -p.z_val])

    v = state.kinematics_estimated.linear_velocity
    v = torch.as_tensor([v.x_val, -v.y_val, -v.z_val])


    rpy = matrix_to_euler_angles(quaternion_to_matrix(q), "XYZ")

    # take images
    responses = client.simGetImages([
        # airsim.ImageRequest("0", airsim.ImageType.DepthVis),
        airsim.ImageRequest("1", airsim.ImageType.DepthPlanar, True)])
    depth = airsim.get_pfm_array(responses[0])
    depth = torch.as_tensor(depth)[None, None]

    x = torch.clamp(1 / depth - 1, -1, 6)
    target_v = p_target - p
    if torch.norm(target_v) < 6:
        p_target = torch.as_tensor([100., -6, 2])
    R = _axis_angle_rotation('Z',  rpy[None, -1])
    target_v_norm = torch.norm(target_v, 2, -1, keepdim=True)
    target_v = target_v / target_v_norm * target_v_norm.clamp_max(6)
    local_v = torch.squeeze(v[None, None] @ R, 1)
    local_v_target = torch.squeeze(target_v[None, None] @ R, 1)
    state = torch.cat([
        local_v,
        rpy[None],
        local_v_target
    ], -1)
    act, h = model(x, state, h)
    r, p, y, c = act[0].tolist()
    client.moveByRollPitchYawrateThrottleAsync(r, p, y, (c + 1) / 2, 0.5)

    sleep(max(0, 1 / 15 - time() + t0))
    print(1 / (time() - t0), torch.norm(target_v))

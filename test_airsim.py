import argparse
from time import sleep, time
import airsim
from airsim.types import Pose, Vector3r, Quaternionr
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_grad_enabled(False)

from rotation import _axis_angle_rotation, matrix_to_euler_angles, quaternion_to_matrix
from model import Model


parser = argparse.ArgumentParser()
parser.add_argument('--resume')
parser.add_argument('--demo', action='store_true')
args = parser.parse_args()


device = torch.device('cpu')
model = Model().eval()
if args.resume:
    model.load_state_dict(torch.load(args.resume, map_location='cpu'))
model.to(device)

# connect to the AirSim simulator
client = airsim.MultirotorClient()
client.reset()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)
client.takeoffAsync()

# target = -20, 18.341997146606445, -3.7896406650543213
waypoints = [
    [-22, 18., -3.],
    [-20, 36, -3],
    [-60, 36, -3],
    [-60, 18, -3],
    [-22, 18., -3.],
    [-20, 36, -3],
    [-60, 36, -3],
    [-60, 18, -3],
    [-22, 18., -3.],
]

# waypoints = [
#     [-210, -250, -2],
#     [210, -250, -2],
# ]

client.simSetVehiclePose(Pose(
    Vector3r(*waypoints.pop(0)),
    Quaternionr(0, 0, 0, 1)),
    ignore_collision=True)
sleep(0.5)

# Async methods returns Future. Call join() to wait for task to complete.

client.startRecording()

states_mean = [3.62, 0, 0, 0, 0, 4.14, 0, 0, 0.125]
states_mean = torch.tensor([states_mean], device=device)
states_std = [2.770, 0.367, 0.343, 0.080, 0.240, 4.313, 0.396, 0.327, 0.073]
states_std = torch.tensor([states_std], device=device)

x, y, z = target = waypoints.pop(0)
p_target = torch.as_tensor([x, -y, -z])
h = None
margin = torch.tensor([0.2])
while True:
    t0 = time()
    state = client.getMultirotorState()

    q = state.kinematics_estimated.orientation
    q = torch.as_tensor([q.w_val, q.x_val, -q.y_val, -q.z_val])

    p = state.kinematics_estimated.position
    p = torch.as_tensor([p.x_val, -p.y_val, -p.z_val])

    v = state.kinematics_estimated.linear_velocity
    v = torch.as_tensor([v.x_val, -v.y_val, -v.z_val])

    rpy = matrix_to_euler_angles(quaternion_to_matrix(q), "ZYX")[[2, 1, 0]]

    # take images
    responses = client.simGetImages([
        # airsim.ImageRequest("0", airsim.ImageType.DepthVis),
        airsim.ImageRequest("front_center_custom", airsim.ImageType.DepthPlanar, True)])
    depth = airsim.get_pfm_array(responses[0])
    depth = torch.as_tensor(depth)[None, None].to(device)

    target_v = p_target - p
    # if torch.norm(target_v) < 8:
    #     p_target = torch.as_tensor([250., -10, 2])
    if torch.norm(target_v) < 2:
        if waypoints:
            x, y, z = target = waypoints.pop(0)
            p_target = torch.as_tensor([x, -y, -z])
        else:
            break
    R = _axis_angle_rotation('Z',  rpy[None, -1])
    target_v_norm = torch.norm(target_v, 2, -1, keepdim=True)
    target_v = target_v / target_v_norm * target_v_norm.clamp_max(6)
    local_v = torch.squeeze(v[None, None] @ R, 1)
    local_v_target = torch.squeeze(target_v[None, None] @ R, 1)
    state = torch.cat([
        local_v,
        rpy[None, :2],
        local_v_target,
        margin[None]
    ], -1).to(device)

    # normalize
    x = 3 / depth.clamp_(0.01, 10) - 0.6
    x = F.adaptive_max_pool2d(x, (12, 16))
    _s = state[0].tolist()
    state = (state - states_mean) / states_std

    act, h = model(x, state, h)
    r, p, y, c = act[0].tolist()
    client.moveByRollPitchYawThrottleAsync(r, p, rpy[2].item() + y, (c + 1) / 2, 0.5)

    sleep(max(0, 2 / 15 - time() + t0))
    print([*_s, r, p, y, c, 1 / (time() - t0)])

client.stopRecording()

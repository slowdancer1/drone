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
model = Model(7, 6).eval()
if args.resume:
    model.load_state_dict(torch.load(args.resume, map_location='cpu'))
model.to(device)

# connect to the AirSim simulator
client = airsim.MultirotorClient()
client.reset()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)
# client.takeoffAsync()

# target = -20, 18.341997146606445, -3.7896406650543213
# waypoints = [
#     [-22, 18., -3.],
#     [-20, 36, -3],
#     [-60, 36, -3],
#     [-60, 18, -3],
#     [-22, 18., -3.],
#     [-20, 36, -3],
#     [-60, 36, -3],
#     [-60, 18, -3],
#     [-22, 18., -3.],
# ]

waypoints = [
    [-210, -250, -2],
    [210, -250, -2],
]

client.moveByRollPitchYawThrottleAsync(0, 0, 0, 0.593, 0.1)
client.simSetVehiclePose(Pose(
    Vector3r(*waypoints.pop(0)),
    Quaternionr(0, 0, 0, 1)),
    ignore_collision=True)
sleep(0.5)

# Async methods returns Future. Call join() to wait for task to complete.

client.startRecording()

# states_mean = [3.62, 0, 0, 0, 0, 4.14, 0, 0, 0.125]
# states_mean = torch.tensor([states_mean], device=device)
# states_std = [2.770, 0.367, 0.343, 0.080, 0.240, 4.313, 0.396, 0.327, 0.073]
# states_std = torch.tensor([states_std], device=device)
states_mean = 0
states_std = 1

x, y, z = target = waypoints.pop(0)
p_target = torch.as_tensor([x, -y, -z])
h = None
margin = torch.tensor([0.3])
state = client.getMultirotorState()
q = state.kinematics_estimated.orientation
q = torch.as_tensor([q.w_val, q.x_val, -q.y_val, -q.z_val])
forward = quaternion_to_matrix(q)[:, 0]

while True:
    t0 = time()
    state = client.getMultirotorState()

    p = state.kinematics_estimated.position
    p = torch.as_tensor([p.x_val, -p.y_val, -p.z_val])

    v = state.kinematics_estimated.linear_velocity
    v = torch.as_tensor([v.x_val, -v.y_val, -v.z_val])

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

    target_v_norm = torch.norm(target_v, 2, -1, keepdim=True)
    target_v = target_v / target_v_norm * target_v_norm.clamp_max(6)
    state = torch.cat([
        v,
        target_v,
        margin
    ], -1)[None].to(device)

    # normalize
    x = 3 / depth.clamp_(0.01, 10) - 0.6
    x = F.adaptive_max_pool2d(x, (12, 16))
    _s = state[0].tolist()
    state = (state - states_mean) / states_std

    act, h = model(x, state, h)
    a_pred = act[0, 3:]
    # print(act[0].tolist())
    a_pred[2] += 9.80665
    a_norm = torch.norm(a_pred)
    forward += v
    forward /= torch.norm(forward)
    yaw = torch.atan2(forward[1], forward[0])
    c, s = torch.cos(yaw), torch.sin(yaw)
    a_fwd = a_pred[0] * c + a_pred[1] * s
    a_left = a_pred[0] * s - a_pred[1] * c
    pitch = torch.atan2(a_fwd, a_pred[2])
    roll = torch.asin(a_left / a_norm)
    roll, pitch, yaw, a_norm = roll.item(), pitch.item(), yaw.item(), a_norm.item()
    client.moveByRollPitchYawThrottleAsync(roll, pitch, yaw, a_norm / 9.80665 * 0.593, 0.1)
    print(target_v.tolist(), roll, pitch, yaw)
    # client.moveByRollPitchYawThrottleAsync(0, 0, 0, 0.593, 0.5)

    # sleep(max(0, 1 / 15 - time() + t0))
    # print([*_s, r, p, y, c, 1 / (time() - t0)])

client.stopRecording()

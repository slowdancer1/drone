import argparse
from time import sleep, time
import airsim
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_grad_enabled(False)



parser = argparse.ArgumentParser()
parser.add_argument('--resume')
parser.add_argument('--demo', action='store_true')
args = parser.parse_args()

# connect to the AirSim simulator
client = airsim.MultirotorClient()
client.reset()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

cam_obj = client.simListSceneObjects('PlayerCameraManager_\d+')[0]
pose = client.simGetObjectPose(cam_obj)
pos = pose.position
q = pose.orientation
pos = pos.x_val, pos.y_val, pos.z_val, q.w_val, q.x_val, q.y_val, q.z_val
print(pos)

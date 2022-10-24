from time import time
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import quadsim

from ratation import (
    axis_angle_to_quaternion,
    quaternion_multiply,
    quaternion_raw_multiply,
    quaternion_to_up,
    quaternion_to_yaw)


class EnvRenderer(quadsim.Env):
    def render(self, cameras):
        z_near = 0.01
        z_far = 10.0
        color, depth = super().render(cameras)
        color = np.flip(color, 1)
        depth = np.flip(2 * depth - 1, 1)
        depth = (2.0 * z_near * z_far) / (z_far + z_near - depth * (z_far - z_near))
        return color, depth


@torch.jit.script
def run(self_p, self_v, self_q, g, thrust, action, ctl_dt:float, rate_ctl_delay):
    alpha = 0.6 ** ctl_dt
    self_p = alpha * self_p + (1 - alpha) * self_p.detach()
    self_v = alpha * self_v + (1 - alpha) * self_v.detach()
    self_q = alpha * self_q + (1 - alpha) * self_q.detach()

    w = action[:, :3].clone()
    w[:, 2] += quaternion_to_yaw(self_q)
    mask = torch.eye(3, device=action.device)
    roll, pitch, yaw = torch.unbind(w.unsqueeze(-1).repeat(1, 1, 3) * mask, -1)
    q1 = axis_angle_to_quaternion(yaw)
    q2 = axis_angle_to_quaternion(pitch)
    q3 = axis_angle_to_quaternion(roll)
    q = quaternion_multiply(quaternion_raw_multiply(q1, q2), q3)
    c = action[:, 3:] + 1

    alpha = rate_ctl_delay ** (ctl_dt / rate_ctl_delay)

    self_q = F.normalize(q * (1 - alpha) + self_q * alpha)
    up_vec = quaternion_to_up(self_q)
    _a = up_vec * c * thrust + g - 0.1 * self_v * torch.norm(self_v, -1)

    self_v = self_v + _a * ctl_dt
    self_p = self_p + self_v * ctl_dt
    return self_p, self_v, self_q


class QuadState:
    def __init__(self, batch_size, device) -> None:
        self.p = torch.zeros((batch_size, 3), device=device)
        self.q = torch.zeros((batch_size, 4), device=device)
        self.q[:, 0] = 1
        self.v = torch.randn((batch_size, 3), device=device)
        self.a = torch.zeros((batch_size, 3), device=device)
        self.g = torch.randn((batch_size, 3), device=device) * 0.1
        self.g[:, 2] -= 9.80665
        self.thrust = torch.randn((batch_size, 1), device=device) + 9.80665

        self.rate_ctl_delay = 0.1 + 0.2 * torch.rand((batch_size, 1), device=device)

    def run(self, action, ctl_dt=1/15):
        self.p, self.v, self.q = run(
            self.p, self.v, self.q, self.g, self.thrust, action, ctl_dt, self.rate_ctl_delay)

    def stat(self):
        print("p:", self.p.tolist())
        print("v:", self.v.tolist())
        print("q:", self.q.tolist())


class Env:
    def __init__(self, batch_size, device='cpu') -> None:
        self.device = device
        self.batch_size = batch_size
        self.r = EnvRenderer(batch_size)
        self.reset()

    def reset(self):
        self.quad = QuadState(self.batch_size, self.device)
        self.obstacles = torch.stack([
            torch.rand((self.batch_size, 40), device=self.device) * 30 + 5,
            torch.rand((self.batch_size, 40), device=self.device) * 10 - 5,
            torch.rand((self.batch_size, 40), device=self.device) * 8 - 2
        ], -1)
        self.r.set_obstacles(self.obstacles.cpu().numpy())

    @torch.no_grad()
    def render(self):
        state = torch.cat([self.quad.p, self.quad.q], -1)
        color, depth = self.r.render(state.cpu().numpy())
        return color, depth

    def step(self, action, ctl_dt=1/15):
        self.quad.run(action, ctl_dt)


@torch.no_grad()
def main():
    env = Env('cpu')
    color, depth = env.render()
    plt.imshow(depth)
    plt.show()
    t0 = time()
    for _ in range(250):
        # w = torch.tensor([0., 0, 0, 0])
        # env.step(w)
        color, depth = env.render()
    print(time() - t0)
    return

    while True:
        color, depth = env.render()
        depth = np.nan_to_num(1 / depth, False, 0, 0, 0)
        x = torch.clamp(torch.as_tensor(depth) - 1, -1, 5)
        plt.imshow(x)
        plt.show()
        w = torch.tensor([0, 0, 2*torch.pi, 0])
        env.step(w, 1.1)
        env.quad.stat()


if __name__ == '__main__':
    main()


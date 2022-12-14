import math
from time import time
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import quadsim


class EnvRenderer(quadsim.Env):
    def render(self, cameras, ctl_dt):
        z_near = 0.01
        z_far = 10.0
        color, depth, nearest_pt = super().render(cameras, ctl_dt, True)
        n, h, n, w, c = color.shape
        color = np.flip(color, 1)
        color = np.transpose(color, (0, 2, 1, 3, 4)).reshape(n**2, h, w, c)
        depth = np.flip(2 * depth - 1, 1)
        depth = np.transpose(depth, (0, 2, 1, 3)).reshape(n**2, h, w)
        depth = (2.0 * z_near * z_far) / (z_far + z_near - depth * (z_far - z_near))
        return color, depth, nearest_pt


def run(self_p, self_v, self_w, g, thrust, action, ctl_dt:float, drag, rate_ctl_delay, grad_decay:float=0.8):
    alpha = grad_decay ** ctl_dt
    self_p = alpha * self_p + (1 - alpha) * self_p.detach()
    self_v = alpha * self_v + (1 - alpha) * self_v.detach()
    self_w = alpha * self_w + (1 - alpha) * self_w.detach()

    alpha = rate_ctl_delay ** (ctl_dt / rate_ctl_delay)
    action = action.clone()
    action[:, 2] = self_w[:, 2] + action[:, 2] * ctl_dt * 3
    self_w = action[:, :3] * (1 - alpha) + self_w * alpha
    cx, cy, cz = torch.cos(self_w).unbind(-1)
    sx, sy, sz = torch.sin(self_w).unbind(-1)
    up_vec = torch.stack([
        cx*cz*sy+sx*sz,
        -cz*sx+cx*sy*sz,
        cx*cy], -1)

    c = action[:, 3:] + 1
    _a = up_vec * c * thrust + g - drag * self_v * torch.norm(self_v, -1)

    self_v = self_v + _a * ctl_dt
    self_p = self_p + self_v * ctl_dt
    return self_p, self_v, self_w


class QuadState:
    def __init__(self, batch_size, device, grad_decay=0.8) -> None:
        self.p = torch.zeros((batch_size, 3), device=device)
        self.a = torch.zeros((batch_size, 3), device=device) \
            * torch.tensor([0.1, 0.1, 0.1], device=device)
        self.v = torch.zeros((batch_size, 3), device=device)
        self.g = torch.randn((batch_size, 3), device=device) * 0.1
        self.drag = torch.rand((batch_size, 1), device=device) * 0.09 + 0.01
        self.g[:, 2] -= 9.80665
        self.rate_ctl_delay = 0.075 + 0.05 * torch.rand((batch_size, 1), device=device)
        self.forward_vec = torch.zeros((batch_size, 3), device=device)
        self.forward_vec[:, 0] = 1
        self.update_state_vec()

    # @torch.jit.script
    def run(self, action, ctl_dt=1/15):
        target_v = action / ctl_dt
        target_a = (target_v - self.v) / ctl_dt

        self.a = target_a
        self.v = target_v
        alpha = 0.4 ** ctl_dt
        self.p = self.p * alpha + self.p.detach() * (1 - alpha) + action
        self.update_state_vec()

    @torch.no_grad()
    def update_state_vec(self):
        a_thr = self.a - self.g + self.drag * self.v * torch.norm(self.v, 2, -1, True)
        thrust = torch.norm(a_thr, 2, -1, True)
        self.up_vec = a_thr / thrust
        forward_vec = self.forward_vec + self.v - torch.sum(self.v * self.up_vec, -1, keepdim=True) * self.up_vec
        forward_vec /= torch.norm(forward_vec, 2, -1, True)
        self.forward_vec = forward_vec
        self.left_vec = torch.cross(self.up_vec, self.forward_vec)


class Env:
    def __init__(self, batch_size, width, height, device='cpu') -> None:
        self.device = device
        self.batch_size = batch_size
        n = int(math.sqrt(batch_size))
        assert n * n == batch_size
        self.r = EnvRenderer(n, n, width, height)
        self.reset()

    def reset(self):
        self.quad = QuadState(self.batch_size, self.device)
        self.r.set_obstacles()

    @torch.no_grad()
    def render(self, ctl_dt):
        state = torch.cat([self.quad.p, self.quad.forward_vec, self.quad.up_vec], -1).cpu()
        return self.r.render(state.numpy(), ctl_dt)

    def step(self, action, ctl_dt=1/15):
        self.quad.run(action, ctl_dt)


@torch.no_grad()
def main():
    env = Env(1, 80, 60, 'cpu')
    color, depth, _ = env.render(1/15)
    plt.imsave('1.png', depth[0])

if __name__ == '__main__':
    main()


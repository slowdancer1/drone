from time import time
from matplotlib import pyplot as plt
import numpy as np
import torch
import quadsim


class EnvRenderer(quadsim.Env):
    def render(self, cameras):
        z_near = 0.01
        z_far = 10.0
        color, depth = super().render(cameras)
        color = np.flip(color, 1)
        depth = np.flip(2 * depth - 1, 1)
        depth = (2.0 * z_near * z_far) / (z_far + z_near - depth * (z_far - z_near))
        return color, depth


def axis_angle_to_quaternion(axis_angle: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as axis/angle to quaternions.
    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.
    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half_angles = angles * 0.5
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    quaternions = torch.cat(
        [torch.cos(half_angles), axis_angle * sin_half_angles_over_angles], dim=-1
    )
    return quaternions


def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.
    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).
    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def quaternion_to_up(quaternions):
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    return torch.stack((
        two_s * (i * k + j * r),
        two_s * (j * k - i * r),
        1 - two_s * (i * i + j * j)), -1)


def quaternion_to_forward(quaternions):
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    return torch.stack((
        1 - two_s * (j * j + k * k),
        two_s * (i * j + k * r),
        two_s * (i * k - j * r)), -1)


@torch.jit.script
def run(self_p, self_v, self_q, self_w, g, thrust, action, ctl_dt:float=1/15, rate_ctl_delay:float=0.1):
    alpha = 0.6 ** ctl_dt
    self_p = alpha * self_p + (1 - alpha) * self_p.detach()
    self_v = alpha * self_v + (1 - alpha) * self_v.detach()
    self_q = alpha * self_q + (1 - alpha) * self_q.detach()
    self_w = alpha * self_w + (1 - alpha) * self_w.detach()

    w = action[:, :3]
    c = action[:, 3:] + 1

    alpha = rate_ctl_delay ** (ctl_dt / rate_ctl_delay)

    self_w = w * (1 - alpha) + self_w * alpha
    self_q = axis_angle_to_quaternion(self_w)
    up_vec = quaternion_to_up(self_q)
    _a = up_vec * c * thrust + g - 0.1 * self_v * torch.norm(self_v, -1)

    self_v = self_v + _a * ctl_dt
    self_p = self_p + self_v * ctl_dt
    return self_p, self_v, self_q, self_w


batch_size = 16


class QuadState:
    def __init__(self, device) -> None:
        self.p = torch.zeros((batch_size, 3), device=device)
        self.q = torch.zeros((batch_size, 4), device=device)
        self.q[:, 0] = 1
        self.v = torch.randn((batch_size, 3), device=device) * 0.1
        self.v[:, 0] += 1
        self.w = torch.zeros((batch_size, 3), device=device)
        self.a = torch.zeros((batch_size, 3), device=device)
        self.g = torch.randn((batch_size, 3), device=device) * 0.1
        self.g[:, 2] -= 9.80665
        self.thrust = torch.randn((batch_size, 1), device=device) + 9.80665

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


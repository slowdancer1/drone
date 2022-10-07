from time import time
from matplotlib import pyplot as plt
import numpy as np
import pyrender
import torch
from scipy.spatial.transform import Rotation
import trimesh

from pytorch3d.transforms.rotation_conversions import (
    quaternion_raw_multiply,
    quaternion_to_matrix,
    axis_angle_to_quaternion,
)


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


camera_pose = np.array([
   [0.0,  0,  -1,   0.0],
   [-1,  0.0, 0.0, 0.0],
   [0,   1,   0,   0.0],
   [0.0,  0.0, 0.0, 1.0],
])


@torch.jit.script
def run(self_p, self_v, self_q, self_w, g, w, c, ctl_dt=1/15, rate_ctl_delay=0.1):
    alpha = rate_ctl_delay ** (ctl_dt / rate_ctl_delay)

    half_alpha = rate_ctl_delay ** (ctl_dt / 2 / rate_ctl_delay)
    half_w = w * (1 - half_alpha) + self_w * half_alpha

    half_q = axis_angle_to_quaternion(half_w)
    half_up_vec = quaternion_to_up(half_q)
    half_a = half_up_vec * c * 9.80665 + g - 0.1 * self_v * torch.norm(self_v)
    half_v = self_v + half_a * ctl_dt / 2

    self_p = self_p + half_v * ctl_dt
    self_v = self_v + half_a * ctl_dt
    self_w = w * (1 - alpha) + self_w * alpha
    self_q = axis_angle_to_quaternion(self_w)
    return self_p, self_v, self_q, self_w


def scale_grad(alpha):
    def forward(x):
        return x * alpha + (1 - alpha) * x.detach()
    return forward


class QuadState:
    def __init__(self, device) -> None:
        self.p = torch.zeros(3, device=device)
        self.q = torch.zeros(4, device=device)
        self.q[0] = 1
        self.v = torch.randn(3, device=device) * 0.1
        self.w = torch.zeros(3, device=device)
        self.a = torch.zeros(3, device=device)
        self.g = torch.tensor([0, 0, -9.80665], device=device)

        self.rate_ctl_delay = 0.2

    def run(self, w, c, ctl_dt=1/15):
        # alpha = 0.9 ** ctl_dt
        # _p, _v, _q, _w = map(scale_grad(alpha), (self.p, self.v, self.q, self.w))
        _p, _v, _q, _w = self.p, self.v, self.q, self.w
        self.p, self.v, self.q, self.w = run(
            _p, _v, _q, _w, self.g, w, c, ctl_dt, self.rate_ctl_delay)

    def stat(self):
        print("p:", self.p.tolist())
        print("v:", self.v.tolist())
        print("q:", self.q.tolist())
        print("w:", self.w.tolist())


class Env:
    def __init__(self, device) -> None:
        self.scene = pyrender.Scene(ambient_light=(1, 1, 1))
        self.r = pyrender.OffscreenRenderer(160, 90)

        camera = pyrender.PerspectiveCamera(yfov=np.pi * 0.35)
        camera_node = pyrender.Node(camera=camera, matrix=camera_pose)
        self.quad_node = pyrender.Node(children=[camera_node])
        self.scene.add_node(self.quad_node)
        self.scene.add(pyrender.DirectionalLight((255, 255, 255), 1))

        fuze_trimesh = trimesh.load('models/ball.obj')
        mesh = pyrender.Mesh.from_trimesh(fuze_trimesh)
        self.obstacles_nodes = [self.scene.add(mesh) for _ in range(25)]

        ground = trimesh.Trimesh([[-10, -10, -1], [10, -10, -1], [10, 10, -1], [-10, 10, -1]], faces=[[0, 1, 2, 3]])
        mesh = pyrender.Mesh.from_trimesh(ground)
        self.scene.add(mesh)
        self.device = device
        self.reset()

    def reset(self):
        self.quad = QuadState(self.device)
        self.obstacles = torch.stack([
            torch.rand(25, device=self.device) * 12 + 2,
            torch.rand(25, device=self.device) * 10 - 5,
            torch.rand(25, device=self.device) * 3 - 1
        ], 1)
        for i, (x, y, z) in enumerate(self.obstacles.tolist()):
            pose = np.array([
                [1.0, 0.0, 0.0, x],
                [0.0, 1.0, 0.0, y],
                [0.0, 0.0, 1.0, z],
                [0.0, 0.0, 0.0, 1.0],
            ])
            self.scene.set_pose(self.obstacles_nodes[i], pose)

    @torch.no_grad()
    def render(self):
        rot_mat = quaternion_to_matrix(self.quad.q).cpu().numpy()
        quad_mat = np.hstack([rot_mat, self.quad.p.cpu().numpy()[:, None]])
        quad_mat = np.vstack([quad_mat, [0, 0, 0, 1]])
        self.scene.set_pose(self.quad_node, quad_mat)
        # depth = self.r.render(self.scene, flags=pyrender.RenderFlags.DEPTH_ONLY)
        color, depth = self.r.render(self.scene)
        return color, depth
        plt.imshow(1 / depth)
        plt.show()

    def step(self, action, ctl_dt=1/15):
        w = quaternion_to_matrix(self.quad.q) @ action[:3]
        self.quad.run(w, action[3] + 1, ctl_dt)


@torch.no_grad()
def main():
    env = Env('cpu')
    color, depth = env.render()
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


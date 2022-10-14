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


def quaternion_to_axis_angle(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to axis/angle.
    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).
    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    """
    norms = torch.norm(quaternions[..., 1:], p=2, dim=-1, keepdim=True)
    half_angles = torch.atan2(norms, quaternions[..., :1])
    angles = 2 * half_angles
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
    return quaternions[..., 1:] / sin_half_angles_over_angles


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
    w = quaternion_to_axis_angle(action[:, :4])
    c = action[:, 4:] + 1

    alpha = rate_ctl_delay ** (ctl_dt / rate_ctl_delay)

    self_w = w * (1 - alpha) + self_w * alpha
    self_q = axis_angle_to_quaternion(self_w)
    up_vec = quaternion_to_up(self_q)
    _a = up_vec * c * thrust + g - 0.1 * self_v * torch.norm(self_v, -1)

    self_v = self_v + _a * ctl_dt
    self_p = self_p + self_v * ctl_dt
    return self_p, self_v, self_q, self_w


batch_size = 1


class QuadState:
    def __init__(self, device) -> None:
        self.p = torch.zeros((batch_size, 3), device=device)
        self.q = torch.zeros((batch_size, 4), device=device)
        self.q[:, 0] = 1
        self.v = torch.randn((batch_size, 3), device=device) * 0.01
        self.w = torch.zeros((batch_size, 3), device=device)
        self.a = torch.zeros((batch_size, 3), device=device)
        self.g = torch.zeros((batch_size, 3), device=device)
        self.g[:, 2] -= 9.80665
        self.thrust = torch.randn((batch_size, 1), device=device) * 0. + 9.80665

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
    import rospy
    from nav_msgs.msg import Odometry
    from geometry_msgs.msg import Pose, PoseWithCovariance, Point, Quaternion, Vector3, TwistWithCovariance, Twist
    from sensor_msgs.msg import Imu, Image
    from mavros_msgs.msg import AttitudeTarget
    import cv_bridge
    rospy.init_node('env_ros', anonymous=True)
    depth_pub = rospy.Publisher('/iris_0/realsense/depth_camera/depth/image_raw', Image, queue_size=10)
    odom_pub = rospy.Publisher('/vins_fusion/imu_propagate', Odometry, queue_size=10)
    odom2_pub = rospy.Publisher('/vins_fusion/odometry', Odometry, queue_size=10)
    imu_pub = rospy.Publisher('/iris_0/mavros/imu/data', Imu, queue_size=10)
    q_target = [1., 0, 0, 0, 0]
    def set_attitude_target(data):
        nonlocal q_target
        x, y, z, w = data.orientation.x, data.orientation.y, data.orientation.z, data.orientation.w
        q_target = w, x, y, z, data.thrust / 0.3 - 1
    rospy.Subscriber('/iris_0/mavros/setpoint_raw/attitude', AttitudeTarget, set_attitude_target, queue_size=10)
    rate = rospy.Rate(120)
    env = Env('cpu')
    bridge = cv_bridge.CvBridge()
    step_id = 0
    while not rospy.is_shutdown():
        step_id += 1
        stamp = rospy.Time.now()
        env.step(torch.tensor(q_target).reshape(1, 5), 1 / 100)
        x, y, z = env.quad.p[0].tolist()
        position = Point(x, y, z + 1)
        w, x, y, z = env.quad.q[0].tolist()
        orientation = Quaternion(x, y, z, w)
        odom_msg = Odometry(
            pose=PoseWithCovariance(pose=Pose(
                position=position,
                orientation=orientation)),
            twist=TwistWithCovariance(twist=Twist(
                linear=Vector3(*env.quad.v[0].tolist()))))
        odom_msg.header.frame_id = 'world'
        odom_msg.header.stamp = stamp
        imu_pub.publish(orientation=orientation)
        if step_id % 4 == 0:
            color, depth = env.render()
            depth = np.ascontiguousarray(depth[0])
            depth_image = bridge.cv2_to_imgmsg(depth, "32FC1")
            depth_image.header.stamp = stamp
            depth_pub.publish(depth_image)
            odom2_pub.publish(odom_msg)
        odom_pub.publish(odom_msg)
        rate.sleep()


if __name__ == '__main__':
    main()


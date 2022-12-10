from random import randint
from matplotlib import pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm


from rotation import _axis_angle_rotation


ctl_dt = 1 / 15


@torch.no_grad()
def eval_once(env, model, batch_size, device):
    states_mean = [3.62, 0, 0, 0, 0, 4.14, 0, 0, 0.125]
    states_mean = torch.tensor([states_mean], device=device)
    states_std = [2.770, 0.367, 0.343, 0.080, 0.240, 4.313, 0.396, 0.327, 0.073]
    states_std = torch.tensor([states_std], device=device)

    env.reset()
    p_history = []
    nearest_pt_history = []
    h = None
    p_target = torch.stack([
        torch.rand((batch_size,), device=device) * 54 + 10,
        torch.rand((batch_size,), device=device) * 12 - 6,
        torch.full((batch_size,), 0, device=device)
    ], -1)

    margin = torch.rand((batch_size,), device=device) * 0.25
    max_speed = torch.rand((batch_size, 1), device=device) * 9 + 3

    act_buffer = []
    for _ in range(randint(1, 3)):
        act = torch.zeros((batch_size, 4), device=device)
        act_buffer.append(act)
    speed_ratios = []
    for t in range(150):
        color, depth, nearest_pt = env.render(ctl_dt)
        p_history.append(env.quad.p)
        nearest_pt_history.append(nearest_pt.copy())

        depth = torch.as_tensor(depth[:, None], device=device)
        target_v = p_target - env.quad.p
        R = _axis_angle_rotation('Z',  env.quad.w[:, -1])
        target_v_norm = torch.norm(target_v, 2, -1, keepdim=True)
        target_v_unit = target_v / max_speed
        target_v = target_v_unit * target_v_norm.clamp_max(max_speed)
        local_v = torch.squeeze(env.quad.v[:, None] @ R, 1)
        local_v.add_(torch.randn_like(local_v) * 0.01)
        local_v_target = torch.squeeze(target_v[:, None] @ R, 1)
        state = torch.cat([
            local_v,
            env.quad.w[:, :2],
            local_v_target,
            margin[:, None]
        ], -1)

        # normalize
        x = 3 / depth.clamp_(0.01, 10) - 0.6
        x = F.max_pool2d(x, 5, 5)
        # states.append(state)
        state = (state - states_mean) / states_std
        # depths.append(depth.clamp_(0.01, 10))
        act, h = model(x, state, h)
        act = act.clone()
        act[:, 2] += env.quad.w[:, 2]

        act_buffer.append(act)
        env.step(act_buffer.pop(0), ctl_dt)

        # loss
        v_forward = torch.sum(target_v_unit * env.quad.v, -1, True)
        speed_ratio = v_forward.div(target_v_norm).clamp(0, 1)
        speed_ratio *= torch.cosine_similarity(target_v, env.quad.v)[:, None]
        speed_ratios.append(speed_ratio)

    p_history = torch.stack(p_history)
    nearest_pt_history = torch.as_tensor(np.stack(nearest_pt_history), device=device)

    distance = torch.norm(p_history - nearest_pt_history, 2, -1) - margin

    success = torch.all(distance > 0, 0)
    speed = torch.cat(speed_ratios, -1).max(-1).values
    return speed, success


def eval(env, model, batch_size, device, num_iters=256):
    res = [eval_once(env, model, batch_size, device) for _ in tqdm(range(num_iters))]
    speed, success = map(torch.cat, zip(*res))
    return torch.mean(speed * success).item()
    idx = speed.argsort()
    speed, success = speed[idx], success[idx]
    success_r = success.cumsum(0) / torch.arange(success.size(0), device=device).add(1)
    fig, ax = plt.subplots()
    ax.plot(speed.tolist(), success_r.tolist())
    plt.show()
    speed[1:] -= speed[:-1].clone()
    return torch.sum(speed * success_r).item()


if __name__ == '__main__':
    import argparse
    from env_gl import Env
    from model import Model

    parser = argparse.ArgumentParser()
    parser.add_argument('--resume')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_iters', type=int, default=64)
    args = parser.parse_args()
    print(args)

    device = torch.device('cuda')

    env = Env(args.batch_size, 80, 60, device)
    # env.quad.grad_decay = 0.7
    model = Model()
    model = model.to(device)

    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location=device))

    avg_s = eval(env, model, args.batch_size, device, args.num_iters)
    print(avg_s)

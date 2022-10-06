import torch

v_vec = torch.randn(3, requires_grad=True)
v_forward = torch.tensor([1., 0, 0], requires_grad=True)
print(v_vec, v_forward)

dot = v_vec @ v_forward

v_norm = torch.norm(v_vec)

breakpoint()

torch.autograd.grad(dot / v_norm, (v_vec, v_forward), retain_graph=True)

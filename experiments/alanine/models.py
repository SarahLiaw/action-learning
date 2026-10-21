"""The summed torus path action shared by LEAP, binary NRE, velocity regression and the untrained control."""
from __future__ import annotations

import torch
from torch import nn

from .reference import wrapped_log_prob

ACTIONS = ('untrained', 'velocity', 'leap', 'binary_nre')


def mlp(ins, width, outs, layers):
    seq = [nn.Linear(ins, width), nn.SiLU()]
    for _ in range(layers-1):
        seq.extend([nn.Linear(width, width), nn.SiLU()])
    return nn.Sequential(*seq, nn.Linear(width, outs))


class PathAction(nn.Module):
    """S = sum_k m_k [-sum_j log q_wrap(x_{k+1,j}; x_k + dt f(x_k), 2 D dt) + dt r(x_k)].

    Features are (cos x, sin x, D, T/600); drift f has 3x128 hidden SiLU layers, residual r 2x64.
    """

    def __init__(self, diffusion, dt, temperature, residual=True):
        super().__init__()
        self.diffusion, self.dt, self.temperature = float(diffusion), float(dt), float(temperature)
        self.use_residual = residual
        self.drift_net = mlp(6, 128, 2, 3)
        self.residual_net = mlp(6, 64, 1, 2)

    def features(self, x):
        c = x.new_tensor([self.diffusion, self.temperature/600]).expand(*x.shape[:-1], 2)
        return torch.cat([x.cos(), x.sin(), c], -1)

    def drift(self, x):
        return self.drift_net(self.features(x))

    def residual(self, x):
        if self.use_residual:
            return self.residual_net(self.features(x)).squeeze(-1)
        return torch.zeros_like(x[..., 0])

    def forward(self, paths, masks):
        active = masks.bool()
        x = paths[:, :-1][active]
        logp = wrapped_log_prob(paths[:, 1:][active], x+self.dt*self.drift(x), 2*self.diffusion*self.dt).sum(-1)
        cost = (-logp+self.dt*self.residual(x))*masks[active]
        row = torch.arange(paths.shape[0], device=paths.device)[:, None].expand_as(masks)[active]
        return paths.new_zeros(paths.shape[0]).scatter_add(0, row, cost)


def offset_net():
    """Binary-NRE offset from the periodic initial-state embedding (it absorbs the success normalizer)."""
    return nn.Sequential(nn.Linear(4, 32), nn.SiLU(), nn.Linear(32, 1))


def load_action(path, diffusion, dt, temperature, method, device='cpu'):
    """Frozen action; velocity regression has no residual."""
    model = PathAction(diffusion, dt, temperature, residual=method != 'velocity').to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return model.eval().requires_grad_(False)


def action(model, paths, masks, ref):
    """Learned action, or minus the reference log density when model is None."""
    return -ref.log_prob(paths, masks) if model is None else model(paths, masks)

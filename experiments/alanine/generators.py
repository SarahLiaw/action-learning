"""Conditional DDPM (100-step cosine schedule) and flow matching (100 Euler steps) on one temporal network."""
from __future__ import annotations

import math

import torch
from torch import nn

from .data import angles, embed, interpolation, wrap

GENERATORS = ('ddpm', 'flow_matching')


class TemporalNet(nn.Module):
    """Width-96 dilated residual convolutions over frames; input = noisy embedding, conditioning, time."""

    def __init__(self, width=96, dilations=(1, 2, 4, 8, 16, 1, 2)):
        super().__init__()
        self.input = nn.Conv1d(16, width, 1)
        self.blocks = nn.ModuleList([nn.Conv1d(width, width, 3, padding=d, dilation=d) for d in dilations])
        self.output = nn.Conv1d(width, 4, 1)

    def forward(self, z, t, cond):
        time = torch.stack([t, (2*math.pi*t).sin(), (2*math.pi*t).cos()], -1)[:, None].expand(-1, z.shape[1], -1)
        h = self.input(torch.cat([z, cond, time], -1).transpose(1, 2))
        for conv in self.blocks:
            h = h+torch.nn.functional.silu(conv(h))/math.sqrt(2)
        return self.output(torch.nn.functional.silu(h)).transpose(1, 2)


def cosine_alphas(device, steps=100):
    """Cumulative alpha-bar of the cosine schedule with betas clipped at 0.999."""
    t = torch.linspace(0, 1, steps+1, device=device)
    a = torch.cos((t+.008)/1.008*math.pi/2).square()
    a = a/a[0]
    beta = (1-a[1:]/a[:-1]).clamp(max=.999)
    return torch.cumprod(1-beta, 0)


def conditioning(paths, masks, obs, ref):
    """Observed embeddings and mask, active-frame mask, time, and b_hat along the observation interpolant."""
    lin = interpolation(paths, obs)
    physics = ref.physics_drift(lin.reshape(-1, 2)).reshape_as(lin).clamp(-20, 20)/20
    knots = torch.cat([torch.ones_like(masks[:, :1]), masks], 1)
    t = torch.arange(paths.shape[1], device=paths.device, dtype=paths.dtype)[None].expand(len(paths), -1)/(paths.shape[1]-1)
    return torch.cat([embed(paths)*obs[..., None], obs[..., None], knots[..., None], t[..., None], physics], -1)


def generator_loss(model, method, target, cond, free, abar):
    """Noise (DDPM) or velocity (flow matching) regression on unobserved frames; observed frames stay clean."""
    eps = torch.randn_like(target)
    if method == 'ddpm':
        ti = torch.randint(len(abar), (len(target),), device=target.device)
        a = abar[ti, None, None]
        t = (ti.float()+1)/len(abar)
        z, goal = a.sqrt()*target+(1-a).sqrt()*eps, eps
    else:
        t = torch.rand(len(target), device=target.device)
        z, goal = (1-t[:, None, None])*eps+t[:, None, None]*target, target-eps
    z = torch.where(cond[..., 4].bool()[..., None], target, z)
    return ((model(z, t, cond)-goal).square().mean(-1)*free).sum()/free.sum().clamp_min(1)


@torch.no_grad()
def generate(model, method, observed, obs, masks, ref, steps=100):
    """One conditional draw per row, observed frames clamped throughout; returns wrapped angle paths."""
    cond, target = conditioning(observed, masks, obs, ref), embed(observed)
    clamp = lambda z: torch.where(obs[..., None], target, z)
    z = clamp(torch.randn_like(target))
    if method == 'flow_matching':
        for i in range(steps):
            z = clamp(z+model(z, torch.full((len(z),), i/steps, device=z.device), cond)/steps)
    else:
        abar = cosine_alphas(z.device, steps)
        for i in reversed(range(steps)):
            a = abar[i]
            prev = abar[i-1] if i else z.new_tensor(1.)
            alpha = a/prev
            beta = 1-alpha
            eps = model(z, torch.full((len(z),), (i+1)/steps, device=z.device), cond)
            mean = (z-beta/(1-a).sqrt()*eps)/alpha.sqrt()
            z = clamp(mean+((beta*(1-prev)/(1-a)).sqrt()*torch.randn_like(z) if i else 0))
    return torch.where(obs[..., None], observed, wrap(angles(z)))


def load_generator(path, device='cpu'):
    model = TemporalNet().to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return model.eval().requires_grad_(False)

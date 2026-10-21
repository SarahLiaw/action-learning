"""Double-well data: Euler-Maruyama paths, hidden integration substeps, observation noise."""
from __future__ import annotations

import math

import torch

SPLIT_SEEDS = {'train': 10_003, 'val': 20_011, 'test': 30_013}   # offsets from the run seed


def drift(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """scale * b(x) with b(x) = x - x^3, the drift of U(x) = (x^2-1)^2/4."""
    return scale*(x-x.pow(3))


def step_log_prob(x, x_next, dt: float, D: float, b) -> torch.Tensor:
    """Euler transition density log N(x_next; x + b(x) dt, 2 D dt)."""
    variance = 2.0*D*dt
    innovation = x_next-x-b(x)*dt
    return -0.5*(innovation.square()/variance+math.log(2.0*math.pi*variance))


def log_prob(paths: torch.Tensor, cfg) -> torch.Tensor:
    """Exact log p(x_1..x_N | x_0) of the data chain; the observed law only when cfg.exact."""
    return step_log_prob(paths[..., :-1], paths[..., 1:], cfg.dt, cfg.D_data, drift).sum(-1)


def simulate(n: int, cfg, seed: int) -> torch.Tensor:
    """[n, N+1] observed paths: Euler at dt/substeps, every substeps-th state, noise after x_0."""
    generator = torch.Generator().manual_seed(seed)
    fine_dt, fine_steps = cfg.dt/cfg.substeps, cfg.n_steps*cfg.substeps
    x = torch.empty(n, fine_steps+1, dtype=torch.float32)
    x[:, 0] = cfg.x0_mean+cfg.x0_std*torch.randn(n, generator=generator, dtype=torch.float32)
    scale = math.sqrt(2.0*cfg.D_data*fine_dt)
    for k in range(fine_steps):
        x[:, k+1] = x[:, k]+drift(x[:, k])*fine_dt+scale*torch.randn(n, generator=generator, dtype=torch.float32)
    observed = x[:, ::cfg.substeps].clone()
    if cfg.obs_noise > 0.0:
        noise = cfg.obs_noise*torch.randn(observed.shape, generator=generator, dtype=torch.float32)
        noise[:, 0] = 0.0   # x_0 is the conditioning context, observed exactly
        observed = observed+noise
    return observed


def splits(cfg) -> dict[str, torch.Tensor]:
    """Independently simulated whole-trajectory train/val/test sets."""
    return {name: simulate(getattr(cfg, f'n_{name}'), cfg, cfg.seed+offset)
            for name, offset in SPLIT_SEEDS.items()}

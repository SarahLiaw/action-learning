"""Euler-Maruyama path densities and the structured LEAP correction."""
from __future__ import annotations

import math

import torch


def em_log_prob(paths: torch.Tensor, drift, dt: float, diffusion: float) -> torch.Tensor:
    """log p(x_1..x_N | x_0) for x_{k+1} = x_k + b dt + sqrt(2 D dt) xi.  paths: [..., N+1, d] -> [...]."""
    x = paths[..., :-1, :]
    resid = paths[..., 1:, :]-x-drift(x)*dt
    n, d = paths.shape[-2]-1, paths.shape[-1]
    return -(resid**2).sum((-2, -1))/(4*diffusion*dt)-0.5*n*d*math.log(4*math.pi*diffusion*dt)


def structured_correction(paths: torch.Tensor, correction_drift, reference_drift, dt: float,
                          diffusion: float) -> torch.Tensor:
    """G = sum_k [u.(v - b_ref)/(2D) - |u|^2/(4D)] dt, the exact log-ratio for a drift shift u (Eq. 9).

    paths: [..., N+1, d] -> [...]; reference_drift may be None for a zero reference drift.
    """
    x = paths[..., :-1, :]
    v = (paths[..., 1:, :]-x)/dt
    u = correction_drift(x)
    w = v-(reference_drift(x) if reference_drift is not None else 0.0)
    return (((u*w).sum(-1)/(2*diffusion)-(u**2).sum(-1)/(4*diffusion))*dt).sum(-1)

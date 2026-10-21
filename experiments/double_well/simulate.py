"""Euler-Maruyama paths of dX = b(X) dt + sqrt(2D) dW selected by their terminal value."""
from __future__ import annotations

import numpy as np
import torch


def drift(x):
    """b(x) = -U'(x) = x - x^3 for U(x) = (x^2-1)^2/4."""
    return x-x**3


def simulate_selected(n_paths: int, *, diffusion: float, dt: float, horizon: float, start: float,
                      target: float, tolerance: float, round_size: int, device,
                      max_rounds: int = 5000) -> torch.Tensor:
    """Rejection-sample paths from `start` with |x_T - target| < tolerance.  Returns [n_paths, N+1, 1]."""
    n_steps = int(horizon/dt)
    kept, n_kept = [], 0
    for _ in range(max_rounds):
        x = torch.zeros(round_size, n_steps+1, 1, device=device)
        x[:, 0] = start
        for k in range(n_steps):
            noise = torch.randn(round_size, 1, device=device)*np.sqrt(2*diffusion*dt)
            x[:, k+1] = x[:, k]+drift(x[:, k])*dt+noise
        kept.append(x[((x[:, -1]-target).abs() < tolerance).squeeze(-1)][:n_paths-n_kept])
        n_kept += len(kept[-1])
        if n_kept == n_paths:
            return torch.cat(kept)
    raise RuntimeError(f'only {n_kept} of {n_paths} paths reached the terminal window')

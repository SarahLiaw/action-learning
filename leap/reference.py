"""Gaussian Euler-Maruyama reference process (Eq. 8): sample paths and evaluate their density."""
from __future__ import annotations

import math

import torch

from .euler import em_log_prob


class EulerReference:
    """x_{k+1} = x_k + b_ref(x_k) dt + sqrt(2 D dt) xi from given starts; drift=None gives Brownian motion."""

    def __init__(self, diffusion: float, dt: float, drift=None):
        self.diffusion, self.dt = diffusion, dt
        self.drift = drift if drift is not None else torch.zeros_like

    @torch.no_grad()
    def sample(self, x0: torch.Tensor, n_steps: int, k: int = 1, generator: torch.Generator | None = None) -> torch.Tensor:
        """k paths from each start: x0 [B, d] -> [B, k, n_steps+1, d]."""
        x = x0.unsqueeze(1).expand(-1, k, -1)
        steps, scale = [x], math.sqrt(2*self.diffusion*self.dt)
        for _ in range(n_steps):
            noise = torch.randn(x.shape, generator=generator, device=x.device, dtype=x.dtype)
            x = x+self.drift(x)*self.dt+scale*noise
            steps.append(x)
        return torch.stack(steps, dim=-2)

    def log_prob(self, paths: torch.Tensor) -> torch.Tensor:
        """log p_ref(x_1..x_N | x_0) for paths [..., N+1, d]."""
        return em_log_prob(paths, self.drift, self.dt, self.diffusion)

"""The Euler reference process on the observed grid: samplers and exact densities."""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from . import dynamics


@dataclass(frozen=True)
class Reference:
    """Euler chain with drift a * b(x) and diffusion D, started at the observed x_0."""
    dt: float
    n_steps: int
    D: float
    a: float

    @classmethod
    def of(cls, cfg) -> 'Reference':
        return cls(cfg.dt, cfg.n_steps, cfg.D_ref, cfg.a_ref)

    def drift(self, x: torch.Tensor) -> torch.Tensor:
        return dynamics.drift(x, self.a)

    def sample(self, x0: torch.Tensor, K: int, generator: torch.Generator) -> torch.Tensor:
        """[B, K, N+1] reference paths from each of the B initial states."""
        paths = torch.empty(len(x0), K, self.n_steps+1, dtype=x0.dtype)
        paths[:, :, 0] = x0[:, None]
        scale = math.sqrt(2.0*self.D*self.dt)
        for k in range(self.n_steps):
            x = paths[:, :, k]
            paths[:, :, k+1] = x+self.drift(x)*self.dt+scale*torch.randn(x.shape, generator=generator, dtype=x.dtype)
        return paths

    def sample_next(self, x: torch.Tensor, K: int, generator: torch.Generator) -> torch.Tensor:
        """[..., K] reference next states from every current state in x."""
        noise = torch.randn(*x.shape, K, generator=generator, dtype=x.dtype)
        return x.unsqueeze(-1)+self.drift(x).unsqueeze(-1)*self.dt+math.sqrt(2.0*self.D*self.dt)*noise

    def step_log_prob(self, x: torch.Tensor, x_next: torch.Tensor) -> torch.Tensor:
        return dynamics.step_log_prob(x, x_next, self.dt, self.D, self.drift)

    def log_prob(self, paths: torch.Tensor) -> torch.Tensor:
        """log p_ref(x_1..x_N | x_0)."""
        return self.step_log_prob(paths[..., :-1], paths[..., 1:]).sum(-1)

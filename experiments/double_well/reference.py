"""Brownian reference conditioned on the same terminal window as the data."""
from __future__ import annotations

import numpy as np
import torch


class TerminalWindowReference:
    """Brownian motion from `start` with diffusion D, conditioned on x_T in [target -+ tolerance]."""

    def __init__(self, *, diffusion: float, dt: float, horizon: float, start: float, target: float,
                 tolerance: float, device):
        self.diffusion, self.dt, self.start = diffusion, dt, start
        self.low, self.high = target-tolerance, target+tolerance
        self.n_steps = int(horizon/dt)
        self.device = device
        f64 = dict(device=device, dtype=torch.float64)
        self.terminal = torch.distributions.Normal(torch.tensor(float(start), **f64),
                                                   torch.tensor(np.sqrt(2*diffusion*horizon), **f64))
        # window edges enter the terminal CDF in float32, as in the paper run
        self.cdf_low = self.terminal.cdf(torch.tensor(self.low, device=device))
        self.cdf_high = self.terminal.cdf(torch.tensor(self.high, device=device))

    def sample(self, n: int) -> torch.Tensor:
        """Truncated-normal endpoint, then the exact discrete Gaussian bridge to it.  [n, N+1, 1]."""
        u = self.cdf_low+(self.cdf_high-self.cdf_low)*torch.rand(n, device=self.device, dtype=torch.float64)
        end = self.terminal.icdf(u).float().view(-1, 1, 1)
        steps = torch.randn(n, self.n_steps, 1, device=self.device)*np.sqrt(2*self.diffusion*self.dt)
        bm = torch.cat([torch.zeros_like(steps[:, :1]), steps.cumsum(1)], 1)
        frac = torch.arange(self.n_steps+1, device=self.device).view(1, -1, 1)/self.n_steps
        bridge = self.start+frac*(end-self.start)+bm-frac*bm[:, -1:]
        bridge[:, 0] = self.start
        bridge[:, -1] = end[:, 0]
        return bridge

    def log_prob(self, paths: torch.Tensor) -> torch.Tensor:
        """Path log density with Gaussian and window normalizers; -inf off the start or window."""
        dx = paths[:, 1:]-paths[:, :-1]
        var = 2*self.diffusion*self.dt
        logp = (-0.5*(dx.square()/var+np.log(2*np.pi*var))).sum((1, 2))
        end = paths[:, -1, 0]
        inside = (end >= self.low) & (end <= self.high) & ((paths[:, 0, 0]-self.start).abs() < 1e-6)
        return torch.where(inside, logp-(self.cdf_high-self.cdf_low).log().to(paths.dtype), -torch.inf)

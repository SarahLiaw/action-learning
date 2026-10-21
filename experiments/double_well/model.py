"""Conservative LEAP model: an MLP potential U_theta with drift b_theta = -dU_theta/dx."""
from __future__ import annotations

import torch
import torch.nn as nn


class ConservativePotential(nn.Module):
    """U_theta: Linear-LayerNorm-GELU hidden layers, scalar output; zero residual."""

    def __init__(self, diffusion: float, dt: float, hidden_dim: int = 512, hidden_layers: int = 3):
        super().__init__()
        self.diffusion, self.dt = diffusion, dt
        layers = [nn.Linear(1, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()]
        for _ in range(hidden_layers-1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()]
        self.U = nn.Sequential(*layers, nn.Linear(hidden_dim, 1))

    def drift(self, x: torch.Tensor) -> torch.Tensor:
        """-dU/dx; keeps its graph in training and when x itself requires grad (path inference)."""
        keep_graph = self.training or (torch.is_grad_enabled() and x.requires_grad)
        with torch.enable_grad():
            xg = x if x.requires_grad else x.detach().requires_grad_(True)
            (g,) = torch.autograd.grad(self.U(xg).sum(), xg, create_graph=keep_graph)
        return -g

    def energy(self, paths: torch.Tensor) -> torch.Tensor:
        """Training energy E = sum_k |v_k - b(x_k)|^2/(4D) dt of paths [B, N+1, 1]."""
        x = paths[:, :-1]
        v = (paths[:, 1:]-x)/self.dt
        b, n, d = x.shape
        vals = (0.25/self.diffusion)*(v.reshape(-1, d)-self.drift(x.reshape(-1, d)))**2
        return torch.sum(vals.reshape(b, n), dim=1)*self.dt

    def action(self, paths: torch.Tensor) -> torch.Tensor:
        """Inference action dt/(4D) sum_k |v_k - b(x_k)|^2; equals `energy` up to summation order."""
        x = paths[:, :-1]
        v = (paths[:, 1:]-x)/self.dt
        b, n, d = x.shape
        mismatch = ((v.reshape(b*n, d)-self.drift(x.reshape(b*n, d)))**2).sum(-1).reshape(b, n)
        return self.dt/(4*self.diffusion)*mismatch.sum(1)


def untrained_potential(seed: int, **kwargs) -> ConservativePotential:
    """Untrained control with the paper run's weights."""
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(seed)
        # the paper run first built an archived potential and a 1-128-128-1 velocity net
        ConservativePotential(**kwargs)
        for shape in [(1, 128), (128, 128), (128, 1)]:
            nn.Linear(*shape)
        return ConservativePotential(**kwargs)

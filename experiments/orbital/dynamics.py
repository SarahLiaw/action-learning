"""Velocity-Verlet gravity, the path-level solar-mass mixture and its exact correction.

Units: AU, AU/day and days; catalog masses in kg.
"""
from __future__ import annotations

import math

import torch

AU_M = 149.6e6*1000.0
DAY_S = 24.0*3600.0
M_SUN_KG = 1.9885e30
G_SI = 6.67428e-11
G = G_SI/AU_M**3*M_SUN_KG*DAY_S**2          # AU^3 / (M_sun day^2)
SOFTENING_AU = 1e-9
LOG_2PI = math.log(2.0*math.pi)


def acceleration(q: torch.Tensor, masses_kg) -> torch.Tensor:
    """Softened Newtonian acceleration; q: [..., M, 3]."""
    m = torch.as_tensor(masses_kg, dtype=q.dtype, device=q.device)/M_SUN_KG
    n = q.shape[-2]
    delta = q.unsqueeze(-3)-q.unsqueeze(-2)                      # [..., i, j] = q_j - q_i
    inv_r3 = (delta.square().sum(dim=-1)+SOFTENING_AU**2).pow(-1.5)
    inv_r3 = inv_r3.masked_fill(torch.eye(n, dtype=torch.bool, device=q.device), 0.0)
    source = m.reshape((1,)*(q.ndim-2)+(1, n, 1))
    return G*(delta*inv_r3.unsqueeze(-1)*source).sum(dim=-2)


def verlet(x: torch.Tensor, masses_kg, dt: float) -> torch.Tensor:
    """One velocity-Verlet step of x = (q, v): [..., M, 6]."""
    q, v = x[..., :3], x[..., 3:]
    a = acceleration(q, masses_kg)
    q1 = q+v*dt+0.5*a*dt**2
    v1 = v+0.5*(a+acceleration(q1, masses_kg))*dt
    return torch.cat((q1, v1), dim=-1)


class SolarMassMixture:
    """Paths whose solar mass m_sun*s_z is drawn once per path, with shared Gaussian noise."""

    def __init__(self, masses_kg, *, dt: float, position_std: float, velocity_std: float,
                 sun_scales=(1.0, 1.0), weights=(0.5, 0.5)):
        base = torch.as_tensor(masses_kg, dtype=torch.float64).clone()
        self.masses = []
        for s in sun_scales:
            m = base.clone()
            m[0] *= s
            self.masses.append(m)
        self.shared = all(torch.equal(m, self.masses[0]) for m in self.masses)
        self.dt = float(dt)
        self.std = (position_std,)*3+(velocity_std,)*3
        self.weights = tuple(weights)

    @classmethod
    def data(cls, masses_kg, eta: float, **kw) -> 'SolarMassMixture':
        """Data law: z in {-1,+1} equally likely, solar mass m_sun(1+z*eta)."""
        return cls(masses_kg, sun_scales=(1.0-eta, 1.0+eta), **kw)

    @classmethod
    def reference(cls, masses_kg, **kw) -> 'SolarMassMixture':
        """Reference law: nominal mass; two identical components keep the RNG stream of the data law."""
        return cls(masses_kg, sun_scales=(1.0, 1.0), **kw)

    def mean(self, x: torch.Tensor, z) -> torch.Tensor:
        """Regime-conditional Verlet mean; z is a component index or a per-path index tensor."""
        if isinstance(z, int):
            return verlet(x, self.masses[z], self.dt)
        if self.shared:
            return verlet(x, self.masses[0], self.dt)
        out = verlet(x, self.masses[0], self.dt)
        pick = z.reshape(z.shape+(1,)*(x.ndim-z.ndim))
        for r in range(1, len(self.masses)):
            out = torch.where(pick == r, verlet(x, self.masses[r], self.dt), out)
        return out

    def _std(self, like: torch.Tensor) -> torch.Tensor:
        return torch.as_tensor(self.std, dtype=like.dtype, device=like.device)

    def sample_latent(self, n: int, like: torch.Tensor, generator) -> torch.Tensor:
        w = torch.as_tensor(self.weights, dtype=like.dtype, device=like.device)
        return torch.multinomial(w.expand(n, -1), 1, replacement=True, generator=generator).reshape(n)

    def rollout(self, x0: torch.Tensor, steps: int, generator, return_latent: bool = False, z=None):
        """Sample [n, steps+1, M, 6] paths from x0 [n, M, 6], holding each path's regime fixed.

        An integer z fixes the component of every path and skips the latent draw.
        """
        if z is None:
            z = self.sample_latent(x0.shape[0], x0, generator)
        std = self._std(x0)
        states, x = [x0], x0
        for _ in range(steps):
            mean = self.mean(x, z)
            x = mean+torch.randn(mean.shape, dtype=mean.dtype, device=mean.device, generator=generator)*std
            states.append(x)
        paths = torch.stack(states, dim=-3)
        return (paths, z) if return_latent else paths

    def component_log_probs(self, paths: torch.Tensor) -> torch.Tensor:
        """log p(x_1..x_N | x_0, z) for every component: [..., R]."""
        std = self._std(paths)
        def given(r):
            resid = (paths[..., 1:, :, :]-self.mean(paths[..., :-1, :, :], r))/std
            return (-0.5*(resid.square()+2.0*torch.log(std)+LOG_2PI)).sum(dim=(-3, -2, -1))
        comps = [given(0)]
        for r in range(1, len(self.masses)):
            comps.append(comps[0] if self.shared else given(r))
        return torch.stack(comps, dim=-1)

    def log_prob(self, paths: torch.Tensor) -> torch.Tensor:
        """Exact latent-marginal log p(x_1..x_N | x_0)."""
        c = self.component_log_probs(paths)
        w = torch.as_tensor(self.weights, dtype=c.dtype, device=c.device)
        return torch.logsumexp(c+torch.log(w/w.sum()), dim=-1)


def exact_correction(data: SolarMassMixture, reference: SolarMassMixture, paths: torch.Tensor) -> torch.Tensor:
    """G* = log[(e^{l_+} + e^{l_-})/2] - l_0, evaluated as S_ref - S_data."""
    return (-reference.log_prob(paths))-(-data.log_prob(paths))

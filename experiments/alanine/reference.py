"""Guided wrapped-Gaussian reference on the torus, the extraction event, and reference banks."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

from leap.utils import sha256, write_json
from .data import BASIN_A, BASIN_B, delta, wrap


def wrapped_log_prob(value, mean, variance):
    """log wrap(N(mean, variance)) per coordinate; images summed through ten standard deviations."""
    variance = torch.as_tensor(variance, device=value.device, dtype=value.dtype)
    radius = max(1, math.ceil((10*math.sqrt(float(variance.detach().max()))+math.pi)/(2*math.pi)))
    shifts = torch.arange(-radius, radius+1, device=value.device, dtype=value.dtype)*2*math.pi
    d = delta(mean, value)[..., None]+shifts
    return torch.logsumexp(-d.square()/(2*variance[..., None])-0.5*torch.log(2*math.pi*variance[..., None]), -1)


def _wrap_scalar(x):
    return float((x+math.pi) % (2.0*math.pi)-math.pi)


class Reference:
    """Absorbed Euler chain with drift b_hat(x) + gamma wrap(x_B - x)/(T - t + 2 dt) and variance 2 D dt."""

    def __init__(self, field, dt, diffusion, steps=200, gamma=15., epsilon_factor=2., grace=5, guidance=True):
        self.field, self.dt, self.diffusion, self.steps = field, float(dt), float(diffusion), int(steps)
        self.gamma, self.epsilon_factor, self.grace, self.guidance = float(gamma), float(epsilon_factor), int(grace), guidance
        self.basin_a, self.basin_b = BASIN_A, BASIN_B
        self.target = (_wrap_scalar(0.5*(BASIN_B['phi_min']+BASIN_B['phi_max'])),
                       _wrap_scalar(0.5*(BASIN_B['psi_min']+BASIN_B['psi_max'])))
        self.sigma = math.sqrt(max(2.0*self.diffusion*self.dt, 1e-12))

    def in_basin(self, points, basin):
        """Closed-rectangle membership; points [B, 2]."""
        phi, psi = points[:, 0], points[:, 1]
        return (phi >= basin['phi_min']) & (phi <= basin['phi_max']) & (psi >= basin['psi_min']) & (psi <= basin['psi_max'])

    def physics_drift(self, points):
        """Nearest-bin training drift b_hat; points [B, 2]."""
        edges = torch.as_tensor(self.field['edges'], device=points.device, dtype=points.dtype)
        grid = [torch.as_tensor(self.field[k], device=points.device, dtype=points.dtype) for k in ('drift_phi', 'drift_psi')]
        n = grid[0].shape[0]
        i = torch.bucketize(points[:, 0].contiguous(), edges[1:], right=False).clamp(0, n-1)
        j = torch.bucketize(points[:, 1].contiguous(), edges[1:], right=False).clamp(0, n-1)
        return torch.stack([grid[0][i, j], grid[1][i, j]], dim=-1)

    def drift(self, points, k, n):
        """Sampler drift at transition k of n."""
        b = self.physics_drift(points)
        if not self.guidance:
            return b
        remaining = max(float(n)*self.dt-float(k)*self.dt+self.epsilon_factor*self.dt, self.epsilon_factor*self.dt)
        target = torch.tensor(self.target, device=points.device, dtype=points.dtype).view(1, 2)
        return b+self.gamma*delta(points, target)/remaining

    @torch.no_grad()
    def sample(self, x0, n):
        """Rollouts absorbed on arrival in B, or in A after the grace steps; absorbed rows stay frozen."""
        paths = torch.zeros((len(x0), n+1, 2), device=x0.device, dtype=x0.dtype)
        paths[:, 0] = x0
        curr, absorbed = x0.clone(), torch.zeros(len(x0), device=x0.device, dtype=torch.bool)
        for k in range(n):
            active = ~absorbed
            proposal = wrap(curr+self.drift(curr, k, n)*self.dt+torch.randn_like(curr)*self.sigma)
            curr = torch.where(active.unsqueeze(-1), proposal, curr)
            paths[:, k+1] = curr
            arrived = self.in_basin(curr, self.basin_b) | (self.in_basin(curr, self.basin_a) & (k+1 > self.grace))
            absorbed = absorbed | (arrived & active)
        return paths

    def success(self, paths):
        """First arrival in B precedes any return to A after the grace steps."""
        n = paths.shape[1]-1
        t = torch.arange(1, n+1, device=paths.device)[None].expand(len(paths), n)
        in_a = self.in_basin(paths[:, 1:].reshape(-1, 2), self.basin_a).reshape(-1, n) & (t > self.grace)
        in_b = self.in_basin(paths[:, 1:].reshape(-1, 2), self.basin_b).reshape(-1, n)
        first_a, first_b = (torch.where(m, t, n+1).min(1).values for m in (in_a, in_b))
        return (first_b < first_a) & (first_b <= n)

    def sample_successful(self, x0, n, max_retries=512, factor=8):
        """Rejection-sampled reactive rollouts and the proposal acceptance rate."""
        init = self.sample(x0, n)
        resolved = self.success(init)
        paths = torch.zeros((len(x0), n+1, 2), dtype=x0.dtype, device=x0.device)
        paths[resolved] = init[resolved]
        total, accepted, rounds = len(x0), int(resolved.sum().item()), 0
        while not resolved.all() and rounds < max_retries:
            rounds += 1
            todo = torch.where(~resolved)[0]
            proposal = self.sample(x0[todo].repeat_interleave(factor, dim=0), n)
            hit = self.success(proposal).reshape(-1, factor)
            found = hit.any(dim=1)
            if found.any():
                chosen = proposal.reshape(len(todo), factor, n+1, 2)[torch.arange(len(todo), device=x0.device), torch.argmax(hit.float(), dim=1)]
                paths[todo[found]] = chosen[found]
                resolved[todo[found]] = True
            total += len(todo)*factor
            accepted += int(hit.sum().item())
        if not resolved.all():
            raise RuntimeError(f'unresolved reactive references: {int((~resolved).sum())}/{len(x0)}')
        return paths, accepted/max(total, 1)

    def step_log_prob(self, paths):
        """Per-transition log density [B, N] under the sampler's drift."""
        x, nxt = paths[:, :-1], paths[:, 1:]
        n = x.shape[1]
        drift = self.physics_drift(x.reshape(-1, 2)).reshape_as(x)
        if self.guidance:
            remaining = ((n-torch.arange(n, device=x.device, dtype=x.dtype)+self.epsilon_factor)*self.dt).clamp_min(self.epsilon_factor*self.dt)
            drift = drift+self.gamma*delta(x, x.new_tensor(self.target))/remaining[None, :, None]
        return wrapped_log_prob(nxt, x+drift*self.dt, self.sigma**2).sum(-1)

    def log_prob(self, paths, mask=None):
        """Summed log density over (masked) transitions."""
        step = self.step_log_prob(paths)
        return (step if mask is None else step*mask.float()).sum(dim=1)


def make_reference(data, field, cfg, guidance=True):
    """Reference from a split's frame spacing and training diffusion; guidance=False gives the physical model."""
    return Reference(field, float(data['dt_ps'][0]), float(data['D_eff'][0]), cfg['event']['horizon'],
                     cfg['reference']['gamma'], cfg['reference']['epsilon_factor'], cfg['event']['grace'], guidance)


def event_valid(paths, masks, ref):
    """The extraction event: start in A, first arrival in B at the last active frame, no return to A after grace."""
    b, n, _ = paths.shape
    tau = masks.sum(-1).long()
    k = torch.arange(n, device=paths.device)[None]
    target = ref.in_basin(paths.reshape(-1, 2), ref.basin_b).reshape(b, n)
    start = ref.in_basin(paths.reshape(-1, 2), ref.basin_a).reshape(b, n)
    early_b = (target & (k > 0) & (k < tau[:, None])).any(-1)
    back_a = (start & (k > ref.grace) & (k <= tau[:, None])).any(-1)
    hit = target.gather(1, tau[:, None]).squeeze(1)
    terminal = paths[torch.arange(b, device=paths.device), tau]
    frozen = ((delta(terminal[:, None], paths).abs().amax(-1) < 1e-5) | (k <= tau[:, None])).all(-1)
    expected = torch.arange(n-1, device=paths.device)[None] < tau[:, None]
    return hit & ~back_a & ~early_b & frozen & start[:, 0] & (expected == masks.bool()).all(-1)


def active(paths):
    """Transitions with nonzero displacement."""
    return (delta(paths[:, :-1], paths[:, 1:]).norm(dim=-1) > 1e-10).float()


def bank(paths_ab, ref, count, seed, chunk=512, max_retries=512, factor=8):
    """`count` reactive references per data initial state: paths [n, count, N+1, 2], masks, log densities."""
    torch.manual_seed(seed)
    x0 = paths_ab[:, 0].repeat_interleave(count, 0)
    parts, rates = [], []
    for s in range(0, len(x0), chunk):
        p, rate = ref.sample_successful(x0[s:s+chunk], ref.steps, max_retries, factor)
        assert event_valid(p, active(p), ref).all(), 'reference event mismatch'
        parts.append(p)
        rates.append(rate)
    p = torch.cat(parts).reshape(len(paths_ab), count, ref.steps+1, 2)
    flat = p.flatten(0, 1)
    m = active(flat)
    with torch.no_grad():
        q = ref.log_prob(flat, m).reshape(p.shape[:2])
    return {'paths': p, 'masks': m.reshape(len(p), count, ref.steps), 'logq': q}, float(np.mean(rates))


def cached_bank(folder, data, ref, count, seed):
    """Load, or sample and store, the bank `reference_{seed}_{n}_{count}.pt`."""
    path = Path(folder)/f'reference_{seed}_{len(data["paths_ab"])}_{count}.pt'
    if path.exists():
        return torch.load(path, map_location=data['paths_ab'].device, weights_only=True)
    refs, rate = bank(data['paths_ab'], ref, count, seed)
    torch.save({k: v.cpu() for k, v in refs.items()}, path)
    write_json(path.with_suffix('.json'), {'mean_proposal_acceptance': rate, 'strict_event_conditioning': True,
                                           'count_per_context': count, 'seed': seed, 'sha256': sha256(path)})
    return refs

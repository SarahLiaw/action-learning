"""Hard-endpoint path inference with L-BFGS, and the true discrete action used to score paths."""
from __future__ import annotations

import torch


def true_residual(paths: torch.Tensor, dt: float) -> torch.Tensor:
    """v_k - b(x_k) for the analytic drift b(x) = x - x^3."""
    x = paths[:, :-1]
    return (paths[:, 1:]-x)/dt-x+x**3


def true_cost(paths: torch.Tensor, dt: float) -> torch.Tensor:
    """Q_true = D S_true = dt/4 sum_k (v_k - b(x_k))^2 of each path [B, N+1, 1]."""
    return (dt/4*true_residual(paths, dt).square()).sum((1, 2))


def analytic_action(dt: float, diffusion: float):
    """The analytic drift's own objective, S_true."""
    return lambda paths: (dt/(4*diffusion)*true_residual(paths, dt).square()).sum((1, 2))


def drift_action(drift, dt: float, diffusion: float):
    """dt/(4D) sum_k (v_k - drift(x_k))^2 for a closed-form drift."""
    def action(paths):
        x = paths[:, :-1]
        return (dt/(4*diffusion)*((paths[:, 1:]-x)/dt-drift(x)).square()).sum((1, 2))
    return action


def initial_paths(start: float, end: float, n_steps: int, waits, device) -> torch.Tensor:
    """Hold `start` for a fraction w of the horizon, then move linearly to `end`.  [len(waits), N+1, 1]."""
    t = torch.linspace(0, 1, n_steps+1, device=device)
    return torch.stack([(start+(end-start)*((t-w)/(1-w)).clamp(0, 1)).view(-1, 1) for w in waits])


def optimize_path(action, endpoints, horizon: float, dt: float, waits, lbfgs: dict, device):
    """Minimize the summed action of all initializations jointly; return the lowest-action path and its action."""
    start, end = endpoints
    interior = initial_paths(start, end, round(horizon/dt), waits, device)[:, 1:-1].clone().requires_grad_()
    n = len(waits)

    def full():
        return torch.cat([interior.new_full((n, 1, 1), start), interior, interior.new_full((n, 1, 1), end)], 1)

    opt = torch.optim.LBFGS([interior], lr=1, line_search_fn='strong_wolfe', **lbfgs)

    def closure():
        opt.zero_grad()
        loss = action(full()).sum()
        loss.backward()
        return loss

    opt.step(closure)
    paths = full().detach()
    with torch.no_grad():
        values = action(paths)
    best = int(values.argmin())
    return paths[best:best+1], values[best]

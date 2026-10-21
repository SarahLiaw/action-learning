"""LEAP on your own system: observed paths, a reference process and a path scorer.

Here the paths come from a double well whose drift LEAP never sees, the reference is Brownian motion from the same
starts and the scorer is the Girsanov form of a learned drift (Eq. 9). Swap in your own paths, reference and network.

    python examples/custom_system.py
"""
import torch
from torch import nn

from leap import EulerReference, fit, optimize_path, structured_correction

D, DT, N = 0.2, 0.01, 100


def true_drift(x):
    return x-x**3


class DriftCorrection(nn.Module):
    """G_theta(path) for a learned drift u_theta added to the reference drift."""

    def __init__(self, width: int = 64):
        super().__init__()
        self.u = nn.Sequential(nn.Linear(1, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 1))

    def forward(self, paths):
        return structured_correction(paths, self.u, None, DT, D)


def true_cost(paths):
    """Q_true of Eq. (B.3) under the true drift."""
    v = paths.diff(dim=-2)/DT
    return DT/4*((v-true_drift(paths[..., :-1, :]))**2).sum((-2, -1))


def main(n_paths: int = 1024, epochs: int = 30, seed: int = 0) -> dict:
    torch.manual_seed(seed)
    # 1. Observed paths, simulated here; LEAP only sees the paths.
    paths = EulerReference(D, DT, drift=true_drift).sample(3*torch.rand(n_paths, 1)-1.5, N)[:, 0]

    # 2. A reference we can sample and evaluate, matched to each path's context (its start).
    reference = EulerReference(D, DT)
    model = DriftCorrection()
    fit(model, paths, lambda batch, k: reference.sample(batch[:, 0], N, k), k=8, epochs=epochs)

    # 3. The learned drift recovers b(x) = x - x^3.
    x = torch.linspace(-1.5, 1.5, 301)[:, None]
    with torch.no_grad():
        drift_mse = ((model.u(x)-true_drift(x))**2).mean().item()

    # 4. Reuse the learned action S = -log p_ref - G: the most likely path from the left well to the saddle at 0
    #    over T = 8, scored with the true path cost against a straight line and the true optimum.
    model.requires_grad_(False)
    line = torch.linspace(-1, 0, 8*N+1)[None, :, None]
    fixed = torch.zeros_like(line, dtype=torch.bool)
    fixed[:, [0, -1]] = True
    leap_path = optimize_path(lambda p: -reference.log_prob(p)-model(p), line, fixed)
    best_path = optimize_path(true_cost, line, fixed)
    out = {'drift_mse': drift_mse, 'cost_line': true_cost(line).item(), 'cost_leap': true_cost(leap_path).item(),
           'cost_optimal': true_cost(best_path).item()}
    print(f"drift MSE {out['drift_mse']:.4f}; true path cost: straight line {out['cost_line']:.4f}, "
          f"LEAP {out['cost_leap']:.4f}, optimal {out['cost_optimal']:.4f}")
    return out


if __name__ == '__main__':
    main()

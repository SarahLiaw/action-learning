"""Path inference with a learned action (Eqs. 14 and 15, Algorithm 2)."""
from __future__ import annotations

import torch


def optimize_path(objective, init: torch.Tensor, fixed: torch.Tensor, steps: int = 500) -> torch.Tensor:
    """Minimize objective over the free entries of init with L-BFGS; entries where fixed is True keep their values.

    objective maps paths [..., N+1, d] to costs [...], e.g. the learned action plus preferences or penalties.
    """
    base = init.detach()
    z = base.clone().requires_grad_(True)
    opt = torch.optim.LBFGS([z], max_iter=steps, line_search_fn='strong_wolfe')

    def closure():
        opt.zero_grad()
        loss = objective(torch.where(fixed, base, z)).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return torch.where(fixed, base, z).detach()

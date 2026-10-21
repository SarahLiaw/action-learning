"""Solar-mass evidence of a path and the mass perturbation implied by a learned correction (Section 4.2)."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
import torch

from .evaluate import physics_for


@torch.no_grad()
def solar_evidence(bundle: dict, batch_size: int = 16) -> tuple[np.ndarray, np.ndarray]:
    """S = sum_k e_k.J_k and I = sum_k |J_k|^2 per test candidate; the path summary is A = eta S."""
    physics = physics_for(bundle)
    score, info = [], []
    for x in bundle['test_candidates'].split(batch_size):
        e, j = physics.features(x.to(torch.float64))
        score.append((e*j).sum(dim=(1, 2, 3)))
        info.append(j.square().sum(dim=(1, 2, 3)))
    return torch.cat(score).numpy(), torch.cat(info).numpy()


def log_cosh(x: np.ndarray) -> np.ndarray:
    """log cosh x without overflow."""
    a = np.abs(x)
    return a+np.log1p(np.exp(-2.0*a))-np.log(2.0)


def centered(v, ids: np.ndarray) -> np.ndarray:
    """v minus its mean within each context."""
    out = np.array(v, dtype=np.float64)
    for c in np.unique(ids):
        out[ids == c] -= out[ids == c].mean()
    return out


def implied_eta(pred, score: np.ndarray, info: np.ndarray, context_ids, eta: float) -> float:
    """The eta whose exact correction log cosh(eta S) - eta^2 I / 2 best fits `pred` within contexts."""
    ids = np.asarray(context_ids)
    target = centered(pred, ids)

    def loss(log_eta):
        e = np.exp(log_eta)
        return float(np.sum((target-centered(log_cosh(e*score)-0.5*e*e*info, ids))**2))

    return float(np.exp(minimize_scalar(loss, bounds=(np.log(eta/100), np.log(eta*30)), method='bounded').x))

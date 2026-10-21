"""Classical drift estimators fit to the selected paths, and the geometric path control."""
from __future__ import annotations

import numpy as np
import torch


def transitions(paths: torch.Tensor, dt: float):
    """States x_k and velocities (x_{k+1}-x_k)/dt of every transition, as float64 vectors."""
    data = paths.numpy()
    x = data[:, :-1].reshape(-1).astype(np.float64)
    v = (np.diff(data, axis=1)/dt).reshape(-1).astype(np.float64)
    return x, v


def fit_cubic(x: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Least-squares coefficients of v on (1, x, x^2, x^3)."""
    design = np.stack([np.ones_like(x), x, x*x, x*x*x], -1)
    return np.linalg.lstsq(design, v, rcond=None)[0]


def polynomial(coef, x):
    """sum_i coef_i x^i, for arrays or tensors."""
    return sum(coef[i]*x**i for i in range(len(coef)))


def kramers_moyal(x: np.ndarray, v: np.ndarray, low: float, high: float, points: int):
    """Mean velocity in equal bins centered at linspace(low, high, points).  Returns (field, counts)."""
    half = (high-low)/(points-1)/2
    idx = np.digitize(x, np.linspace(low-half, high+half, points+1))-1
    keep = (idx >= 0) & (idx < points)
    count = np.bincount(idx[keep], minlength=points)
    if count.min() == 0:
        raise ValueError('a Kramers-Moyal bin holds no transitions')
    return np.bincount(idx[keep], weights=v[keep], minlength=points)/count, count


def geometric_path(start: float, end: float, horizon: float, dt: float, crossing: float) -> np.ndarray:
    """Wait at `start`, then move linearly to `end` during the final min(T, crossing)."""
    t = np.arange(round(horizon/dt)+1)*dt
    return start+(end-start)*np.clip((t-max(horizon-crossing, 0))/min(horizon, crossing), 0, 1)

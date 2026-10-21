"""Drift and path metrics, and their aggregation over seeds."""
from __future__ import annotations

import numpy as np
import torch

from .baselines import geometric_path
from .inference import optimize_path, true_cost
from .simulate import drift


def model_drift_mse(model, low: float, high: float, points: int, device) -> float:
    """MSE of a learned drift against b on linspace(low, high, points)."""
    x = torch.linspace(low, high, points, device=device)[:, None]
    return (model.drift(x).detach()-drift(x)).square().mean().item()


def field_mse(field: np.ndarray, low: float, high: float, points: int) -> float:
    """MSE of drift values given on linspace(low, high, points)."""
    x = np.linspace(low, high, points)
    return float(np.mean((field-drift(x))**2))


def infer(action, *, diffusion: float, dt: float, endpoints, horizons, waits, lbfgs: dict, device):
    """Solve every horizon with `action`; returns rows of scaled costs and the chosen paths."""
    rows, paths = [], {}
    for horizon in horizons:
        path, value = optimize_path(action, endpoints, horizon, dt, waits, lbfgs, device)
        rows.append({'horizon': horizon, 'true_scaled_cost': true_cost(path, dt).item(),
                     'learned_scaled_cost': diffusion*value.item()})
        paths[horizon] = path.cpu().numpy()
    return rows, paths


def geometric(*, dt: float, endpoints, horizons, crossing: float):
    """True scaled cost of the wait-then-cross path at each horizon, in float64."""
    rows, paths = [], {}
    for horizon in horizons:
        path = torch.from_numpy(geometric_path(*endpoints, horizon, dt, crossing)).view(1, -1, 1)
        rows.append({'horizon': horizon, 'true_scaled_cost': true_cost(path, dt).item()})
        paths[horizon] = path.numpy()
    return rows, paths


def summarize(records: list[dict]) -> dict:
    """Mean and sample SD over seeds of each drift MSE and each (method, horizon) true cost."""
    def stat(values):
        v = np.asarray(values, dtype=np.float64)
        return {'mean': float(v.mean()), 'sd': float(v.std(ddof=1)) if v.size > 1 else float('nan')}

    def cost(record, method, horizon):
        return next(r['true_scaled_cost'] for r in record['path_costs']
                    if r['method'] == method and r['horizon'] == horizon)

    costs = {}
    for row in records[0]['path_costs']:
        m, h = row['method'], row['horizon']
        costs.setdefault(m, {})[f'{h:g}'] = stat([cost(r, m, h) for r in records])
    return {'seeds': [r['seed'] for r in records],
            'drift_mse': {m: stat([r['drift_mse'][m] for r in records]) for m in records[0]['drift_mse']},
            'true_scaled_cost': costs}

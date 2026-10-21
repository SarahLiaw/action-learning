"""Circular MSE and path diagnostics, cross-fit summaries, transfer AUROC, and landscape/example selection."""
from __future__ import annotations

import math

import numpy as np
import torch
from scipy.ndimage import gaussian_filter, map_coordinates

from leap.metrics import binary_auroc
from .data import delta, wrap_np
from .inference import continuous_valid, preference
from .reference import event_valid

R_KCAL = 0.00198720425864083


def dense_path(path, subdivisions=8):
    """Knots plus `subdivisions` points per shortest-arc segment."""
    f = np.arange(subdivisions)/subdivisions
    p = wrap_np(path[:-1, None]+f[None, :, None]*wrap_np(path[1:]-path[:-1])[:, None])
    return np.concatenate([p.reshape(-1, 2), path[-1:]])


def occupancy_cost(path, counts):
    """F_occ = -log[(n(x)+1)/(n_max+1)] with periodic bilinear interpolation of training counts."""
    energy = -np.log((counts+1)/(counts.max()+1))
    coord = (wrap_np(path)+np.pi)/(2*np.pi)*len(counts)-.5
    return map_coordinates(np.tile(energy, (3, 3)), (coord+len(counts)).T, order=1, mode='nearest')


def winding(path):
    return np.rint((wrap_np(path[1:]-path[:-1]).sum(0)-(path[-1]-path[0]))/(2*np.pi)).astype(int)


def metrics(p, truth, obs, m, ref, physical, counts):
    """Per-path circular MSE on missing active frames, feasibility, projected cost and occupancy diagnostics."""
    knots = torch.cat([torch.ones_like(m[:, :1]), m], 1).bool()
    free = knots & ~obs
    n = m.sum(-1).clamp_min(1)
    with torch.no_grad():
        nll = -physical.log_prob(p, m)/n
    maxphi = torch.where(knots, p[..., 0].abs(), 0).amax(-1)*180/math.pi
    out = {'circular_mse_rad2': (delta(p, truth).square().mean(-1)*free).sum(-1)/free.sum(-1).clamp_min(1),
           'projected_transition_nll_per_step': nll,
           'largest_step_deg': torch.where(m.bool(), delta(p[:, :-1], p[:, 1:]).norm(dim=-1), 0).amax(-1)*180/math.pi,
           'max_abs_phi_deg': maxphi, 'event_valid': event_valid(p, m, ref), 'fence_feasible': maxphi <= 150.0001,
           'continuous_fence_feasible': continuous_valid(p, m),
           'observation_max_error_rad': torch.where(obs[..., None], delta(p, truth).abs(), 0).amax((1, 2)),
           'active_steps': n, 'free_knots': free.sum(-1)}
    out = {k: v.cpu().tolist() for k, v in out.items()}
    extra = {k: [] for k in ('segment_fence_feasible', 'knot_occupancy_max', 'segment_occupancy_max',
                             'segment_occupancy_mean', 'phi_winding')}
    for path, tau in zip(p.detach().cpu().numpy(), m.sum(-1).cpu().numpy().astype(int)):
        path = path[:tau+1]
        dense = occupancy_cost(dense_path(path), counts)
        extra['segment_fence_feasible'].append(bool((abs(path[:, 0]) <= np.deg2rad(150)+2e-6).all()
                                                    and (abs(np.diff(path[:, 0])) <= np.pi-1e-3+2e-6).all()))
        extra['knot_occupancy_max'].append(float(occupancy_cost(path, counts).max()))
        extra['segment_occupancy_max'].append(float(dense.max()))
        extra['segment_occupancy_mean'].append(float(dense.mean()))
        extra['phi_winding'].append(int(winding(path)[0]))
    return out | extra


def summarize(records):
    """Mean and sample SD across fits of context-averaged metrics; `mse` averages contexts with missing frames."""
    groups = {}
    for r in records:
        groups.setdefault((r['task'], r['method']), []).append(r)
    rows = []
    for (task, method), runs in groups.items():
        per = []
        for r in runs:
            met = {k: np.asarray(v) for k, v in r['metrics'].items()}
            d = {k: float(v.mean()) for k, v in met.items() if k != 'free_knots'}
            d['mse'] = float(met['circular_mse_rad2'][met['free_knots'] > 0].mean())
            per.append(d)
        rows.append({'task': task, 'method': method, 'n_paths': len(runs[0]['case_indices']), 'n_runs': len(runs),
                     'metrics': {k: {'mean': float(np.mean([q[k] for q in per])), 'sd': float(np.std([q[k] for q in per], ddof=1))}
                                 for k in per[0]}, 'per_run': per})
    return rows


def spread(values):
    return {'mean': float(np.mean(values)), 'sd': float(np.std(values, ddof=1)), 'n': len(values), 'values': values}


def context_auroc(positive, reference):
    """Tie-aware AUROC of each data path against its own references; positive [n], reference [n, K]."""
    labels = np.r_[1, np.zeros(reference.shape[1])]
    return np.array([binary_auroc(np.r_[positive[i], reference[i]], labels) for i in range(len(positive))])


def summarize_preference(records, methods, weights):
    """Context means per fit, then mean/SD over fits; changes relative to the same fit at weight 0."""
    keys = ('preference', 'base_action', 'projected_transition_nll_per_step', 'largest_step_deg', 'event_valid',
            'observation_max_error_rad', 'circular_mse_rad2')
    rows = []
    for method in methods:
        for weight in weights:
            subset = [r for r in records if r['method'] == method and r['weight'] == weight]
            out = {k: spread([float(np.mean(r['metrics'][k])) for r in subset]) for k in keys}
            for key in ('base_action', 'projected_transition_nll_per_step'):
                change = []
                for r in subset:
                    base = next(t for t in records if t['run'] == r['run'] and t['method'] == method and t['weight'] == 0)
                    change.append(float(np.mean(np.array(r['metrics'][key])-base['metrics'][key])))
                out['change_'+key] = spread(change)
            rows.append({'method': method, 'weight': weight, 'n_fits': len(subset),
                         'n_contexts': len(subset[0]['case_indices']), 'metrics': out})
    return rows


def landscape(counts, temperature):
    """Empirical free energy (kcal/mol) on a 1-degree grid: periodic one-bin Gaussian smoothing, pseudocount 0.5."""
    n = len(counts)
    density = gaussian_filter(counts, 1., mode='wrap')+.5
    grid = np.linspace(-180, 180, 361)
    a, b = np.meshgrid(grid, grid, indexing='ij')
    smooth = map_coordinates(np.tile(density, (3, 3)), np.stack([(a+180)/360*n-.5, (b+180)/360*n-.5])+n, order=1, mode='nearest')
    return grid, -R_KCAL*temperature*np.log(smooth/smooth.max())


def landscape_examples(test, runs, cfg, constraint=False):
    """Boundary-matched context (endpoints closest to the basin centers); optionally the first with a bound response.

    runs: (evaluation, predictions) per fit; the first fit's eligibility defines the common task domain.
    """
    n = test['masks_ab'].sum(-1).astype(int)
    centers = np.radians(cfg['centers_deg'])
    distance = lambda i: float(np.square(wrap_np(test['paths_ab'][i, [0, n[i]]]-centers)).sum())
    eligible = sorted((i for i in runs[0][0]['eligibility']['completion_fence']['included_indices'] if n[i] >= cfg['min_steps']),
                      key=lambda i: (distance(i), i))
    found = {'case': eligible[0], 'boundary_distance_rad2': distance(eligible[0]), 'eligible': eligible}
    if constraint:
        def path(i, task, fit):
            meta, pred = runs[fit]
            return pred[f'{task}__leap'][meta['eligibility'][task]['included_indices'].index(i), :n[i]+1].astype(float)
        change = {i: float(np.mean([np.degrees(np.linalg.norm(wrap_np(path(i, 'endpoints', f)-path(i, 'endpoints_fence', f)), axis=-1).mean())
                                    for f in range(len(runs))])) for i in eligible}
        found |= {'constraint_case': next(i for i in eligible if change[i] >= cfg['min_change_deg']), 'change_deg': change}
    return found


def composed_table(paths, model, geometry, cfg):
    """S_theta, M, peak restrained energy rise and largest saved step for each displayed request."""
    center, width = [math.radians(c) for c in cfg['center_deg']], math.radians(cfg['width_deg'])
    rows = []
    for key, path in paths.items():
        x, m = torch.tensor(path, dtype=torch.float32)[None], torch.ones(1, len(path)-1)
        with torch.no_grad():
            rows.append({'path': key, 'action_nats': float(model(x, m)[0]), 'preference': float(preference(x, m, center, width)[0]),
                         **geometry[key]})
    return rows

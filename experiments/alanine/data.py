"""Torus helpers, reactive path extraction, replica-disjoint splits and observation masks."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

from leap.utils import sha256, write_json

# C7eq (A) and C7ax (B) rectangles, radians.
BASIN_A = {'phi_min': float(np.radians(-150.)), 'phi_max': float(np.radians(-30.)),
           'psi_min': float(np.radians(0.)), 'psi_max': float(np.radians(180.))}
BASIN_B = {'phi_min': float(np.radians(30.)), 'phi_max': float(np.radians(130.)),
           'psi_min': float(np.radians(-180.)), 'psi_max': float(np.radians(0.))}


def wrap(x):
    """Principal angle in [-pi, pi)."""
    return torch.remainder(x+math.pi, 2.0*math.pi)-math.pi


def delta(a, b):
    """Shortest periodic displacement b - a."""
    return torch.atan2(torch.sin(b-a), torch.cos(b-a))


def wrap_np(x):
    """Principal angle in [-pi, pi)."""
    return (x+np.pi) % (2.0*np.pi)-np.pi


def embed(x):
    """(cos phi, cos psi, sin phi, sin psi)."""
    return torch.cat([x.cos(), x.sin()], -1)


def angles(z):
    return torch.atan2(z[..., 2:], z[..., :2])


def in_basin_np(phi, psi, basin):
    """Periodic rectangle membership used for extraction."""
    inside = []
    for x, lo, hi in ((phi, basin['phi_min'], basin['phi_max']), (psi, basin['psi_min'], basin['psi_max'])):
        v, lo, hi = wrap_np(x), wrap_np(lo), wrap_np(hi)
        inside.append((v >= lo) & (v <= hi) if lo <= hi else (v >= lo) | (v <= hi))
    return inside[0] & inside[1]


def _next_true(flag):
    """Index of the first True at or after each position (len if none)."""
    index = np.where(flag, np.arange(len(flag)), len(flag))
    return np.minimum.accumulate(index[::-1])[::-1]


def extract(phi, psi, start, target, horizon=200, grace=5):
    """Greedy first arrivals start -> target without return after `grace` steps; later frames repeat the terminal one."""
    s, t = in_basin_np(phi, psi, start), in_basin_np(phi, psi, target)
    i = np.arange(max(len(phi)-horizon-1, 0))
    arrive = _next_true(t)[i+1]-i
    back = _next_true(s)[i+grace+1]-i
    candidates = np.flatnonzero(s[i] & (arrive <= horizon) & (arrive < back))
    first, tau, position = [], [], 0
    while (j := np.searchsorted(candidates, position)) < len(candidates):
        a = candidates[j]
        first.append(a)
        tau.append(arrive[a])
        position = a+arrive[a]+1
    first, tau = np.asarray(first, dtype=np.int64), np.asarray(tau, dtype=np.int64)
    frames = first[:, None]+np.minimum(np.arange(horizon+1)[None], tau[:, None])
    return {'paths': np.stack([phi[frames], psi[frames]], -1).astype(np.float32),
            'masks': (np.arange(horizon)[None] < tau[:, None]).astype(np.float32), 'tau': tau}


def drift_field(phis, psis, dt, bins=50, min_count=10):
    """Binned mean angular velocity and effective diffusion from within-replica increments."""
    x = np.concatenate([p[:-1] for p in phis])
    y = np.concatenate([p[:-1] for p in psis])
    vx = np.concatenate([wrap_np(p[1:]-p[:-1])/dt for p in phis])
    vy = np.concatenate([wrap_np(p[1:]-p[:-1])/dt for p in psis])
    edges = np.linspace(-np.pi, np.pi, bins+1)
    i = np.minimum(np.searchsorted(edges[1:], x), bins-1)
    j = np.minimum(np.searchsorted(edges[1:], y), bins-1)
    counts = np.zeros((bins, bins))
    np.add.at(counts, (i, j), 1.0)
    valid = counts >= float(min_count)
    drift, var = [], []
    for v in (vx, vy):
        b = np.zeros_like(counts)
        np.add.at(b, (i, j), v)
        b[valid] /= counts[valid]
        b[~valid] = 0.0
        s = np.zeros_like(counts)
        np.add.at(s, (i, j), (v-b[i, j])**2)
        s[valid] /= counts[valid]
        drift.append(b)
        var.append(s)
    d = [float(np.nanmedian(s[valid])*dt*0.5) for s in var]
    return {'drift_phi': drift[0], 'drift_psi': drift[1], 'counts': counts, 'centers': 0.5*(edges[:-1]+edges[1:]),
            'edges': edges, 'var_phi': var[0], 'var_psi': var[1], 'D_eff_phi': np.array([d[0]]),
            'D_eff_psi': np.array([d[1]]), 'D_eff': np.array([0.5*(d[0]+d[1])])}


def replicas(raw_dir, temperature, count):
    """Completed replica files, in index order."""
    files = sorted(Path(raw_dir).glob(f'alanine_T{temperature}_rep[0-9][0-9][0-9].npz'))
    if len(files) != count:
        raise RuntimeError(f'expected {count} completed replicas in {raw_dir}, found {len(files)}')
    parts = []
    for f in files:
        with np.load(f) as d:
            if int(d['is_complete'][0]) != 1 or float(d['temperature_k'][0]) != float(temperature):
                raise RuntimeError(f'incomplete or mismatched replica {f}')
            parts.append({'path': f, 'phi': d['phi'].astype(np.float64), 'psi': d['psi'].astype(np.float64),
                          'theta': d['theta'].astype(np.float64), 'dt_ps': float(d['dt_ps'][0])})
    if len({p['dt_ps'] for p in parts}) != 1:
        raise RuntimeError('replica frame spacing differs')
    return parts


def prepare(raw_dir, out_dir, temperature, cfg):
    """Replica-disjoint reactive splits (both directions) and the training-replica drift field."""
    splits, horizon, grace = cfg['data']['splits'], cfg['event']['horizon'], cfg['event']['grace']
    parts = replicas(raw_dir, temperature, sum(len(v) for v in splits.values()))
    train = [parts[i] for i in splits['train']]
    dt = train[0]['dt_ps']
    field = drift_field([p['phi'] for p in train], [p['psi'] for p in train], dt, cfg['data']['drift_bins'], cfg['data']['min_count'])
    diffusion = float(field['D_eff'][0])
    if not np.isfinite(diffusion) or diffusion <= 0:
        raise RuntimeError('invalid training-only diffusion estimate')
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out/'drift_field_train.npz', **field)
    records = {}
    for name, index in splits.items():
        arrays, counts = {}, []
        for tag, a, b in (('ab', BASIN_A, BASIN_B), ('ba', BASIN_B, BASIN_A)):
            found = [extract(parts[i]['phi'], parts[i]['psi'], a, b, horizon, grace) for i in index]
            arrays |= {f'paths_{tag}': np.concatenate([f['paths'] for f in found]),
                       f'masks_{tag}': np.concatenate([f['masks'] for f in found]),
                       f'tau_{tag}': np.concatenate([f['tau'] for f in found])}
            counts.append([len(f['tau']) for f in found])
        path = out/f'reactive_{name}.npz'
        np.savez_compressed(path, **arrays, dt_ps=np.array([dt]), temperature_k=np.array([float(temperature)]),
                            D_eff=np.array([diffusion]))
        records[name] = {'path': str(path.resolve()), 'sha256': sha256(path), 'replicas': index,
                         'A_to_B_paths': len(arrays['tau_ab']), 'B_to_A_paths': len(arrays['tau_ba']),
                         'per_replica_counts': [{'A_to_B': a, 'B_to_A': b} for a, b in zip(*counts)]}
    write_json(out/'manifest.json', {
        'temperature_k': temperature, 'horizon_transitions': horizon, 'grace_steps': grace, 'dt_ps': dt, 'D_eff_train_only': diffusion,
        'split_policy': 'replica-disjoint', 'replica_boundary_steps_included': False,
        'sources': [{'replica_index': k, 'path': str(p['path'].resolve()), 'sha256': sha256(p['path']), 'frames': int(p['phi'].size),
                     'cis_peptide_fraction': float((abs(wrap_np(p['theta'])) < np.pi/2).mean())} for k, p in enumerate(parts)],
        'drift_field': {'path': str((out/'drift_field_train.npz').resolve()), 'sha256': sha256(out/'drift_field_train.npz')},
        'splits': records})


def load_split(root, split, device='cpu'):
    """Arrays of reactive_{split}.npz as tensors."""
    with np.load(Path(root)/f'reactive_{split}.npz') as d:
        return {k: torch.tensor(d[k], device=device) for k in d.files}


def load_field(root):
    """The training-replica drift field."""
    with np.load(Path(root)/'drift_field_train.npz') as d:
        return {k: d[k] for k in d.files}


def observations(paths, masks, mode):
    """Endpoints (plus frames nearest 1/4, 1/2, 3/4 of the duration for completion); padding counts as observed."""
    tau = masks.sum(-1).long()
    obs = torch.zeros(paths.shape[:2], device=paths.device, dtype=torch.bool)
    rows = torch.arange(len(paths), device=paths.device)
    obs[:, 0] = True
    obs[rows, tau] = True
    if mode == 'completion':
        for f in (.25, .5, .75):
            obs[rows, (tau.float()*f).round().long()] = True
    return obs | (torch.arange(paths.shape[1], device=paths.device)[None] >= tau[:, None])


def interpolation(paths, obs):
    """Shortest-arc linear interpolation between consecutive observed frames."""
    result = torch.zeros_like(paths)
    for i in range(len(paths)):
        anchors = torch.where(obs[i])[0].tolist()
        for a, b in zip(anchors[:-1], anchors[1:]):
            f = torch.arange(b-a+1, device=paths.device, dtype=paths.dtype)[:, None]/(b-a)
            result[i, a:b+1] = wrap(paths[i, a]+f*delta(paths[i, a], paths[i, b]))
        result[i, obs[i]] = paths[i, obs[i]]
    return result

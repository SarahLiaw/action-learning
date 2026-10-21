"""Frozen-model inference: observation tasks, the angular bound, a basin preference and composed requests."""
from __future__ import annotations

import math

import numpy as np
import torch

from .data import delta, interpolation, observations, wrap, wrap_np
from .generators import generate
from .models import action
from .reference import event_valid

BOUND = math.radians(150)
STEP_LIMIT = math.pi-1e-3
TASKS = ('completion', 'completion_fence', 'endpoints', 'endpoints_fence')


def compatible(paths, obs, bounded):
    """Contexts whose supplied frames satisfy |phi| <= 150 deg and, between fixed neighbours, the step limit."""
    if not bounded:
        return torch.ones(len(paths), dtype=torch.bool, device=paths.device)
    ok = (~obs | (paths[..., 0].abs() <= BOUND+1e-6)).all(-1)
    pair = obs[:, 1:] & obs[:, :-1]
    return ok & (~pair | (paths[:, 1:, 0]-paths[:, :-1, 0]).abs().le(STEP_LIMIT)).all(-1)


def continuous_valid(paths, masks):
    """|phi| <= 150 deg at active frames and chart steps below pi: every shortest arc obeys the bound."""
    active = torch.cat([torch.ones_like(masks[:, :1]), masks], -1).bool()
    bounded = (~active | paths[..., 0].abs().le(BOUND+2e-6)).all(-1)
    steps = (~masks.bool() | (paths[:, 1:, 0]-paths[:, :-1, 0]).abs().le(STEP_LIMIT+2e-6)).all(-1)
    return bounded & steps


def project_phi(phi, observed, obs, iterations=24):
    """Alternating projections onto the phi interval, the fixed frames and the step-limit slabs."""
    x = torch.where(obs, observed, phi.clamp(-BOUND, BOUND)).clone()
    for _ in range(iterations):
        for parity in (0, 1):
            left = torch.arange(parity, x.shape[1]-1, 2, device=x.device)
            right = left+1
            diff = x[:, right]-x[:, left]
            excess = diff-diff.clamp(-STEP_LIMIT, STEP_LIMIT)
            free_l, free_r = (~obs[:, left]).to(x.dtype), (~obs[:, right]).to(x.dtype)
            denominator = (free_l+free_r).clamp_min(1)
            x[:, left] += excess*free_l/denominator
            x[:, right] -= excess*free_r/denominator
        x = torch.where(obs, observed, x.clamp(-BOUND, BOUND))
    return x


def _to_edges(p, basin, inside, bounded):
    """Move flagged frames to the nearest (bound-admissible) point just outside a basin rectangle."""
    c = p[..., None, :].expand(*p.shape[:2], 4, 2).clone()
    c[..., 0, 0] = basin['phi_min']-1e-4
    c[..., 1, 0] = basin['phi_max']+1e-4
    c[..., 2, 1] = basin['psi_min']-1e-4
    c[..., 3, 1] = basin['psi_max']+1e-4
    c = wrap(c)
    distance = delta(p[..., None, :], c).square().sum(-1)
    if bounded:
        phi = c[..., 0]
        good = phi.abs().le(BOUND+1e-7)
        good[:, 1:] &= (phi[:, 1:]-p[:, :-1, 0, None]).abs().le(STEP_LIMIT+1e-7)
        good[:, :-1] &= (phi[:, :-1]-p[:, 1:, 0, None]).abs().le(STEP_LIMIT+1e-7)
        distance = torch.where(good, distance, torch.inf)
    choice = distance.argmin(-1)
    moved = c.gather(-2, choice[..., None, None].expand(*choice.shape, 1, 2)).squeeze(-2)
    return torch.where(inside[..., None], moved, p)


def project(paths, observed, obs, masks, ref, bounded):
    """Task domain: observed frames fixed, the absorbed event at saved frames, optionally the angular bound."""
    k = torch.arange(paths.shape[1], device=paths.device)[None]
    tau = masks.sum(-1).long()[:, None]
    if bounded:
        p = torch.stack([project_phi(paths[..., 0], observed[..., 0], obs), wrap(paths[..., 1])], -1)
    else:
        p = wrap(paths)
    for basin, eligible in ((ref.basin_a, (k > ref.grace) & (k < tau)), (ref.basin_b, (k > 0) & (k < tau))):
        # Alternating parity keeps bounded moves from altering both ends of one transition.
        for parity in ((0, 1) if bounded else (None,)):
            inside = ref.in_basin(p.reshape(-1, 2), basin).reshape(p.shape[:2]) & eligible & ~obs
            if parity is not None:
                inside &= k % 2 == parity
            p = _to_edges(p, basin, inside, bounded)
    p = torch.where(obs[..., None], observed, p)
    terminal = observed[torch.arange(len(p), device=p.device), tau[:, 0]]
    return torch.where((k > tau)[..., None], terminal[:, None], p)


def bounded_interpolation(paths, obs):
    """Interpolation with phi linear inside the permitted chart and psi periodic."""
    result = interpolation(paths, obs)
    for i in range(len(paths)):
        anchors = torch.where(obs[i])[0].tolist()
        for a, b in zip(anchors[:-1], anchors[1:]):
            f = torch.arange(b-a+1, device=paths.device, dtype=paths.dtype)/(b-a)
            result[i, a:b+1, 0] = paths[i, a, 0]+f*(paths[i, b, 0]-paths[i, a, 0])
    return result


def descend(cost, start, project, frozen, steps, lr, clip=None, through=False, final=False):
    """Projected Adam on path coordinates; returns each row's lowest-cost iterate and its cost.

    through: evaluate the cost at project(z) (gradients pass the projection); frozen: coordinates with zero gradient;
    final: also evaluate after the last update.
    """
    z = start.detach().clone().requires_grad_()
    opt = torch.optim.Adam([z], lr=lr)
    best = torch.full((len(z),), float('inf'), device=z.device)
    best_path = z.detach().clone()
    for step in range(steps+final):
        p = project(z) if through else z
        c = cost(p)
        assert torch.isfinite(c).all()
        with torch.no_grad():
            better = c < best
            best_path[better], best[better] = p[better], c[better]
        if step == steps:
            break
        opt.zero_grad()
        c.sum().backward()
        if frozen is not None:
            z.grad.masked_fill_(frozen, 0.)
        if clip is not None:
            torch.nn.utils.clip_grad_norm_([z], clip)
        opt.step()
        with torch.no_grad():
            z.copy_(project(z))
    return best_path.detach(), best.detach()


def starts(lin, observed, count, noise):
    """`count` starts per context: the interpolant, then noisy copies (common across methods)."""
    initial = lin.repeat_interleave(count, 0)+noise*torch.randn_like(observed)
    initial[::count] = lin
    return initial


def reuse(actions, generators, test, ref, cfg, seed):
    """Yield, per task, every method's selected path on the eligible held-out contexts.

    Actions: projected descent from common starts, lowest own cost. Generators: `candidates` draws, same projection,
    lowest reference action (first draws kept as a control).
    """
    paths, masks = test['paths_ab'], test['masks_ab']
    for task in TASKS:
        bounded = task.endswith('_fence')
        allobs = observations(paths, masks, task.split('_')[0])
        keep = compatible(paths, allobs, bounded)
        p, m, obs = paths[keep], masks[keep], allobs[keep]
        rr, n = cfg['candidates'], int(keep.sum())
        observed, ob, mm = p.repeat_interleave(rr, 0), obs.repeat_interleave(rr, 0), m.repeat_interleave(rr, 0)
        lin = (bounded_interpolation if bounded else interpolation)(p, obs)
        torch.manual_seed(seed+cfg['start_seed_offset'])
        initial = starts(lin, observed, rr, cfg['noise'])
        domain = lambda z: project(z, observed, ob, mm, ref, bounded)
        out = {'interpolation': project(lin, p, obs, m, ref, bounded)}
        for name, model in {'reference': None, **actions}.items():
            best, value = descend(lambda z: action(model, z, mm, ref), domain(initial), domain,
                                  ob[..., None] if bounded else None, cfg['steps'], cfg['lr'], cfg['clip'], through=not bounded)
            assert event_valid(best, mm, ref).all() and (not bounded or continuous_valid(best, mm).all())
            out[name] = best.reshape(n, rr, *best.shape[1:])[torch.arange(n, device=best.device), value.reshape(n, rr).argmin(-1)]
        for name, model in generators.items():
            torch.manual_seed(seed+cfg['generator_seed_offset'])
            draw = domain(generate(model, name, observed, ob, mm, ref))
            assert event_valid(draw, mm, ref).all() and (not bounded or continuous_valid(draw, mm).all())
            with torch.no_grad():
                choice = (-ref.log_prob(draw, mm)).reshape(n, rr).argmin(-1)
            view = draw.reshape(n, rr, *draw.shape[1:])
            out[name], out[name+'_first_draw'] = view[torch.arange(n, device=draw.device), choice], view[:, 0]
        out['heldout_observed'] = p
        yield task, torch.where(keep)[0], obs, out


def preference(paths, masks, center, width):
    """M = mean over interior active frames of exp(sum_j (cos(x_j - c_j) - 1)/s^2); 0 for one transition."""
    n = masks.sum(-1).long()
    k = torch.arange(paths.shape[1], device=paths.device)[None]
    interior = (k > 0) & (k < n[:, None])
    density = torch.exp(((paths-paths.new_tensor(center)).cos()-1).sum(-1)/width**2)
    return (density*interior).sum(-1)/interior.sum(-1).clamp_min(1)


def preference_pools(actions, generators, test, ref, cfg, seed):
    """Endpoint-task candidate pools: each action optimizes S - lambda M at every weight; generators draw the pool."""
    p, m = test['paths_ab'], test['masks_ab']
    n, rr, weights = len(p), cfg['starts'], cfg['weights']
    size = rr*len(weights)
    center, width = [math.radians(c) for c in cfg['center_deg']], math.radians(cfg['width_deg'])
    obs = observations(p, m, 'endpoints')
    observed, ob, mm = p.repeat_interleave(rr, 0), obs.repeat_interleave(rr, 0), m.repeat_interleave(rr, 0)
    torch.manual_seed(seed+cfg['start_seed_offset'])
    initial = starts(interpolation(p, obs), observed, rr, cfg['noise'])
    domain = lambda z: project(z, observed, ob, mm, ref, False)
    pools = {}
    for name, model in {'reference': None, **actions}.items():
        found = []
        for w in weights:
            best, _ = descend(lambda z: action(model, z, mm, ref)-w*preference(z, mm, center, width), domain(initial),
                              domain, ob[..., None], cfg['steps'], cfg['lr'], cfg['clip'], final=True)
            assert event_valid(best, mm, ref).all()
            found.append(best.reshape(n, rr, *best.shape[1:]))
        pools[name] = torch.cat(found, dim=1)
    for name, model in generators.items():
        torch.manual_seed(seed+cfg['generator_seed_offset'])
        po, pb, pm = p.repeat_interleave(size, 0), obs.repeat_interleave(size, 0), m.repeat_interleave(size, 0)
        pieces = []
        for s in range(0, len(po), cfg['generator_chunk']):
            e = s+cfg['generator_chunk']
            v = project(generate(model, name, po[s:e], pb[s:e], pm[s:e], ref), po[s:e], pb[s:e], pm[s:e], ref, False)
            assert event_valid(v, pm[s:e], ref).all()
            pieces.append(v)
        pools[name] = torch.cat(pieces).reshape(n, size, *p.shape[1:])
    return obs, pools


def select_preference(pools, actions, test, ref, cfg):
    """Common-pool selection at every weight: actions by S - lambda M, generators by S_ref - lambda M."""
    m = test['masks_ab']
    n = len(m)
    center, width = [math.radians(c) for c in cfg['center_deg']], math.radians(cfg['width_deg'])
    chosen = {}
    for name, pool in pools.items():
        flat, masks = pool.flatten(0, 1), m.repeat_interleave(pool.shape[1], 0)
        with torch.no_grad():
            base = action(actions.get(name), flat, masks, ref).reshape(n, -1)
            reward = preference(flat, masks, center, width).reshape(n, -1)
        rows = torch.arange(n, device=m.device)
        for w in cfg['weights']:
            cost = base-w*reward
            choice = cost.argmin(-1)
            chosen[(name, w)] = {'paths': pool[rows, choice], 'choice': choice, 'preference': reward[rows, choice],
                                 'base_action': base[rows, choice], 'tilted_objective': cost[rows, choice]}
    return chosen


def route_domains(n, margin, first):
    """Convex boxes making each approach explicit: left of B, above B, or around its upper-right corner."""
    def base():
        return (np.full((n+1, 2), [-np.radians(150), -np.pi+margin]), np.full((n+1, 2), [np.radians(150), np.pi-margin]))
    names, low, high = [], [], []
    lo, hi = base()
    hi[1:n, 0], hi[first:, 1] = np.radians(30)-margin, -margin
    names.append('left_of_target'); low.append(lo); high.append(hi)
    lo, hi = base()
    lo[1:n, 1], lo[first:, 0] = margin, np.radians(-30)+margin
    names.append('above_target'); low.append(lo); high.append(hi)
    for pivot in range(first, n-1):
        lo, hi = base()
        lo[1:pivot+1, 1], lo[pivot:n, 0], lo[first:pivot, 0] = margin, np.radians(130)+margin, np.radians(-30)+margin
        names.append(f'right_of_target_pivot{pivot}'); low.append(lo); high.append(hi)
    return names, np.array(low), np.array(high)


def _crossings(a, b, lo, hi):
    """Parameter intervals where the shortest arc a -> b crosses periodic copies of an open rectangle."""
    v, hits = wrap_np(b-a), []
    for m in (-1, 0, 1):
        for n in (-1, 0, 1):
            lower, upper = np.asarray(lo)+2*np.pi*np.array([m, n]), np.asarray(hi)+2*np.pi*np.array([m, n])
            enter, leave, valid = 0., 1., True
            for j in range(2):
                if abs(v[j]) < 1e-12:
                    if not lower[j] < a[j] < upper[j]:
                        valid = False
                        break
                else:
                    q = sorted([(lower[j]-a[j])/v[j], (upper[j]-a[j])/v[j]])
                    enter, leave = max(enter, q[0]), min(leave, q[1])
            if valid and leave > enter+1e-10:
                hits.append([float(enter), float(leave)])
    return hits


def segment_audit(p, dt=0.1, first=6):
    """Exact checks on every shortest arc: no preterminal entry to B, no return to A from `first`, |phi| <= 150 deg."""
    A, B = np.radians([[-150, 0], [-30, 180]]), np.radians([[30, -180], [130, 0]])
    early, late, phi = [], [], []
    for k, (a, b) in enumerate(zip(p[:-1], p[1:])):
        h = _crossings(a, b, *B)
        if h and k < len(p)-2:
            early.append({'segment': k, 'fractions': h, 'time_ps': [[dt*(k+u), dt*(k+v)] for u, v in h]})
        h = _crossings(a, b, *A)
        if h and k >= first:
            late.append({'segment': k, 'fractions': h})
        start = float(wrap_np(a[0]))
        end = start+float(wrap_np(b[0]-a[0]))
        # An arc across the chart seam reaches |phi| = pi even when both knots obey the bound.
        phi.append(float(np.pi) if max(start, end) > np.pi or min(start, end) < -np.pi else max(abs(start), abs(end)))
    return {'early_target_intersections': early, 'late_initial_intersections': late,
            'max_abs_phi_deg': float(np.degrees(max(phi))), 'continuous_event_after_frame6': not early and not late,
            'phi_bound_feasible': bool(max(phi) <= np.radians(150)+1e-6)}


def _box(lo, hi, observed, fixed_rows=None, frame=None, value=None):
    def project_box(z):
        p = z.maximum(lo).minimum(hi)
        p[:, 0], p[:, -1] = observed[0], observed[-1]
        if fixed_rows is not None:
            p[fixed_rows, frame] = value
        return p
    return project_box


def route_candidates(model, fenced, ref, cfg):
    """Refine a bounded endpoint route: 3 starts (saved, linear, midway) in each route domain, exact segment audit."""
    n = len(fenced)-1
    observed = torch.tensor(fenced, dtype=torch.float32)
    names, low, high = route_domains(n, cfg['margin'], ref.grace+1)
    lo, hi = (torch.tensor(np.repeat(b, 3, axis=0), dtype=torch.float32) for b in (low, high))
    linear = torch.stack([torch.linspace(float(observed[0, j]), float(observed[-1, j]), n+1) for j in range(2)], -1)
    init = torch.stack([observed, linear, .5*(observed+linear)]).repeat(len(names), 1, 1)
    m = torch.ones((1, n)).expand(len(init), -1)
    frozen = torch.zeros((1, n+1, 1), dtype=torch.bool)
    frozen[:, [0, -1]] = True
    domain = _box(lo, hi, observed)
    best, cost = descend(lambda z: model(z, m), domain(init), domain, frozen, cfg['steps'], cfg['lr'])
    discrete = event_valid(best, m, ref).numpy()
    rows = []
    for i, (c, path) in enumerate(zip(cost.tolist(), best.numpy())):
        check = segment_audit(path, ref.dt, ref.grace+1)
        rows.append({'family': names[i//3], 'start': i % 3, 'action_nats': c, 'audit': check,
                     'eligible': bool(discrete[i] and check['continuous_event_after_frame6'] and check['phi_bound_feasible'])})
    valid = [i for i, r in enumerate(rows) if r['eligible']]
    assert valid, 'no continuously feasible route candidate'
    return best, rows, min(valid, key=lambda i: float(cost[i]))


def composed_requests(model, candidates, original, waypoint, frame, ref, cfg):
    """Bound + preference at each weight, and bound + an observed interior frame, from the route candidates."""
    n = len(original)-1
    center, width = [math.radians(c) for c in cfg['center_deg']], math.radians(cfg['width_deg'])
    names, low, high = route_domains(n, cfg['margin'], ref.grace+1)
    low, high = np.repeat(low, 3, axis=0), np.repeat(high, 3, axis=0)
    tasks, index = [], []
    for w in cfg['weights']:
        for k in range(len(candidates)):
            tasks.append({'request': 'bound_preference', 'weight': w, 'family': names[k//3], 'start': k % 3})
            index.append(k)
    for k in range(len(candidates)):
        if np.all(waypoint.numpy() >= low[k, frame]) and np.all(waypoint.numpy() <= high[k, frame]):
            tasks.append({'request': 'bound_waypoint', 'weight': 0, 'family': names[k//3], 'start': k % 3})
            index.append(k)
    lo, hi = (torch.tensor(b[index], dtype=torch.float32) for b in (low, high))
    weight = torch.tensor([t['weight'] for t in tasks], dtype=torch.float32)
    fixed = torch.tensor([i for i, t in enumerate(tasks) if t['request'] == 'bound_waypoint'], dtype=torch.long)
    observed = torch.tensor(original, dtype=torch.float32)
    m = torch.ones(len(tasks), n)
    frozen = torch.zeros((len(tasks), n+1, 1), dtype=torch.bool)
    frozen[:, [0, -1]] = True
    frozen[fixed, frame] = True
    domain = _box(lo, hi, observed, fixed, frame, waypoint)
    best, _ = descend(lambda z: model(z, m)-weight*preference(z, m, center, width),
                      domain(candidates[index]), domain, frozen, cfg['steps'], cfg['lr'], final=True)
    with torch.no_grad():
        costs, rewards = model(best, m).numpy(), preference(best, m, center, width).numpy()
        discrete = event_valid(best, m, ref).numpy()
    rows = []
    for i, (path, task) in enumerate(zip(best.numpy(), tasks)):
        check = segment_audit(path, ref.dt, ref.grace+1)
        error = float(abs(wrap_np(path[frame]-waypoint.numpy())).max()) if task['request'] == 'bound_waypoint' else None
        eligible = bool(discrete[i] and check['continuous_event_after_frame6'] and check['phi_bound_feasible'])
        rows.append({**task, 'action_nats': float(costs[i]), 'preference': float(rewards[i]), 'audit': check,
                     'eligible': eligible and (error is None or error < 1e-6), 'waypoint_error_rad': error})
    pool = [i for i, r in enumerate(rows) if r['request'] == 'bound_preference' and r['eligible']]
    selected = {f'bound_preference_{w:g}': min(pool, key=lambda i: float(costs[i]-w*rewards[i])) for w in cfg['weights']}
    through = [i for i, r in enumerate(rows) if r['request'] == 'bound_waypoint' and r['eligible']]
    assert through, 'no valid route through the observed frame'
    selected['bound_waypoint'] = min(through, key=lambda i: float(costs[i]))
    return best, rows, selected

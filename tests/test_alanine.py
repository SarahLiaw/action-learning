"""Self-contained checks of the alanine port: densities, event, projections, optimizers, training, CLI and MD."""
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from leap import read_json
from experiments.alanine.data import BASIN_A, BASIN_B, drift_field, extract, observations, wrap_np
from experiments.alanine.evaluate import summarize
from experiments.alanine.generators import TemporalNet, generate
from experiments.alanine.inference import (compatible, continuous_valid, descend, interpolation, preference, project,
                                           segment_audit)
from experiments.alanine.models import PathAction
from experiments.alanine.reference import Reference, active, bank, event_valid, wrapped_log_prob
from experiments.alanine.run import main
from experiments.alanine.train import train_actions, train_generators

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT/'configs/alanine'
CENTER, WIDTH = (math.radians(-60), math.radians(-40)), math.radians(30)


def field(seed=0, bins=50):
    g = np.random.default_rng(seed)
    edges = np.linspace(-np.pi, np.pi, bins+1)
    return {'edges': edges, 'drift_phi': g.normal(size=(bins, bins)), 'drift_psi': g.normal(size=(bins, bins))}


def reactive(n, seed=0, count=1):
    """Reference-sampled A->B paths standing in for data."""
    ref = Reference(field(seed), .1, .45)
    torch.manual_seed(seed)
    x0 = torch.tensor(np.radians([[-90., 90.]]), dtype=torch.float32).repeat(n, 1)+.1*torch.randn(n, 2)
    p, _ = ref.sample_successful(x0, 200)
    return ref, p, active(p)


def random_walk(n, seed):
    g = np.random.default_rng(seed)
    return wrap_np(np.cumsum(g.normal(scale=.3, size=(n, 2)), 0)+np.radians([-90, 90]))


def test_wrapped_density_is_normalized_with_exact_moments():
    grid = torch.linspace(-math.pi, math.pi, 20001, dtype=torch.double)
    for var in (.01, .1, 1., 9.):
        density = wrapped_log_prob(grid, torch.tensor(2.8, dtype=torch.double), var).exp()
        assert abs(float(torch.trapezoid(density, grid))-1) < 1e-10
        assert abs(float(torch.trapezoid(density*grid.cos(), grid))-math.cos(2.8)*math.exp(-var/2)) < 1e-7


def test_reference_density_uses_the_sampler_drift():
    ref = Reference(field(), .1, .5, steps=20)
    x = torch.randn(2, 21, 2, dtype=torch.double, requires_grad=True)
    direct = torch.stack([wrapped_log_prob(x[:, k+1], x[:, k]+ref.dt*ref.drift(x[:, k], k, 20), ref.sigma**2).sum(-1) for k in range(20)], 1)
    torch.testing.assert_close(direct, ref.step_log_prob(x))
    assert torch.autograd.gradcheck(lambda p: ref.log_prob(p).sum(), (x,), fast_mode=True)


def test_references_satisfy_the_extraction_event():
    ref, p, m = reactive(12)
    assert event_valid(p, m, ref).all()
    refs, rate = bank(p[:3], ref, 4, 1)
    assert refs['paths'].shape == (3, 4, 201, 2) and 0 < rate <= 1
    assert torch.equal(refs['paths'][:, :, 0], p[:3, None, 0].expand(-1, 4, -1))
    assert event_valid(refs['paths'].flatten(0, 1), refs['masks'].flatten(0, 1), ref).all()
    torch.testing.assert_close(refs['logq'], ref.log_prob(refs['paths'].flatten(0, 1), refs['masks'].flatten(0, 1)).reshape(3, 4))


def test_action_gradient_ignores_padding():
    model = PathAction(.5, .1, 600).double()
    x = torch.randn(2, 21, 2, dtype=torch.double, requires_grad=True)
    mask = torch.ones(2, 20, dtype=torch.double)
    mask[0, 10:] = 0
    assert torch.autograd.gradcheck(lambda p: model(p, mask).sum(), (x,), fast_mode=True)
    grad = torch.autograd.grad(model(x, mask).sum(), x)[0]
    assert torch.count_nonzero(grad[0, 11:]) == 0


def test_extraction_follows_the_first_arrival_rule():
    x = random_walk(6000, 1)
    found = extract(x[:, 0], x[:, 1], BASIN_A, BASIN_B)
    inside = lambda p, b: (b['phi_min'] <= p[..., 0] <= b['phi_max']) and (b['psi_min'] <= p[..., 1] <= b['psi_max'] or b is BASIN_A and p[..., 1] == -np.pi)
    starts, i = [], 0
    while i < len(x)-201:                            # the documented greedy scan, written out
        k = next((k for k in range(1, 201) if inside(x[i+k], BASIN_B) or (k > 5 and inside(x[i+k], BASIN_A))), None)
        if inside(x[i], BASIN_A) and k is not None and inside(x[i+k], BASIN_B):
            starts.append((i, k))
            i += k+1
        else:
            i += 1
    assert [int(t) for t in found['tau']] == [k for _, k in starts] and len(starts) > 5
    for (i, k), p, m in zip(starts, found['paths'], found['masks']):
        assert np.array_equal(p[:k+1], x[i:i+k+1].astype(np.float32)) and (p[k+1:] == p[k]).all() and m.sum() == k


def test_drift_field_recovers_a_constant_velocity():
    g = np.random.default_rng(2)
    phis = [wrap_np(np.cumsum(.05+g.normal(scale=.2, size=100000))) for _ in range(2)]
    psis = [wrap_np(np.cumsum(-.02+g.normal(scale=.2, size=100000))) for _ in range(2)]
    f = drift_field(phis, psis, .1)
    valid = f['counts'] >= 10
    assert f['counts'].sum() == 2*99999
    assert abs(np.average(f['drift_phi'][valid], weights=f['counts'][valid])-.5) < .02
    assert abs(np.average(f['drift_psi'][valid], weights=f['counts'][valid])+.2) < .02
    assert abs(float(f['D_eff'][0])/(.2**2/(2*.1))-1) < .03        # median of per-bin variances


@pytest.mark.parametrize('bounded', [False, True])
def test_projection_keeps_observations_event_and_bound(bounded):
    ref, p, m = reactive(10, seed=3)
    for mode in ('completion', 'endpoints'):
        obs = observations(p, m, mode)
        keep = compatible(p, obs, bounded)
        q, mm, ob = p[keep], m[keep], obs[keep]
        for z in (interpolation(q, ob), q+2*torch.randn_like(q)):
            fixed = project(z, q, ob, mm, ref, bounded)
            assert event_valid(fixed, mm, ref).all() and torch.equal(fixed[ob], q[ob])
            torch.testing.assert_close(project(fixed, q, ob, mm, ref, bounded), fixed)
            if bounded:
                assert continuous_valid(fixed, mm).all()


def test_descend_returns_each_rows_best_iterate():
    target = torch.tensor([[1., -2.], [.5, .5]])
    frozen = torch.zeros(2, 2, dtype=torch.bool)
    frozen[:, 0] = True
    best, cost = descend(lambda z: (z-target).square().sum(-1), torch.zeros(2, 2), lambda z: z.clamp(-1.5, 1.5), frozen, 300, .05)
    torch.testing.assert_close(best, torch.tensor([[0., -1.5], [0., .5]]), atol=1e-3, rtol=0)
    assert torch.allclose(cost, (best-target).square().sum(-1))


def test_preference_uses_interior_active_frames():
    p = torch.tensor([[[-1., 1.], list(CENTER), [.8, -.5], [.8, -.5]]], dtype=torch.float64)
    m = torch.tensor([[1., 1., 0.]], dtype=torch.float64)
    assert torch.allclose(preference(p, m, CENTER, WIDTH), torch.ones(1, dtype=p.dtype))
    moved = p.clone()
    moved[:, 0] += 1
    moved[:, 2:] += 2
    assert torch.equal(preference(p, m, CENTER, WIDTH), preference(moved, m, CENTER, WIDTH))
    assert torch.allclose(preference(p+2*math.pi, m, CENTER, WIDTH), preference(p, m, CENTER, WIDTH))
    assert preference(p, torch.tensor([[1., 0., 0.]], dtype=p.dtype), CENTER, WIDTH).item() == 0


def test_segment_audit_sees_crossings_between_frames():
    through_b = np.radians([[-90., 90.], [0., -60.], [80., -60.], [80., -90.]])
    check = segment_audit(through_b)
    assert check['early_target_intersections'] and not check['continuous_event_after_frame6'] and check['phi_bound_feasible']
    seam = np.radians([[-170., 30.], [170., 30.], [80., -90.]])
    assert segment_audit(seam)['max_abs_phi_deg'] == 180. and not segment_audit(seam)['phi_bound_feasible']
    direct = np.radians([[-90., 90.], [-60., 20.], [20., -20.], [60., -60.]])
    check = segment_audit(direct)
    assert check['continuous_event_after_frame6'] and check['phi_bound_feasible']


@pytest.mark.parametrize('method', ['ddpm', 'flow_matching'])
def test_generators_clamp_observed_frames(method):
    ref, p, m = reactive(3, seed=4)
    obs = observations(p, m, 'completion')
    torch.manual_seed(0)
    draw = generate(TemporalNet().eval(), method, p, obs, m, ref, steps=10)
    assert torch.equal(draw[obs], p[obs]) and draw.abs().max() <= math.pi


def test_training_writes_selected_checkpoints(tmp_path):
    ref, p, m = reactive(14, seed=5)
    split = lambda a, b: {'paths_ab': p[a:b], 'masks_ab': m[a:b], 'dt_ps': torch.tensor([.1], dtype=torch.float64),
                          'D_eff': torch.tensor([.45], dtype=torch.float64), 'temperature_k': torch.tensor([600.], dtype=torch.float64)}
    cfg = read_json(CONFIGS/'T600.json')
    train_actions(split(0, 10), split(10, 14), ref, cfg['actions'] | {'steps': 2, 'bank': 2, 'negatives': 2, 'validation_references': 2}, tmp_path, 1)
    train_generators(split(0, 10), split(10, 14), ref, cfg['generators'], tmp_path, 1, 2)
    train_generators(split(0, 10), split(10, 14), ref, cfg['generators'], tmp_path, 1, 1, tmp_path)
    for name in ('untrained', 'velocity', 'leap', 'binary_nre', 'ddpm', 'flow_matching'):
        assert (tmp_path/f'{name}.pt').exists()
    assert [r['step'] for r in read_json(tmp_path/'ddpm_history.json')] == [0, 1, 2]
    assert all(math.isfinite(r['validation']) for r in read_json(tmp_path/'leap_history.json'))


def test_configs_hold_the_paper_settings():
    t500, t600 = read_json(CONFIGS/'T500.json'), read_json(CONFIGS/'T600.json')
    for c in (t500, t600):
        a, g, i = c['actions'], c['generators'], c['inference']
        assert (a['steps'], a['batch'], a['lr'], a['weight_decay'], a['bank'], a['negatives'], a['residual_penalty']) == (1200, 32, 3e-4, 1e-5, 32, 8, .01)
        assert (g['steps'], g['batch'], g['lr']) == (3000, 64, 3e-4)
        assert (i['candidates'], i['steps'], i['lr'], i['noise']) == (8, 400, .035, .15)
        assert c['reference'] == {'gamma': 15.0, 'epsilon_factor': 2.0} and c['event'] == {'horizon': 200, 'grace': 5}
        assert c['data']['splits'] == {'train': [0, 1, 2, 3, 4, 5], 'validation': [6], 'test': [7]}
        assert (c['simulation']['timestep_fs'], c['simulation']['friction_ps'], c['simulation']['save_every']) == (1.0, 10.0, 100)
    assert (t500['generators']['continuation_steps'], t600['generators']['continuation_steps']) == (0, 9000)
    assert read_json(CONFIGS/'transfer.json')['references'] == 32
    assert read_json(CONFIGS/'preference.json')['weights'] == [0, 5, 20, 50]
    for f in CONFIGS.glob('*.json'):
        assert all(v.startswith('{out}') for v in read_json(f)['inputs'].values()), f


def write_replicas(folder, temperature, frames=1500):
    folder.mkdir(parents=True)
    for r in range(8):
        x = random_walk(frames, 10+r)
        np.savez_compressed(folder/f'alanine_T{temperature}_rep{r:03d}.npz', phi=x[:, 0], psi=x[:, 1], theta=np.full(frames, np.pi),
                            dt_ps=np.array([.1]), temperature_k=np.array([float(temperature)]), is_complete=np.array([1]))


def test_cli_pipeline_on_synthetic_replicas(tmp_path):
    """prepare -> train -> infer -> evaluate, then the preference and transfer studies, with tiny budgets."""
    write_replicas(tmp_path/'raw', 600)
    small = read_json(CONFIGS/'T600.json')
    small['actions'] |= {'steps': 2, 'bank': 2, 'negatives': 2, 'validation_references': 2}
    small['generators'] |= {'steps': 2, 'continuation_steps': 1}
    small['inference'] |= {'candidates': 2, 'steps': 2}
    small['seeds'] = [1]
    (tmp_path/'T600.json').write_text(json.dumps(small))
    base = ['--config', str(tmp_path/'T600.json'), '--out', str(tmp_path/'out'), '--device', 'cpu', '--threads', '1']
    main(['prepare', *base, '--input', f'raw={tmp_path}/raw'])
    for command in ('train', 'infer', 'evaluate'):
        main([command, *base])
    rows = read_json(tmp_path/'out/reuse/T600/summary.json')['rows']
    assert {r['task'] for r in rows} == {'completion', 'completion_fence', 'endpoints', 'endpoints_fence'}
    assert {r['method'] for r in rows} >= {'interpolation', 'reference', 'leap', 'binary_nre', 'velocity', 'untrained', 'ddpm', 'flow_matching'}
    assert all(r['metrics']['event_valid']['mean'] == 1 for r in rows if r['method'] != 'heldout_observed')
    pref = read_json(CONFIGS/'preference.json') | {'seeds': [1], 'weights': [0, 5], 'starts': 1, 'steps': 2}
    transfer = read_json(CONFIGS/'transfer.json') | {'seeds': [1], 'temperatures': [600], 'references': 2}
    for name, cfg in (('preference', pref), ('transfer', transfer)):
        (tmp_path/f'{name}.json').write_text(json.dumps(cfg))
    for command in ('infer', 'evaluate'):
        main([command, '--config', str(tmp_path/'preference.json'), *base[2:]])
    main(['evaluate', '--config', str(tmp_path/'transfer.json'), *base[2:]])
    assert len(read_json(tmp_path/'out/preference/T600/summary.json')['rows']) == 10
    auroc = read_json(tmp_path/'out/transfer/results.json')['rows']
    assert all(0 <= r['within_context_auroc']['mean'] <= 1 for r in auroc)


def test_summary_statistics():
    rows = [{'task': 'endpoints', 'method': 'leap', 'case_indices': [0, 1],
             'metrics': {'circular_mse_rad2': [v, 2*v], 'free_knots': [3, 0]}} for v in (1., 3.)]
    (row,) = summarize(rows)
    assert row['metrics']['mse'] == {'mean': 2., 'sd': math.sqrt(2.)} and row['metrics']['circular_mse_rad2']['mean'] == 3.


def test_simulation_resumes_exactly(tmp_path, monkeypatch):
    pytest.importorskip('openmmtools')
    from experiments.alanine import simulate as md
    cfg = read_json(CONFIGS/'T600.json')['simulation'] | {'total_steps': 400, 'checkpoint_every': 200, 'platform': 'Reference'}
    md.simulate(cfg, 600, 7, tmp_path/'whole.npz')
    write = md._write

    def interrupt(path, series, cfg, temperature, steps, complete):
        write(path, series, cfg, temperature, steps, complete)
        raise KeyboardInterrupt
    monkeypatch.setattr(md, '_write', interrupt)
    with pytest.raises(KeyboardInterrupt):
        md.simulate(cfg, 600, 7, tmp_path/'resumed.npz')
    monkeypatch.setattr(md, '_write', write)
    md.simulate(cfg, 600, 7, tmp_path/'resumed.npz')
    with np.load(tmp_path/'whole.npz') as a, np.load(tmp_path/'resumed.npz') as b:
        assert len(a['phi']) == 4 and int(b['is_complete'][0]) == 1
        for k in ('phi', 'psi', 'theta'):
            assert np.array_equal(a[k], b[k])


def test_realizer_reaches_target_torsions():
    pytest.importorskip('openmmtools')
    from experiments.alanine.audit import Realizer, realize_path
    realizer = Realizer(read_json(CONFIGS/'audit.json')['realizer'])
    path = np.radians([[-80., 70.], [-60., 20.]]).astype(np.float32)
    frames, xyz, summary = realize_path(realizer, path, realizer.pairs())
    assert xyz.shape == (3, 22, 3) and summary['valid'] == summary['configurations'] == 3
    assert summary['chirality_preserved'] == 3 and summary['cis_peptide_counts'] == [0, 0] and summary['severe_overlap_count'] == 0


def test_figure(tmp_path):
    from experiments.alanine import figure
    rng = np.random.default_rng(0)
    walk = lambda n, start: figure.wrap(start+np.cumsum(rng.normal(scale=.3, size=(n, 2)), 0))
    folder = tmp_path/'composed'
    folder.mkdir()
    np.savez(folder/'displayed_paths.npz', T600__completion__leap=walk(41, np.array([-1.4, 1.2])))
    keys = [f'preference_sweep_{w}' for w in (0, 5, 20, 50)]+['preference_only_20', 'bound_preference_20', 'original']
    np.savez(folder/'composed_paths.npz', **{k: walk(30, np.array([3.0, -3.0])) for k in keys})
    grid = np.linspace(-180, 180, 37)
    np.savez(folder/'landscape_T600.npz', degrees=grid, free_energy_kcal_mol=rng.uniform(0, 12, (37, 37)))
    figure.plot(tmp_path, tmp_path/'figure7.pdf')
    assert (tmp_path/'figure7.pdf').stat().st_size > 0


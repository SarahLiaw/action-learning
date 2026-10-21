"""Alanine dipeptide pipeline: python -m experiments.alanine.run <command> --config configs/alanine/<name>.json.

study reuse (T500.json, T600.json): simulate, prepare, train, infer, evaluate
study preference: infer, evaluate    study transfer: evaluate    study composed: infer, evaluate    study audit: audit
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from leap.utils import read_json, sha256, write_json
from . import audit as geometry
from .data import load_field, load_split, prepare
from .evaluate import composed_table, context_auroc, landscape, landscape_examples, metrics, spread, summarize, summarize_preference
from .generators import GENERATORS, load_generator
from .inference import TASKS, composed_requests, preference_pools, reuse, route_candidates, select_preference
from .models import ACTIONS, load_action
from .reference import cached_bank, event_valid, make_reference
from .simulate import simulate
from .train import train_actions, train_generators


class Run:
    """Configuration, output root and input locations ({out}, {temperature}, {seed} templates)."""

    def __init__(self, args):
        self.args, self.out, self.device = args, args.out, torch.device(args.device)
        self.cfg = read_json(args.config)
        for item in args.input:
            key, value = item.split('=', 1)
            self.cfg['inputs'][key] = value

    def input(self, key, **fields):
        return Path(self.cfg['inputs'][key].format(out=self.out, **fields))

    def seeds(self):
        return [self.args.seed] if self.args.seed is not None else self.cfg['seeds']

    def split(self, temperature, name, device=None):
        return load_split(self.input('data', temperature=temperature), name, device or self.device)

    def numpy_split(self, temperature, name):
        return {k: v.numpy() for k, v in self.split(temperature, name, 'cpu').items()}

    def references(self, temperature, split='train'):
        """A split, its guided reference, the physical (unguided) reference and the training occupancy counts."""
        data, field = self.split(temperature, split), load_field(self.input('data', temperature=temperature))
        return data, make_reference(data, field, self.cfg), make_reference(data, field, self.cfg, guidance=False), field['counts']

    def checkpoints(self, temperature, seed):
        """{method: {path, sha256}} from a checkpoint manifest; hashes are verified."""
        found = read_json(self.input('checkpoints', temperature=temperature, seed=seed))['checkpoints']
        for method, ck in found.items():
            if sha256(ck['path']) != ck['sha256']:
                raise RuntimeError(f'checkpoint changed: {method} {ck["path"]}')
        return found

    def models(self, temperature, seed, actions, generators, train):
        ck = self.checkpoints(temperature, seed)
        diffusion, dt = float(train['D_eff'][0]), float(train['dt_ps'][0])
        return ({m: load_action(ck[m]['path'], diffusion, dt, temperature, m, self.device) for m in actions},
                {m: load_generator(ck[m]['path'], self.device) for m in generators})

    def saved(self, key, temperature, seed):
        """(evaluation record, saved paths) of a reuse or preference run."""
        folder = self.input(key, temperature=temperature, seed=seed)
        with np.load(folder/('inferred_paths.npz' if key == 'reuse' else 'paths.npz')) as d:
            return read_json(folder/'evaluation.json'), dict(d)


def pick(saved, test, i, task, method='leap'):
    """Active frames of context i from a saved task output."""
    meta, pred = saved
    return pred[f'{task}__{method}'][meta['eligibility'][task]['included_indices'].index(i), :int(test['masks_ab'][i].sum())+1]


def cmd_simulate(run):
    cfg, t = run.cfg['simulation'], run.cfg['temperature']
    for r in ([run.args.replica] if run.args.replica is not None else range(cfg['replicas'])):
        path = run.out/'simulations'/f'T{t}'/f'alanine_T{t}_rep{r:03d}.npz'
        simulate(cfg, t, cfg['seed']+r, path)
        write_json(path.with_suffix('.json'), {**cfg, 'temperature_k': t, 'seed': cfg['seed']+r, 'sha256': sha256(path)})


def cmd_prepare(run):
    t = run.cfg['temperature']
    prepare(run.input('raw', temperature=t), run.input('data', temperature=t), t, run.cfg)


def cmd_train(run):
    cfg, t = run.cfg, run.cfg['temperature']
    train, ref, _, _ = run.references(t)
    val = run.split(t, 'validation')
    for seed in run.seeds():
        folder = run.out/'runs'/f'T{t}'/f'seed{seed}'
        folder.mkdir(parents=True, exist_ok=True)
        train_actions(train, val, ref, cfg['actions'], folder, seed)
        train_generators(train, val, ref, cfg['generators'], folder, seed, cfg['generators']['steps'])
        if cfg['generators']['continuation_steps']:
            train_generators(train, val, ref, cfg['generators'], folder, seed, cfg['generators']['continuation_steps'], folder)
        write_json(folder/'checkpoints.json', {'temperature': t, 'seed': seed, 'config': cfg, 'checkpoints': {
            m: {'path': str((folder/f'{m}.pt').resolve()), 'sha256': sha256(folder/f'{m}.pt')} for m in (*ACTIONS, *GENERATORS)}})


def cmd_infer_reuse(run):
    t = run.cfg['temperature']
    train, ref, physical, counts = run.references(t)
    test = run.split(t, 'test')
    for seed in run.seeds():
        actions, generators = run.models(t, seed, ACTIONS, GENERATORS, train)
        folder = run.out/'reuse'/f'T{t}'/f'seed{seed}'
        folder.mkdir(parents=True, exist_ok=True)
        rows, paths, eligibility = [], {}, {}
        for task, case, obs, outputs in reuse(actions, generators, test, ref, run.cfg['inference'], seed):
            p, m = test['paths_ab'][case], test['masks_ab'][case]
            eligibility[task] = {'included_indices': case.tolist(),
                                 'excluded_incompatible_observations': sorted(set(range(len(test['paths_ab'])))-set(case.tolist()))}
            for name, pred in outputs.items():
                met = metrics(pred, p, obs, m, ref, physical, counts)
                assert max(met['observation_max_error_rad']) < 1e-5
                rows.append({'task': task, 'method': name, 'case_indices': case.tolist(), 'metrics': met})
                paths[f'{task}__{name}'] = pred.cpu().numpy()
            np.savez_compressed(folder/'inferred_paths.npz', **paths)
        write_json(folder/'evaluation.json', {'seed': seed, 'config': run.cfg, 'checkpoints': run.checkpoints(t, seed),
                                               'eligibility': eligibility, 'rows': rows})


def cmd_evaluate_reuse(run):
    t = run.cfg['temperature']
    records = [row for seed in run.cfg['seeds'] for row in run.saved('reuse', t, seed)[0]['rows']]
    write_json(run.out/'reuse'/f'T{t}'/'summary.json', {'temperature': t, 'seeds': run.cfg['seeds'], 'rows': summarize(records),
                                                          'uncertainty': 'sample SD across fits on one held-out replica'})


def cmd_infer_preference(run):
    cfg, t = run.cfg, run.cfg['temperature']
    train, ref, physical, counts = run.references(t)
    test = run.split(t, 'test')
    for seed in run.seeds():
        actions, generators = run.models(t, seed, cfg['actions'], cfg['generators'], train)
        folder = run.out/'preference'/f'T{t}'/f'seed{seed}'
        folder.mkdir(parents=True, exist_ok=True)
        obs, pools = preference_pools(actions, generators, test, ref, cfg, seed)
        np.savez_compressed(folder/'candidate_pools.npz', **{k: v.cpu().numpy() for k, v in pools.items()})
        rows, paths = [], {}
        for (name, w), s in select_preference(pools, actions, test, ref, cfg).items():
            met = metrics(s['paths'], test['paths_ab'], obs, test['masks_ab'], ref, physical, counts)
            met |= {k: s[k].cpu().tolist() for k in ('preference', 'base_action', 'tilted_objective')}
            assert all(met['event_valid']) and max(met['observation_max_error_rad']) < 1e-5
            rows.append({'method': name, 'weight': w, 'case_indices': list(range(len(obs))), 'selection_indices': s['choice'].tolist(),
                         'metrics': met})
            paths[f'{name}__lambda{w:g}'] = s['paths'].cpu().numpy()
        for name in pools:
            m = np.array([r['metrics']['preference'] for r in rows if r['method'] == name])
            assert (np.diff(m, axis=0) > -1e-6).all(), 'common-pool selection cannot lower the preference'
        np.savez_compressed(folder/'paths.npz', **paths)
        write_json(folder/'evaluation.json', {'seed': seed, 'config': cfg, 'checkpoints': run.checkpoints(t, seed), 'rows': rows})


def cmd_evaluate_preference(run):
    t = run.cfg['temperature']
    records = [dict(run=seed, **row) for seed in run.cfg['seeds'] for row in run.saved('preference', t, seed)[0]['rows']]
    write_json(run.out/'preference'/f'T{t}'/'summary.json', {'temperature': t, 'seeds': run.cfg['seeds'],
                                                               'rows': summarize_preference(records, run.cfg['summary_methods'], run.cfg['weights'])})


def cmd_evaluate_transfer(run):
    cfg, K = run.cfg, run.cfg['references']
    rows, targets = [], []
    for target in cfg['temperatures']:
        folder = run.out/'transfer'/f'T{target}'
        folder.mkdir(parents=True, exist_ok=True)
        data, ref, _, _ = run.references(target, 'test')
        positive, pm = data['paths_ab'], data['masks_ab']
        refs = cached_bank(folder, data, ref, K, cfg['bank_seed']+target)
        negative, nm = refs['paths'].flatten(0, 1), refs['masks'].flatten(0, 1)
        with torch.no_grad():
            qp, qn = ref.log_prob(positive, pm), ref.log_prob(negative, nm).reshape(len(positive), K)
        assert event_valid(positive, pm, ref).all() and event_valid(negative, nm, ref).all()
        assert torch.equal(refs['paths'][:, :, 0], positive[:, None, 0].expand(-1, K, -1))
        assert torch.allclose(qn, refs['logq'], atol=1e-5, rtol=1e-6)
        targets.append({'temperature': target, 'n_data': len(positive), 'n_reference_per_context': K, 'diffusion': float(data['D_eff'][0])})
        for source in cfg['temperatures']:
            for method in cfg['methods']:
                values, used = [], []
                for seed in cfg['seeds']:
                    ck = run.checkpoints(source, seed)[method]
                    model = load_action(ck['path'], float(data['D_eff'][0]), float(data['dt_ps'][0]), target, method, run.device)
                    with torch.no_grad():
                        gp = -model(positive, pm)-qp
                        gn = torch.cat([-model(negative[i:i+cfg['chunk']], nm[i:i+cfg['chunk']])
                                        for i in range(0, len(negative), cfg['chunk'])]).reshape(len(positive), K)-qn
                    per = context_auroc(gp.cpu().numpy(), gn.cpu().numpy())
                    np.savez_compressed(folder/f'{method}_source{source}_seed{seed}.npz', positive_score=gp.cpu().numpy(),
                                        reference_score=gn.cpu().numpy(), per_context_auroc=per)
                    values.append(float(per.mean()))
                    used.append({'seed': seed, **ck})
                rows.append({'source': source, 'target': target, 'method': method, 'within_context_auroc': spread(values), 'checkpoints': used})
    write_json(run.out/'transfer'/'results.json', {'metric': 'mean within-context data/reference AUROC, tie-aware',
                                                   'score': '-S_theta - log q_ref (summed)', 'targets': targets, 'rows': rows})


def cmd_infer_composed(run):
    cfg, t, seed = run.cfg, run.cfg['temperature'], run.cfg['seed']
    folder = run.out/'composed'
    folder.mkdir(parents=True, exist_ok=True)
    examples, shown = {}, {}
    for temperature in cfg['temperatures']:
        test = run.numpy_split(temperature, 'test')
        fits = [run.saved('reuse', temperature, s) for s in cfg['seeds']]
        e = examples[temperature] = landscape_examples(test, fits, cfg['examples'], constraint=temperature == t)
        fixed, i = fits[cfg['seeds'].index(seed)], e['case']
        shown[f'T{temperature}__observed'] = test['paths_ab'][i, :int(test['masks_ab'][i].sum())+1]
        shown |= {f'T{temperature}__{task}__{m}': pick(fixed, test, i, task, m) for task in ('completion', 'endpoints') for m in ('leap', *GENERATORS)}
    test, e = run.numpy_split(t, 'test'), examples[t]
    fixed, tilt = run.saved('reuse', t, seed), run.saved('preference', t, seed)[1]
    j, i, n = e['constraint_case'], e['case'], lambda c: int(test['masks_ab'][c].sum())
    original, fenced = pick(fixed, test, j, 'endpoints'), pick(fixed, test, j, 'endpoints_fence')
    shown |= {f'T{t}__constraint_endpoints': original, f'T{t}__constraint_endpoints_fence': fenced}
    shown |= {f'T{t}__preference_lambda{w:g}': tilt[f'leap__lambda{w:g}'][i, :n(i)+1] for w in cfg['display']['landscape_weights']}
    np.savez_compressed(folder/'displayed_paths.npz', **shown)
    write_json(folder/'examples.json', {str(k): v for k, v in examples.items()})
    train, ref, _, _ = run.references(t)
    model = run.models(t, seed, ('leap',), (), train)[0]['leap'].cpu()
    candidates, route_rows, chosen = route_candidates(model, fenced, ref, cfg['route'])
    np.savez_compressed(folder/'route_paths.npz', original=original, archived_constrained=fenced,
                        checked_constrained=candidates[chosen].numpy(), all_candidates=candidates.numpy())
    write_json(folder/'route.json', {'case_index': j, 'selected_candidate': chosen, 'rows': route_rows})
    frame = round(n(j)*cfg['requests']['waypoint_fraction'])
    waypoint = torch.tensor(test['paths_ab'][j, frame], dtype=torch.float32)
    best, rows, selected = composed_requests(model, candidates, original, waypoint, frame, ref, cfg['requests'])
    np.savez_compressed(folder/'request_candidates.npz', paths=best.numpy())
    paths = {k: best[r].numpy() for k, r in selected.items()} | {'original': original}
    for family in ('left_of_target', 'above_target'):
        ids = [r for r, row in enumerate(route_rows) if row['family'] == family and row['eligible']]
        paths[family] = candidates[min(ids, key=lambda r: route_rows[r]['action_nats'])].numpy()
    d = cfg['display']
    paths[f'preference_only_{d["preference_only"]:g}'] = tilt[f'leap__lambda{d["preference_only"]:g}'][j, :n(j)+1]
    paths |= {f'preference_sweep_{w:g}': tilt[f'leap__lambda{w:g}'][i, :n(i)+1] for w in d['sweep_weights']}
    np.savez_compressed(folder/'composed_paths.npz', **paths)
    write_json(folder/'requests.json', {'case_index': j, 'transitions': n(j), 'waypoint': {'frame': frame, 'angles_rad': waypoint.tolist()},
                                         'selected': selected, 'rows': rows})


def cmd_evaluate_composed(run):
    cfg, folder = run.cfg, run.out/'composed'
    for temperature in cfg['temperatures']:
        grid, free = landscape(load_field(run.input('data', temperature=temperature))['counts'], temperature)
        np.savez_compressed(folder/f'landscape_T{temperature}.npz', degrees=grid, free_energy_kcal_mol=free)
    train, _, _, _ = run.references(cfg['temperature'])
    model = run.models(cfg['temperature'], cfg['seed'], ('leap',), (), train)[0]['leap'].cpu()
    geometry_rows = read_json(run.input('audit')/'composed'/'results.json')['rows']
    with np.load(folder/'composed_paths.npz') as d:
        rows = composed_table(dict(d), model, {r['path']: r['summary'] for r in geometry_rows}, cfg['requests'])
    write_json(folder/'table.json', {'rows': rows})


def reuse_paths(run, temperature):
    """Population diagnostics, the three example roles, and their observed and method paths (fixed fit)."""
    cfg, seed = run.cfg, run.cfg['seed']
    test, train = run.numpy_split(temperature, 'test'), run.numpy_split(temperature, 'train')
    fits = {s: run.saved('reuse', temperature, s) for s in cfg['seeds']}
    rows = geometry.population(test, train, load_field(run.input('data', temperature=temperature))['counts'], fits, TASKS, cfg['population_methods'])
    selected = geometry.select_examples(test, rows, fits[seed][0], seed, cfg['min_steps'])
    write_json(run.out/'audit'/f'selection_T{temperature}.json', {'selected': selected, 'population': rows})
    paths = {}
    for r in selected:
        i = r['case_index']
        paths[f'{r["role"]}__observed'] = test['paths_ab'][i, :int(test['masks_ab'][i].sum())+1]
        paths |= {f'{r["role"]}__{task}__{m}': pick(fits[seed], test, i, task, m) for task in cfg['role_tasks'][r['role']] for m in cfg['reuse_methods']}
    return paths


def preference_paths(run):
    """Observed, task and preference-sweep paths at the representative and constraint contexts (fixed fit)."""
    cfg, seed = run.cfg['preference'], run.cfg['seed']
    t = cfg['temperature']
    test = run.numpy_split(t, 'test')
    roles = {r['role']: r['case_index'] for r in read_json(run.out/'audit'/f'selection_T{t}.json')['selected']}
    fixed, tilt = run.saved('reuse', t, seed), run.saved('preference', t, seed)[1]
    paths = {}
    for role, tasks in cfg['tasks'].items():
        i = roles[role]
        paths[f'case{i}__observed'] = test['paths_ab'][i, :int(test['masks_ab'][i].sum())+1]
        paths |= {f'case{i}__{task}__leap': pick(fixed, test, i, task) for task in tasks}
    i = roles[cfg['role']]
    for method, weights in cfg['weights'].items():
        paths |= {f'case{i}__{method}__lambda{w:g}': tilt[f'{method}__lambda{w:g}'][i, :int(test['masks_ab'][i].sum())+1] for w in weights}
    return paths


def cmd_audit(run):
    cfg = run.cfg
    for name in run.args.sets or cfg['sets']:
        if name.startswith('reuse_T'):
            paths = reuse_paths(run, int(name[len('reuse_T'):]))
        elif name == 'preference':
            paths = preference_paths(run)
        else:
            file = {'landscape': 'displayed_paths.npz', 'route': 'route_paths.npz', 'composed': 'composed_paths.npz'}[name]
            with np.load(run.input('composed')/file) as d:
                paths = {k: d[k] for k in cfg['keys'][name]} if name in cfg['keys'] else dict(d)
        folder = run.out/'audit'/name
        folder.mkdir(parents=True, exist_ok=True)
        realizer = geometry.Realizer(cfg['realizer'])
        pairs, rows, xyz = realizer.pairs(), [], {}
        for key, path in paths.items():
            frames, xyz[key], summary = geometry.realize_path(realizer, path, pairs)
            rows.append({'path': key, 'frames': frames, 'summary': summary})
            print(name, key, json.dumps(summary), flush=True)
        np.savez_compressed(folder/'coordinates.npz', **xyz)
        write_json(folder/'results.json', {'system_sha256': realizer.system_sha, 'roundtrip_errors_rad': realizer.roundtrip,
                                           'realizer': cfg['realizer'], 'bonds': realizer.bonds, 'elements': realizer.elements,
                                           'aggregate': geometry.aggregate(rows), 'rows': rows})


HANDLERS = {('simulate', 'reuse'): cmd_simulate, ('prepare', 'reuse'): cmd_prepare, ('train', 'reuse'): cmd_train,
            ('infer', 'reuse'): cmd_infer_reuse, ('evaluate', 'reuse'): cmd_evaluate_reuse,
            ('infer', 'preference'): cmd_infer_preference, ('evaluate', 'preference'): cmd_evaluate_preference,
            ('evaluate', 'transfer'): cmd_evaluate_transfer, ('infer', 'composed'): cmd_infer_composed,
            ('evaluate', 'composed'): cmd_evaluate_composed, ('audit', 'audit'): cmd_audit}


def main(argv=None):
    ap = argparse.ArgumentParser(prog='python -m experiments.alanine.run', description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('command', choices=sorted({c for c, _ in HANDLERS}))
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--out', type=Path, default=Path('results/alanine'))
    ap.add_argument('--seed', type=int, help='one fit (default: every configured seed)')
    ap.add_argument('--replica', type=int, help='simulate one replica (default: all)')
    ap.add_argument('--set', dest='sets', action='append', help='audit path set (default: every configured set)')
    ap.add_argument('--input', action='append', default=[], metavar='KEY=TEMPLATE',
                    help='override an input location; templates may use {out}, {temperature}, {seed}')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--threads', type=int, default=4)
    args = ap.parse_args(argv)
    torch.set_num_threads(args.threads)
    run = Run(args)
    handler = HANDLERS.get((args.command, run.cfg['study']))
    if handler is None:
        ap.error(f'{args.command} does not apply to study {run.cfg["study"]!r}')
    start = time.time()
    handler(run)
    print(f'{args.command} {args.config}: {time.time()-start:.0f}s', flush=True)


if __name__ == '__main__':
    main()

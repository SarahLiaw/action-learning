"""Run one orbital config.

python -m experiments.orbital.run --config configs/orbital/<name>.json [--out results/orbital] [--seed S]

Config kinds: train (default), transfer, supervised, geotdm, bundle, calibrate, horizons.
Configs name files only; bundles resolve against --bundles and HORIZONS arrays against --horizons.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import os
from pathlib import Path
import platform
import statistics
import time

import torch

from leap import read_json, seed_everything, sha256, write_json

from . import data as D
from .evaluate import load_transferred, metrics, physics_for, predict
from .models import METHODS, build_model
from .supervised_features import fold_study, selection_study
from .train import TrainConfig, train_mle, train_ratio


def _environment(device: str) -> dict:
    gpu = torch.cuda.get_device_name(0) if device.startswith('cuda') and torch.cuda.is_available() else None
    return {'python': platform.python_version(), 'torch': torch.__version__, 'device': device, 'gpu': gpu}


def run_train(cfg: dict, args, seed: int) -> dict:
    bundle_path = args.bundles/cfg['bundle']
    bundle = D.load_bundle(bundle_path)
    method = cfg['method']
    objective = METHODS[method][1]
    override = dict(cfg.get('overrides', {}).get(str(seed), {}))
    eval_batch = override.pop('eval_batch_size', cfg['eval_batch_size'])
    tc = replace(TrainConfig(**cfg['train']), **override, seed=seed, device=args.device)
    run = args.out/cfg['name']/f'seed{seed}'
    seed_everything(seed)
    model = build_model(method, physics_for(bundle), **cfg['model'])
    data, reference = D.laws(bundle['config'], float(bundle['epsilon']))
    t0 = time.time()
    if objective == 'mle':
        history = train_mle(model, bundle, data, tc, run)
    else:
        history = train_ratio(model, objective, bundle, data, reference, tc, run)
    pred = predict(model, bundle, device=args.device, batch_size=eval_batch)
    torch.save(pred, run/'predictions.pt')
    val_key = 'validation_nll' if objective == 'mle' else 'validation_loss'
    result = {
        'config': cfg['name'], 'paper': cfg.get('paper'), 'method': method, 'objective': objective, 'seed': seed,
        'train': asdict(tc), 'global_l2_applied': tc.global_l2 if getattr(model, 'has_path', False) else 0.0,
        'eval_batch_size': eval_batch, 'parameters': sum(p.numel() for p in model.parameters()),
        'epochs_completed': len(history), 'best_validation': min(r[val_key] for r in history),
        'bundle': {'file': cfg['bundle'], 'sha256': sha256(bundle_path)},
        'metrics': metrics(pred, bundle), 'runtime_seconds': time.time()-t0, 'environment': _environment(args.device),
    }
    write_json(run/'result.json', result)
    print(f"{cfg['name']} seed {seed}: NRMSE_dG = {result['metrics']['nrmse']:.4f}", flush=True)
    return result


def run_transfer(cfg: dict, args) -> dict:
    """Fixed source weights scored on each target with the target's nominal physics."""
    source = read_json(args.config.parent/f"{cfg['source']}.json")
    checkpoints = [args.out/source['name']/f'seed{s}'/'best_model.pt' for s in source['seeds']]
    rows = {}
    for target, name in cfg['targets'].items():
        bundle = D.load_bundle(args.bundles/name)
        nrmse = [metrics(predict(load_transferred(ck, bundle, cfg['method'], **cfg['model']), bundle,
                                 device=args.device, batch_size=cfg['eval_batch_size']), bundle)['nrmse']
                 for ck in checkpoints]
        rows[target] = {'nrmse': nrmse, 'mean': statistics.mean(nrmse), 'sd': statistics.stdev(nrmse)}
        print(f"{cfg['name']} {target}: {rows[target]['mean']:.4f} +- {rows[target]['sd']:.4f}", flush=True)
    result = {'config': cfg['name'], 'paper': cfg.get('paper'), 'source': cfg['source'], 'targets': rows}
    write_json(args.out/cfg['name']/'result.json', result)
    return result


def run_supervised(cfg: dict, args) -> dict:
    folds = [fold_study(D.load_bundle(args.bundles/n), folds=cfg['folds'], seed=cfg['seed'], epochs=cfg['epochs'])
             for n in cfg['bundles']]
    selection = selection_study(D.load_bundle(args.bundles/cfg['selection_bundle']), penalty=cfg['penalty'],
                                epochs=cfg['epochs'])
    result = {'config': cfg['name'], 'paper': cfg.get('paper'), 'folds': folds, 'selection': selection}
    write_json(args.out/cfg['name']/'result.json', result)
    return result


def run_bundle(cfg: dict, args) -> Path:
    path = args.bundles/cfg['file']
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(D.build_bundle(args.horizons/cfg['horizons'], float(cfg['eta']), D.BundleConfig(**cfg['bundle'])), path)
    print(f'wrote {path}', flush=True)
    return path


def run_calibrate(cfg: dict, args) -> dict:
    """eta per body count: canonical step-distance rule or matched oracle AUROC."""
    results = []
    for count in cfg['body_counts']:
        names = D.bodies(count)
        m = D.masses_kg(names)
        states = D.load_states(args.horizons/cfg['horizons'], names)[slice(*cfg['contexts'])]
        if cfg['rule'] == 'step_distance':
            eta, distance = D.calibrate_step_distance(torch.as_tensor(states, dtype=torch.float64), m,
                                                      target=cfg['target'])
            results.append({'body_count': count, 'eta': eta, 'median_step_distance': distance})
        else:
            ctx = torch.as_tensor(D.to_barycentric(states, m), dtype=torch.float64)
            results.append({'body_count': count, **D.calibrate_auroc(
                ctx, m, target=cfg['target'], steps=cfg['num_steps'], seed=cfg['seed']+count, device=args.device)})
        print(results[-1], flush=True)
    result = {'config': cfg['name'], 'results': results}
    write_json(args.out/'calibration'/f"{cfg['name']}.json", result)
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--out', type=Path, default=Path('results/orbital'))
    ap.add_argument('--bundles', type=Path, default=Path(os.environ.get('LEAP_ORBITAL_BUNDLES', 'data/orbital/bundles')))
    ap.add_argument('--horizons', type=Path, default=Path(os.environ.get('LEAP_ORBITAL_HORIZONS', 'data/orbital/horizons')))
    ap.add_argument('--seed', type=int, default=None, help='run one seed of a multi-seed config')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()
    cfg = read_json(args.config)
    kind = cfg.get('kind', 'train')
    seeds = [args.seed] if args.seed is not None else cfg.get('seeds', [])
    if kind == 'train':
        for seed in seeds:
            run_train(cfg, args, seed)
    elif kind == 'geotdm':
        from .geotdm.run import run_geotdm
        for seed in seeds:
            run_geotdm(cfg, args, seed)
    elif kind == 'transfer':
        run_transfer(cfg, args)
    elif kind == 'supervised':
        run_supervised(cfg, args)
    elif kind == 'bundle':
        run_bundle(cfg, args)
    elif kind == 'calibrate':
        run_calibrate(cfg, args)
    elif kind == 'horizons':
        args.horizons.mkdir(parents=True, exist_ok=True)
        print('wrote', D.assemble_horizons(args.horizons, extended=cfg['extended']), flush=True)
    else:
        raise ValueError(f'unknown config kind {kind!r}')


if __name__ == '__main__':
    main()

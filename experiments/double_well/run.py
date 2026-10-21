"""Double-well experiment: python -m experiments.double_well.run --config CONFIG --out DIR [--seed S]."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch

from leap import read_json, seed_everything, sha256, write_json

from .baselines import fit_cubic, kramers_moyal, polynomial, transitions
from .evaluate import field_mse, geometric, infer, model_drift_mse, summarize
from .inference import analytic_action, drift_action
from .model import ConservativePotential, untrained_potential
from .reference import TerminalWindowReference
from .simulate import simulate_selected
from .train import train

log = logging.getLogger(__name__)
CHECKPOINT = 'matched_reference.pt'     # the paper run's file name for the LEAP weights
ORDER = ['kramers_moyal', 'cubic', 'zero', 'untrained', 'leap', 'geometric', 'analytic']


def run_seed(cfg: dict, seed: int, out: Path, device, checkpoints: Path | None = None) -> dict:
    """Simulate, train, fit the controls and infer paths for one repetition."""
    system, arch, inf, grid = cfg['system'], cfg['model'], cfg['inference'], cfg['evaluation']
    diffusion, dt = system['diffusion'], system['dt']
    source = checkpoints/f'seed{seed}' if checkpoints else out
    out.mkdir(parents=True, exist_ok=True)
    seed_everything(seed)
    if checkpoints:
        data = torch.load(source/'training_paths.pt', weights_only=True)
    else:
        data = simulate_selected(cfg['simulation']['n_paths'], **system,
                                 round_size=cfg['simulation']['round_size'], device=device)
        torch.save(data.cpu(), out/'training_paths.pt')
    model = ConservativePotential(diffusion, dt, **arch).to(device)     # initialized after simulation
    losses = None
    if checkpoints:
        model.load_state_dict(torch.load(source/CHECKPOINT, map_location=device, weights_only=True))
    else:
        losses = train(model, data, TerminalWindowReference(**system, device=device), **cfg['training'])
        torch.save(model.state_dict(), out/CHECKPOINT)
        write_json(out/'training_loss.json', losses)
    model.eval().requires_grad_(False)
    untrained = untrained_potential(seed, diffusion=diffusion, dt=dt, **arch).to(device).eval().requires_grad_(False)

    x, v = transitions(data.cpu(), dt)
    beta = fit_cubic(x, v)
    km, counts = kramers_moyal(x, v, **grid)
    drift_mse = {'kramers_moyal': field_mse(km, **grid),
                 'cubic': field_mse(polynomial(beta, np.linspace(grid['low'], grid['high'], grid['points'])), **grid),
                 'zero': field_mse(np.zeros(grid['points']), **grid),
                 'untrained': model_drift_mse(untrained, **grid, device=device),
                 'leap': model_drift_mse(model, **grid, device=device)}
    log.info('seed %d drift MSE %s', seed, drift_mse)

    coef = torch.tensor(beta, dtype=torch.float32)
    actions = {'cubic': drift_action(lambda q: polynomial(coef, q), dt, diffusion),
               'zero': drift_action(torch.zeros_like, dt, diffusion),
               'untrained': untrained.action, 'leap': model.action,
               'analytic': analytic_action(dt, diffusion)}
    solve = dict(diffusion=diffusion, dt=dt, endpoints=inf['endpoints'], horizons=inf['horizons'],
                 waits=inf['waits'], lbfgs=inf['lbfgs'])
    rows, paths = [], {}
    for method, action in actions.items():
        # the paper run solved the closed-form controls on the CPU
        found, chosen = infer(action, **solve, device='cpu' if method in ('cubic', 'zero') else device)
        rows += [{'method': method, **r} for r in found]
        paths.update({f'{method}_T{h:g}': p for h, p in chosen.items()})
        log.info('seed %d %s %s', seed, method, [round(r['true_scaled_cost'], 4) for r in found])
    found, chosen = geometric(dt=dt, endpoints=inf['endpoints'], horizons=inf['horizons'],
                              crossing=inf['geometric_crossing'])
    rows += [{'method': 'geometric', **r} for r in found]
    paths.update({f'geometric_T{h:g}': p for h, p in chosen.items()})
    np.savez_compressed(out/'paths.npz', **paths)

    record = {'seed': seed, 'device': str(device), 'torch': torch.__version__, 'retrained': checkpoints is None,
              'training_paths_sha256': sha256(source/'training_paths.pt'),
              'drift_mse': drift_mse, 'path_costs': rows, 'cubic_coefficients': beta.tolist(),
              'kramers_moyal_min_count': int(counts.min()), 'training_loss': losses}
    write_json(out/'results.json', record)
    return record


def table(summary: dict) -> str:
    """The paper's table as plain text: drift MSE and true path cost, mean +- sample SD."""
    def fmt(s):
        return f"{s['mean']:.6f} +- {s['sd']:.6f}"
    horizons = list(next(iter(summary['true_scaled_cost'].values())))
    lines = [f"{'method':<14}{'drift MSE':>24}"+''.join(f'{"T="+h:>24}' for h in horizons)]
    for m in ORDER:
        mse = fmt(summary['drift_mse'][m]) if m in summary['drift_mse'] else '-'
        cost = summary['true_scaled_cost'].get(m, {})
        lines.append(f'{m:<14}{mse:>24}'+''.join(f'{fmt(cost[h]) if h in cost else "-":>24}' for h in horizons))
    return '\n'.join(lines)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--seed', type=int, help='run only this seed of the config, e.g. one array task')
    ap.add_argument('--checkpoints', type=Path,
                    help=f'evaluate saved runs, DIR/seed<s>/{{training_paths.pt,{CHECKPOINT}}}, instead of training')
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(message)s')
    cfg = read_json(args.config)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    write_json(args.out/'config.json', cfg)
    for seed in cfg['seeds'] if args.seed is None else [args.seed]:
        log.info('seed %d on %s', seed, device)
        run_seed(cfg, seed, args.out/f'seed{seed}', device, args.checkpoints)
    done = [args.out/f'seed{s}'/'results.json' for s in cfg['seeds']]
    if all(p.exists() for p in done):
        summary = summarize([read_json(p) for p in done])
        write_json(args.out/'summary.json', summary)
        print(table(summary))


if __name__ == '__main__':
    main()

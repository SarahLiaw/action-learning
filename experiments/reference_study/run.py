"""Run reference-study sweeps.

python -m experiments.reference_study.run --config configs/reference_study/<sweep>.json --out results/reference_study
Each (setting, seed) writes <out>/<setting>/seed<seed>/{models.pt, record.json}; finished runs are skipped.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import torch

from leap import write_json

from . import dynamics, sweeps
from .evaluate import evaluate
from .reference import Reference
from .train import train_all


def run(setting: str, cfg: sweeps.Config, out: Path) -> dict:
    """Simulate, fit the three estimators and evaluate; saves the models before the record."""
    torch.set_num_threads(1)   # single-threaded CPU numerics, as in the paper runs
    data = dynamics.splits(cfg)
    ref = Reference.of(cfg)
    models, fits = train_all(cfg, data, ref)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({name: model.state_dict() for name, model in models.items()}, out/'models.pt')
    metrics, oracle_auroc = evaluate(models, cfg, data['test'], ref)
    record = {'setting': setting, 'config': asdict(cfg), 'exact_action': cfg.exact, 'metrics': metrics,
              'oracle_heldout_auroc': oracle_auroc, 'training': {k: asdict(f) for k, f in fits.items()}}
    write_json(out/'record.json.tmp', record)
    (out/'record.json.tmp').replace(out/'record.json')
    return record


def _headline(m: dict) -> str:
    """NRMSE when the exact action is known, else the held-out AUROC."""
    return f'NRMSE {m["correction"]["pairwise_delta_nrmse"]:.4f}' if m['correction'] else f'AUROC {m["heldout_auroc"]:.4f}'


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', type=Path, nargs='+', required=True, help='sweep JSON file(s)')
    parser.add_argument('--out', type=Path, default=Path('results/reference_study'))
    parser.add_argument('--task', type=int, help='run only this member (SLURM array index)')
    parser.add_argument('--list', action='store_true', help='print the members and exit')
    args = parser.parse_args(argv)
    members = sweeps.members(args.config)
    if args.list:
        for i, (setting, cfg) in enumerate(members):
            print(i, setting, cfg.seed)
        return
    for setting, cfg in members if args.task is None else [members[args.task]]:
        out = args.out/setting/f'seed{cfg.seed}'
        if (out/'record.json').exists():
            print(f'skip {out}: record exists', flush=True)
            continue
        metrics = run(setting, cfg, out)['metrics']
        print(f'{setting} seed {cfg.seed}: '+', '.join(f'{k} {_headline(m)}' for k, m in metrics.items()), flush=True)


if __name__ == '__main__':
    main()

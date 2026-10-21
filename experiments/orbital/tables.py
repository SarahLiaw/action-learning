"""Collect finished runs into the paper's orbital tables (mean +- sample SD over seeds).

python -m experiments.orbital.tables [--out results/orbital] [--bundles DIR]   -> prints and writes <out>/tables.json
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import statistics

import torch

from leap import read_json, write_json
from . import data as D
from .evidence import implied_eta, solar_evidence

_ENC = [('deepset', 'Deep Sets'), ('endpoints', 'Endpoints'), ('temporal_conv', 'Temporal convolution'),
        ('hybrid_conv', 'Hybrid convolution'), ('gru', 'GRU'), ('transformer', 'Transformer')]
_TRANSITION = [('LEAP-Local', 'nre_leap_local'), ('Transition NCE', 'nre_transition_nce'), ('LEAP-Global', 'nre_leap_global')]
_OBJECTIVE = [('Categorical NRE, Transformer + features', 'nre_categorical_summary'),
              ('Binary NRE, path encoder', 'nre_binary'), ('Binary NRE, Transformer + features', 'nre_binary_summary')]
TABLES = {
    'Table 2: trajectory information': _TRANSITION,
    'Table 3: accumulated path summaries': _OBJECTIVE,
    'Table 9: orbital controls': _TRANSITION + [('LEAP-Hybrid', 'nre_leap_hybrid'), ('Gaussian MLE', 'nre_gaussian_mle'),
                                                ('History Gaussian MLE', 'nre_history_gaussian_mle')] + _OBJECTIVE,
    'Table 10: comparison and data budgets': [('K=4, validation stop', 'k4_leap_global'), ('K=4, 275 epochs', 'k4_leap_global_275'),
                                              ('K=16', 'k16_leap_global'), ('K=32', 'k32_leap_global')]
                                             + [(f'{n} training paths, {label}', f'data_{c}_ppc{p}') for p, n in ((16, '4,064'), (32, '8,128'))
                                                for c, label in (('leap_global', 'LEAP-Global'), ('binary_summary', 'Binary NRE'))],
    'Table 11: scaling': [(f'{m} bodies, {n} transitions, {label}', f'scaling_m{m}_n{n}_{method}') for m in (31, 42)
                          for n in (32, 64, 128, 256) for method, label in (('leap_local', 'LEAP-Local'), ('leap_global', 'LEAP-Global'))],
    'Additional: encoders': [(f'{label}, lr 3e-4', f'encoders_{m}_lr3e-4') for m, label in _ENC]
                            + [(f'{label}, lr 1e-3', f'encoders_{m}_lr1e-3') for m, label in _ENC[2:3]+_ENC[4:]],
    'Additional: K and capacity': [(f'LEAP-Global K={k}', f'k{k}_leap_global') for k in (8, 64)]
                                  + [(f'Binary NRE + summary K={k}', f'k{k}_binary_summary') for k in (4, 16)]
                                  + [('LEAP-Global width 192', 'capacity_leap_global_w192'), ('LEAP-Global width 384', 'capacity_leap_global_w384'),
                                     ('Binary NRE + summary width 192', 'capacity_binary_summary_w192'),
                                     ('LEAP-Local width 192', 'capacity_leap_local_w192')],
    'Additional: objectives': [('Balanced NRE, path encoder', 'nre_balanced'), ('Endpoint Gaussian score', 'nre_endpoint_gaussian_mle')]
                              + [(f'Discovery K={k}', f'discovery_k{k}') for k in (4, 16)],
}


def _stats(values: list[float]) -> dict:
    return {'mean': statistics.mean(values), 'sd': statistics.stdev(values) if len(values) > 1 else 0.0, 'n': len(values)}


def _runs(out: Path, name: str) -> dict | None:
    files = sorted((out/name).glob('seed*/result.json'))
    if not files:
        return None
    runs = [read_json(f) for f in files]
    return {**_stats([r['metrics']['nrmse'] for r in runs]), 'seeds': [r['seed'] for r in runs]}


def recovered_eta(out: Path, bundles: Path) -> dict | None:
    """Section 4.2: the mass perturbation implied by each LEAP-Global fit, against the bundle's true eta."""
    files = sorted((out/'nre_leap_global').glob('seed*/predictions.pt'))
    if not files:
        return None
    bundle = D.load_bundle(bundles/read_json(files[0].parent/'result.json')['bundle']['file'])
    score, info = solar_evidence(bundle)
    eta, ids = float(bundle['epsilon']), bundle['test_context_ids'].numpy()
    fits = [implied_eta(torch.load(f).numpy(), score, info, ids, eta) for f in files]
    return {**_stats(fits), 'true_eta': eta}


def collect(out: Path, bundles: Path | None = None) -> dict:
    tables = {t: {label: _runs(out, name) for label, name in rows} for t, rows in TABLES.items()}
    if bundles is not None and (eta := recovered_eta(out, bundles)) is not None:
        tables['Section 4.2: recovered eta'] = {'LEAP-Global': eta}
    files = sorted((out/'diffusion_geotdm').glob('seed*/result.json'))
    if files:
        runs = [read_json(f) for f in files]
        tables['Table 12: trajectory diffusion baseline'] = {
            'denoising NRMSE_dG': _stats([r['simple']['nrmse'] for r in runs]),
            'denoising AUROC': _stats([r['simple']['auroc'] for r in runs]),
            'variational RelErr_aff': _stats([r['variational']['relerr_aff'] for r in runs]),
            'variational AUROC': _stats([r['variational']['auroc'] for r in runs]),
            'variational Spearman': _stats([r['variational']['spearman'] for r in runs]),
            'oracle AUROC': runs[0]['oracle_auroc']}
    for k in (4, 16):
        f = out/f'transfer_k{k}'/'result.json'
        if f.is_file():
            tables[f'Additional: transfer K={k}'] = {t: _stats(v['nrmse']) for t, v in read_json(f)['targets'].items()}
    f = out/'supervised_features'/'result.json'
    if f.is_file():
        r = read_json(f)
        tables['Additional: supervised features'] = {f"N={row['num_steps']}": {k: row[k] for k in (
            'additive_pooled', 'mlp_pooled', 'mlp_fold_mean', 'mlp_fold_sd', 'closed_form')} for row in r['folds']}
        tables['Additional: candidate selection'] = {k: v for k, v in r['selection'].items() if isinstance(v, dict)}
    return tables


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, default=Path('results/orbital'))
    ap.add_argument('--bundles', type=Path, default=Path(os.environ.get('LEAP_ORBITAL_BUNDLES', 'data/orbital/bundles')),
                    help='bundle folder, for the recovered eta')
    args = ap.parse_args()
    tables = collect(args.out, args.bundles if args.bundles.is_dir() else None)
    for table, rows in tables.items():
        print(f'\n{table}')
        for label, v in rows.items():
            if v is None:
                print(f'  {label:44s} (not run)')
            elif isinstance(v, dict) and 'mean' in v:
                print(f"  {label:44s} {v['mean']:.4g} +- {v['sd']:.2g}  (n={v['n']})")
            else:
                print(f'  {label:44s} {v}')
    write_json(args.out/'tables.json', tables)


if __name__ == '__main__':
    main()

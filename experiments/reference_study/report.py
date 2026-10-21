"""Aggregate run records into the paper's reference-study tables.

python -m experiments.reference_study.report --results results/reference_study --out results/reference_study/report
python -m experiments.reference_study.report --paper-records <dir of original resolved_results.json> --out <dir>
"""
from __future__ import annotations

import argparse
from dataclasses import fields
import math
from pathlib import Path
import statistics

from leap import read_json, write_json

from .sweeps import CONFIGS, Config, read, setting_name

METHODS = {'leap': 'LEAP', 'transition_nce': 'Transition NCE', 'gaussian_mle': 'Gaussian MLE'}
METRICS = ('nrmse', 'slope', 'auroc', 'within_auroc', 'regret', 'learned_diffusion')
HIGHER_IS_BETTER = {'auroc', 'within_auroc'}

# Paper tables: panels of (title, sweep, metric); untitled panels share one tabular.
TABLES = {
    'sensitivity_reference': [(None, 'reference_diffusion', 'nrmse'), (None, 'reference_drift', 'nrmse')],
    'sensitivity_observation': [(r'A. Grid resolution, NRMSE $\downarrow$', 'grid', 'nrmse'),
                                (r'B. Observation noise, AUROC $\uparrow$', 'observation_noise', 'auroc'),
                                (r'C. Hidden integration steps, AUROC $\uparrow$', 'substeps', 'auroc')],
    'sensitivity_budget': [(None, 'reference_count', 'nrmse'), (None, 'data_budget', 'nrmse')],
}
CAPTIONS = {   # Table 7; Table 8 as panel A; the robustness and budget numbers quoted in Appendix D
    'sensitivity_reference': (
        r'\textbf{Reference choice.} Correction NRMSE $\downarrow$. The diffusion sweep uses $b_{\rm ref}=0$; the '
        r'drift sweep uses $D_{\rm ref}=0.20$. Contrastive models use $K=4$.'),
    'sensitivity_observation': (
        r'\textbf{Temporal resolution and observation quality.} Panel A is Table 8, correction NRMSE $\downarrow$ at '
        r'fixed $T=0.96$. Panels B and C give the AUROC under observation noise and hidden integration substeps '
        r'quoted in Appendix D. Mean $\pm$ sample SD over three seeds.'),
    'sensitivity_budget': (
        r'\textbf{Reference and data budgets.} Correction NRMSE $\downarrow$ for the budgets quoted in Appendix D. '
        r'The reference count sweep uses $256$ training paths; the data sweep uses $K=4$.'),
}
# Numbers quoted in the Appendix D text: (setting, method, metric), paper value in the comment.
PROSE = [('default', 'leap', 'nrmse'),             # 0.6641
         ('a_ref=0.5', 'leap', 'nrmse'),           # 0.2229
         ('a_ref=1.0', 'leap', 'nrmse'),           # 0.0791
         ('substeps=2', 'leap', 'auroc'),          # 0.9171
         ('substeps=8', 'leap', 'auroc'),          # 0.9216
         ('default', 'leap', 'auroc'),             # 0.9156
         ('obs_noise=0.1', 'leap', 'auroc'),       # 0.5948
         ('n_train=32', 'leap', 'nrmse'),          # 0.8693
         ('default', 'leap', 'regret'),            # 0.0075
         ('default', 'gaussian_mle', 'regret')]    # 0.0230

# Original ExperimentConfig names of renamed Config fields, and original method names.
ORIGINAL_FIELDS = {'n_val': 'n_validation', 'x0_mean': 'initial_mean', 'x0_std': 'initial_std',
                   'substeps': 'effective_fine_substeps', 'obs_noise': 'observation_noise',
                   'a_ref': 'reference_drift_scale', 'eval_K': 'evaluation_K', 'lr': 'learning_rate',
                   'hidden': 'hidden_dim'}
ORIGINAL_METHODS = {'leap_path_nce': 'leap', 'local_transition_nce': 'transition_nce',
                    'local_gaussian_mle': 'gaussian_mle'}


def label(overrides: dict) -> str:
    """The paper's row label of a sweep setting."""
    if 'dt' in overrides:
        return rf'$\Delta t={overrides["dt"]:.2f}$, $N={overrides["n_steps"]}$'
    (key, value), = overrides.items()
    if key == 'obs_noise' and value == 0:
        return 'Clean Euler chain'
    return {'D_ref': r'$D_{{\rm ref}}={:.2f}$', 'a_ref': r'$a_{{\rm ref}}={:.2f}$',
            'obs_noise': r'$\sigma_{{\rm obs}}={:.2f}$', 'substeps': '${}$ integration substeps',
            'K': '$K={}$', 'n_train': '${}$ training paths'}[key].format(value)


def _flat(m: dict) -> dict:
    c = m['correction'] or {}
    return {'nrmse': c.get('pairwise_delta_nrmse'), 'slope': c.get('calibration_slope_true_on_predicted'),
            'auroc': m['heldout_auroc'], 'within_auroc': m['within_context_auroc'], 'regret': m['regret'],
            'learned_diffusion': m.get('learned_diffusion')}


def load_results(root) -> list[dict]:
    """Runs written by run.py."""
    runs = []
    for path in sorted(Path(root).glob('*/seed*/record.json')):
        r = read_json(path)
        runs.append({'setting': r['setting'], 'config': r['config'],
                     'methods': {k: _flat(m) for k, m in r['metrics'].items()}})
    return runs


def original_config(resolved: dict) -> dict:
    """Config values (without seed) of an original resolved_config."""
    if resolved['data_drift_scale'] != 1.0:
        raise ValueError('only the unscaled double-well data drift is supported')
    return {f.name: resolved[ORIGINAL_FIELDS.get(f.name, f.name)] for f in fields(Config) if f.name != 'seed'}


def _from_original(m: dict) -> dict:
    """An original method record in run.py's metric layout."""
    cal = m['exact_action_calibration']['metrics'] or {}
    return {'heldout_auroc': m['heldout_auroc'], 'correction': cal.get('correction_primary'),
            'within_context_auroc': cal.get('within_context_ratio', {}).get('data_reference_auroc_mean'),
            'regret': cal.get('action_selection', {}).get('mean_regret_nats'),
            'learned_diffusion': m.get('learned_diffusion')}


def load_paper_records(root) -> list[dict]:
    """Runs of the original sensitivity code, the records behind the paper tables."""
    base = read(CONFIGS/'default.json')[0]
    runs = []
    for path in sorted(Path(root).rglob('resolved_results.json')):
        r = read_json(path)
        values = original_config(r['resolved_config'])
        runs.append({'setting': setting_name(values, base), 'config': {'seed': r['resolved_config']['seed'], **values},
                     'methods': {new: _flat(_from_original(r['metrics'][old])) for old, new in ORIGINAL_METHODS.items()}})
    return runs


def _stats(values: list) -> dict | None:
    v = [x for x in values if x is not None]
    if not v:
        return None
    return {'mean': statistics.mean(v), 'sd': statistics.stdev(v) if len(v) > 1 else math.nan, 'n': len(v)}


def aggregate(runs: list[dict]) -> dict:
    """{setting: {method: {metric: {mean, sd, n}}}}: mean and sample SD over seeds."""
    groups = {}
    for r in runs:
        groups.setdefault(r['setting'], []).append(r)
    summary = {}
    for setting, rs in groups.items():
        configs = [{k: v for k, v in r['config'].items() if k != 'seed'} for r in rs]
        if any(c != configs[0] for c in configs) or len({r['config']['seed'] for r in rs}) < len(rs):
            raise ValueError(f'{setting}: runs must differ only by a unique seed')
        summary[setting] = {method: {metric: _stats([r['methods'][method][metric] for r in rs])
                                     for metric in METRICS} for method in METHODS}
    return summary


def cell(s: dict | None) -> str:
    """'$mean\\pm sd$' at four decimals; SDs below 5e-5 in scientific notation, as in the paper."""
    if s is None:
        return '---'
    if math.isnan(s['sd']):
        return f'${s["mean"]:.4f}$'
    sd = f'{s["sd"]:.4f}'
    if 0 < s['sd'] < 5e-5:
        mantissa, exponent = f'{s["sd"]:.1e}'.split('e')
        sd = rf'{mantissa}\times10^{{{int(exponent)}}}'
    return rf'${s["mean"]:.4f}\pm{sd}$'


def latex_row(name: str, stats: dict, higher: bool) -> str:
    """'label & cells \\\\'; bold marks the best printed mean, as in the paper."""
    cells = [cell(s) for s in stats.values()]
    means = [float(f'{s["mean"]:.4f}') if s else None for s in stats.values()]
    shown = [x for x in means if x is not None]
    best = (max if higher else min)(shown) if shown else None
    cells = [rf'$\mathbf{{{c[1:-1]}}}$' if x is not None and x == best else c for c, x in zip(cells, means)]
    return ' & '.join([name, *cells])+r' \\'


def build(summary: dict) -> dict:
    """{table: [{title, metric, rows: [{setting, label, stats, latex}]}]} in the paper's order."""
    tables = {}
    for table, panels in TABLES.items():
        tables[table] = []
        for title, sweep, metric in panels:
            base, settings, _ = read(CONFIGS/f'{sweep}.json')
            rows = []
            for overrides in settings:
                setting = setting_name({**base, **overrides}, base)
                stats = {m: summary[setting][m][metric] if setting in summary else None for m in METHODS}
                rows.append({'setting': setting, 'label': label(overrides), 'stats': stats,
                             'latex': latex_row(label(overrides), stats, metric in HIGHER_IS_BETTER)})
            tables[table].append({'title': title, 'metric': metric, 'rows': rows})
    return tables


def latex(table: str, panels: list) -> str:
    """The paper's layout: one tabular split by rules, or titled tabular* panels."""
    header = 'Setting & '+' & '.join(METHODS.values())+r' \\'
    lines = [r'\begin{table}[htbp]', r'\centering\small', rf'\caption{{{CAPTIONS[table]}}}', rf'\label{{tab:{table}}}']
    if panels[0]['title'] is None:
        lines += [r'\begin{tabular}{@{}lccc@{}}', r'\toprule', header, r'\midrule']
        for i, panel in enumerate(panels):
            if i:
                lines.append(r'\midrule')
            lines += [r['latex'] for r in panel['rows']]
        lines += [r'\bottomrule', r'\end{tabular}']
    else:
        for i, panel in enumerate(panels):
            if i:
                lines.append(r'\par\medskip')
            lines += [r'\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lccc@{}}', r'\toprule',
                      rf'\multicolumn{{4}}{{@{{}}l}}{{\textbf{{{panel["title"]}}}}} \\', r'\addlinespace[2pt]',
                      header, r'\midrule', *[r['latex'] for r in panel['rows']], r'\bottomrule', r'\end{tabular*}']
    return '\n'.join(lines+[r'\end{table}'])


def _text(s: dict | None) -> str:
    return '---' if s is None else f'{s["mean"]:.4f} ± {s["sd"]:.4f}'


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--results', type=Path, help='directory written by run.py')
    source.add_argument('--paper-records', type=Path, help='directory of original resolved_results.json files')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    runs = load_results(args.results) if args.results else load_paper_records(args.paper_records)
    summary = aggregate(runs)
    tables = build(summary)
    prose = [{'setting': s, 'method': m, 'metric': k, **(summary.get(s, {}).get(m, {}).get(k) or {})}
             for s, m, k in PROSE]
    missing = sorted({r['setting'] for panels in tables.values() for p in panels for r in p['rows']}-set(summary))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/'tables.tex').write_text('\n\n'.join(latex(t, p) for t, p in tables.items())+'\n')
    write_json(args.out/'summary.json', {'runs': len(runs), 'tables': tables, 'prose': prose,
                                         'settings': summary, 'missing': missing})
    for table, panels in tables.items():
        for panel in panels:
            print(f'\n{table}: {panel["title"] or panel["metric"]}')
            print(f'  {"setting":<22}'+''.join(f'{name:>20}' for name in METHODS.values()))
            for r in panel['rows']:
                print(f'  {r["setting"]:<22}'+''.join(f'{_text(s):>20}' for s in r['stats'].values()))
    print('\nprose: '+', '.join(f'{p["setting"]} {p["method"]} {p["metric"]} {p.get("mean", math.nan):.4f}'
                                for p in prose))
    if missing:
        print('missing settings: '+', '.join(missing))
    print(f'wrote {args.out}/tables.tex and summary.json')


if __name__ == '__main__':
    main()

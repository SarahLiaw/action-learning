"""Figure 6: learned orbital corrections against the path summary A, with the exact log cosh A.

python -m experiments.orbital.figure --results results/orbital --out results/figures/figure6.pdf
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from leap import read_json

from . import data as D
from .evidence import centered, log_cosh, solar_evidence

LEAP_BLUE, LOCAL_RED, EXACT, GRID = '#2171b5', '#d62728', '#1b1b1e', '#e9e9e9'
W_IN, H_IN = .48*5.5, 1.95     # half the text width
STYLE = {'font.family': 'serif', 'font.serif': ['STIXGeneral'], 'mathtext.fontset': 'stix', 'pdf.fonttype': 42,
         'font.size': 7.5, 'axes.labelsize': 9.5, 'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5,
         'legend.fontsize': 8.5, 'axes.linewidth': .6, 'axes.spines.top': False, 'axes.spines.right': False,
         'xtick.major.width': .6, 'ytick.major.width': .6, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
         'xtick.major.pad': 1.5, 'ytick.major.pad': 1.5, 'axes.labelpad': 2, 'legend.frameon': False,
         'legend.handlelength': 1.6, 'legend.borderaxespad': .2}
METHODS = [('nre_leap_local', 'LEAP-Local', LOCAL_RED, 's'), ('nre_leap_global', 'LEAP-Global', LEAP_BLUE, 'o')]


def mean_prediction(results: Path, config: str) -> np.ndarray:
    """Mean over seeds of a config's learned correction on the test candidates."""
    return np.mean([torch.load(f).numpy() for f in sorted((results/config).glob('seed*/predictions.pt'))], axis=0)


def plot(results: Path, bundles: Path, out: Path) -> None:
    bundle = D.load_bundle(bundles/read_json(next((results/'nre_leap_global').glob('seed*/result.json')))['bundle']['file'])
    score, _ = solar_evidence(bundle)
    A, ids = float(bundle['epsilon'])*score, bundle['test_context_ids'].numpy()
    # corrections are identified up to a context constant: align each context with the exact mean
    shift = log_cosh(A)-centered(log_cosh(A), ids)
    plt.rcParams.update(STYLE)
    fig = plt.figure(figsize=(W_IN, H_IN))
    ax = fig.add_axes([.47/W_IN, .42/H_IN, (W_IN-.53)/W_IN, (H_IN-.50)/H_IN])
    ax.grid(axis='y', color=GRID, lw=.5, zorder=0)
    ax.set_axisbelow(True)
    for config, label, color, marker in METHODS:
        dot = marker == 'o'
        ax.scatter(A, centered(mean_prediction(results, config), ids)+shift, s=5 if dot else 4.5, marker=marker,
                   color=color, alpha=.7 if dot else .45, lw=0, zorder=4 if dot else 3, label=label, rasterized=True)
    xs = np.linspace(-7.6, 6.6, 400)
    ax.plot(xs, np.log(np.cosh(xs)), color=EXACT, lw=1., ls=(0, (4, 2)), zorder=5, label=r'Exact, $\log\cosh A$')
    ax.set(xlim=(-7.6, 6.6), ylim=(-.4, 7.4), xticks=[-6, -3, 0, 3, 6], yticks=[0, 2, 4, 6])
    ax.set_xlabel(r'Accumulated evidence $A$')
    ax.set_ylabel('Correction (nats)')
    handles, labels = ax.get_legend_handles_labels()
    order = [labels.index(r'Exact, $\log\cosh A$'), labels.index('LEAP-Global'), labels.index('LEAP-Local')]
    legend = ax.legend([handles[i] for i in order], [labels[i] for i in order], loc='upper center', markerscale=2.2,
                       handletextpad=.3, borderaxespad=.1, labelspacing=.25)
    for handle in legend.legend_handles:
        handle.set_alpha(1)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=400)
    plt.close(fig)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--results', type=Path, default=Path('results/orbital'))
    ap.add_argument('--bundles', type=Path, default=Path(os.environ.get('LEAP_ORBITAL_BUNDLES', 'data/orbital/bundles')))
    ap.add_argument('--out', type=Path, default=Path('results/figures/figure6.pdf'))
    args = ap.parse_args(argv)
    plot(args.results, args.bundles, args.out)
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()

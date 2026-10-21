"""Figure 5: the double well potential and the drift recovered from the selected paths.

python -m experiments.double_well.figure --results results/double_well --out results/figures/figure5.pdf
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from leap import read_json

from .model import ConservativePotential
from .run import CHECKPOINT

LEAP_BLUE, CUBIC_RED, INK, GUIDE, A_RED, B_BLUE = '#2171b5', '#d62728', '#1B1B1E', '#c4c4c4', '#d6604d', '#2b6cb0'
W_IN, H_IN, LEFT, RIGHT, TOP = 1.98, 1.6144, .34, .025, .02     # sized for a 0.36 \textwidth column
AX_B, H_B, GAP = .415, .62, .05
STYLE = {'font.family': 'serif', 'font.serif': ['STIXGeneral'], 'mathtext.fontset': 'stix', 'pdf.fonttype': 42,
         'axes.spines.top': False, 'axes.spines.right': False}


def drift_curves(cfg: dict, results: Path, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """LEAP and cubic regression drift of each seed at the states x, two [seeds, len(x)] arrays."""
    leap, cubic = [], []
    for seed in cfg['seeds']:
        run = results/f'seed{seed}'
        model = ConservativePotential(cfg['system']['diffusion'], cfg['system']['dt'], **cfg['model']).eval()
        model.load_state_dict(torch.load(run/CHECKPOINT, map_location='cpu', weights_only=True))
        leap.append(model.drift(torch.as_tensor(x, dtype=torch.float32)[:, None]).numpy().ravel())
        beta = read_json(run/'results.json')['cubic_coefficients']
        cubic.append(sum(b*x**i for i, b in enumerate(beta)))
    return np.array(leap), np.array(cubic)


def _axes(fig, bottom, height, letter):
    ax = fig.add_axes([LEFT/W_IN, bottom/H_IN, (W_IN-LEFT-RIGHT)/W_IN, height/H_IN])
    for v in (-1, 0, 1):
        ax.axvline(v, color=GUIDE, lw=.5, ls=(0, (2.5, 2)), zorder=0)
    ax.grid(axis='y', color='#e6e6e6', lw=.5, zorder=0)
    ax.set_xlim(-1.5, 1.5)
    ax.set_xticks([-1, 0, 1])
    ax.tick_params(labelsize=7, length=2, width=.5, pad=1.5)
    for side in ('left', 'bottom'):
        ax.spines[side].set_linewidth(.6)
    ax.yaxis.label.set_y(.4)
    ax.text(-(LEFT-.012)/(W_IN-LEFT-RIGHT), 1., letter, transform=ax.transAxes, fontsize=8, fontweight='bold',
            ha='left', va='top')
    return ax


def plot(cfg: dict, results: Path, out: Path) -> None:
    x = np.linspace(-1.5, 1.5, 601).astype(np.float32)
    leap, cubic = drift_curves(cfg, results, x)
    plt.rcParams.update(STYLE)
    fig = plt.figure(figsize=(W_IN, H_IN))
    ax_b = _axes(fig, AX_B, H_B, 'b')
    ax_a = _axes(fig, AX_B+H_B+GAP, H_IN-TOP-(AX_B+H_B+GAP), 'a')

    # (a) U(x) = (x^2-1)^2/4 with the start basin A and the target basin B
    U = (x**2-1)**2/4
    ax_a.fill_between(x, 0, U, where=x <= -.7, color=A_RED, alpha=.16, lw=0, zorder=1)
    ax_a.fill_between(x, 0, U, where=x >= .7, color=B_BLUE, alpha=.16, lw=0, zorder=1)
    ax_a.plot(x, U, color=INK, lw=1.2, zorder=3, solid_capstyle='round')
    ax_a.annotate('', xy=(0, .25), xytext=(0, 0), zorder=2,
                  arrowprops=dict(arrowstyle='<->', color='#5b5b5b', lw=.7, shrinkA=0, shrinkB=0, mutation_scale=5))
    ax_a.text(0, .262, r'$\Delta U=0.25$', fontsize=6.5, color='#3b3b3b', ha='center', va='bottom')
    for x0, name, color in [(-1, 'A', A_RED), (1, 'B', B_BLUE)]:
        ax_a.scatter([x0], [0], s=13, color=color, zorder=4, clip_on=False, lw=0)
        ax_a.text(x0, .035, rf'${name}$', fontsize=7.5, color=color, ha='center', va='bottom')
    ax_a.set_ylim(-.015, .345)
    ax_a.set_yticks([0, .2])
    ax_a.set_yticklabels(['0', '0.2'])
    ax_a.tick_params(labelbottom=False)
    ax_a.set_ylabel('$U(x)$', fontsize=7.5, labelpad=1.5)

    # (b) mean drift over seeds with a band of one sample SD, against the true drift
    handles = []
    for arr, color, ls, lw, label in [(leap, LEAP_BLUE, '-', 1.5, 'LEAP'),
                                      (cubic, CUBIC_RED, (0, (1, 1.3)), 1.3, 'Cubic reg.')]:
        mu, sd = arr.mean(0), arr.std(0, ddof=1)
        ax_b.fill_between(x, mu-sd, mu+sd, color=color, alpha=.2, lw=0, zorder=1)
        handles.append(ax_b.plot(x, mu, color=color, ls=ls, lw=lw, label=label, zorder=2)[0])
    true, = ax_b.plot(x, x-x**3, color=INK, ls=(0, (3.5, 2)), lw=.9, label='True $b$', zorder=3)
    ax_b.set_ylim(-2.5, 2.3)
    ax_b.set_yticks([-2, 0, 2])
    ax_b.set_ylabel('$b(x)$', fontsize=7.5, labelpad=1.5)
    ax_b.set_xlabel('State $x$', fontsize=7.5, labelpad=1)
    fig.legend(handles=[true, *handles], loc='lower center', ncol=3, frameon=False, fontsize=6.5,
               bbox_to_anchor=(.5, 0.), handlelength=1.5, handletextpad=.4, columnspacing=.8, borderpad=.15,
               borderaxespad=.05)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=400)
    plt.close(fig)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', type=Path, default=Path('configs/double_well/paper.json'))
    ap.add_argument('--results', type=Path, default=Path('results/double_well'))
    ap.add_argument('--out', type=Path, default=Path('results/figures/figure5.pdf'))
    args = ap.parse_args(argv)
    plot(read_json(args.config), args.results, args.out)
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()

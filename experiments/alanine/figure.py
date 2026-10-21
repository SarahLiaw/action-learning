"""Figure 7: completion, preference weight and an angular bound with a preference, all with one 600 K action.

python -m experiments.alanine.figure --results results/alanine --out results/figures/figure7.pdf
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle
import numpy as np

RED, BLUE, WHITE, PINK, INK, MARK = '#fa765a', '#5ed8f4', '#f7f7f7', '#f7a4e0', '#30334c', '#242538'
DASH, DOT, DASHDOT = (0, (3.2, 1.8)), (0, (1.2, 1.4)), (0, (3.2, 1.3, 1.0, 1.3))
W_IN, LEFT, GAP, S, CB_GAP, CB_W, BOTTOM, TOP_PAD = 5.5, .45, .14, 1.377, .10, .09, .75, .24   # inches
H_IN = BOTTOM+S+TOP_PAD


def wrap(x):
    return (x+np.pi) % (2*np.pi)-np.pi


def segments(path):
    """A (phi, psi) path in degrees, joined by shortest angular steps and cut where it crosses the periodic seam."""
    a, pieces = wrap(np.asarray(path)), []
    for start, v in zip(a[:-1], wrap(np.diff(path, axis=0))):
        cuts = sorted({0., 1.} | {float(t) for j in range(2) if abs(v[j]) > 1e-12
                                  for t in ((edge-start[j])/v[j] for edge in (-np.pi, np.pi)) if 0 < t < 1})
        for left, right in zip(cuts[:-1], cuts[1:]):
            shift = 2*np.pi*np.floor((start+.5*(left+right)*v+np.pi)/(2*np.pi))
            pieces.append(np.degrees(np.array([start+left*v, start+right*v])-shift))
    return pieces


def stroke(lw, color):
    width = lw+(1.1 if color == WHITE else .55)      # white curves need a heavier outline on light regions
    return [pe.Stroke(linewidth=width, foreground=INK, alpha=.9 if color == WHITE else .7), pe.Normal()]


def draw(ax, path, color, ls='-', lw=1.15, knots=False, zorder=5):
    """A periodic path, with arrowheads where it leaves one edge and re-enters at the opposite one."""
    runs = []
    for xy in segments(path):
        if runs and np.allclose(runs[-1][-1], xy[0], atol=1e-7):
            runs[-1] = np.vstack([runs[-1], xy[1]])
        else:
            runs.append(xy.copy())
    for xy in runs:
        ax.plot(xy[:, 0], xy[:, 1], color=color, ls=ls, lw=lw, zorder=zorder, solid_capstyle='round',
                dash_capstyle='round', path_effects=stroke(lw, color))
    for a, b in zip(runs[:-1], runs[1:]):
        axis = 0 if abs(abs(a[-1][0])-180) < 1e-6 else 1
        heading = np.sign(a[-1][axis])
        marker = ('<' if heading < 0 else '>') if axis == 0 else ('v' if heading < 0 else '^')
        for point, inward in [(a[-1], -heading), (b[0], heading)]:
            q = point.copy()
            q[axis] += 5*inward
            ax.scatter(*q, marker=marker, s=14, color=color, edgecolors=INK, linewidths=.45, zorder=zorder+2,
                       clip_on=False)
    if knots:
        frames = np.degrees(wrap(path))
        ax.scatter(frames[1:-1, 0], frames[1:-1, 1], s=3.2, color=color, edgecolors=INK, linewidths=.2,
                   zorder=zorder+1)
    return Line2D([], [], color=color, ls=ls, lw=lw+.2, path_effects=stroke(lw+.2, color))


def background(ax, degrees, free_energy):
    ax.set_rasterization_zorder(2)
    filled = ax.contourf(degrees, degrees, free_energy.T, levels=np.linspace(0, 12, 49), cmap='viridis', extend='max',
                         zorder=0)
    ax.contour(degrees, degrees, free_energy.T, levels=np.arange(1.5, 12, 1.5), colors='white', linewidths=.3,
               alpha=.38, zorder=1)
    ticks = [-180, -90, 0, 90, 180]
    ax.set(xlim=(-180, 180), ylim=(-180, 180), xticks=ticks, yticks=ticks)
    ax.set_xticklabels(['', r'$-90$', '0', '90', ''])
    ax.set_yticklabels(['', r'$-90$', '0', '90', ''])
    ax.set_aspect('equal')
    ax.tick_params(length=2.2, width=.5, pad=1.5, labelsize=8)
    for spine in ax.spines.values():
        spine.set_linewidth(.55)
        spine.set_color('#363b48')
    ax.set_xlabel(r'$\phi$ (degrees)', fontsize=9, labelpad=1.5)
    return filled


def endpoints(ax, path, observed=()):
    a = np.degrees(wrap(np.asarray(path)))
    style = dict(facecolors='white', edgecolors=MARK, linewidths=.7, zorder=9)
    ax.scatter(*a[0], s=16, marker='D', **style)
    ax.scatter(*a[-1], s=17, marker='s', **style)
    if len(observed):
        ax.scatter(a[observed, 0], a[observed, 1], s=14, marker='o', **style)


def symbol(marker):
    return Line2D([], [], color=MARK, marker=marker, ls='none', markersize=4.2, markerfacecolor='white',
                  markeredgewidth=.7)


def title(ax, letter, text):
    ax.annotate(letter, (0, 1), xycoords='axes fraction', xytext=(0, 4), textcoords='offset points', fontsize=9,
                fontweight='bold', va='bottom')
    ax.annotate(text, (0, 1), xycoords='axes fraction', xytext=(10, 4), textcoords='offset points', fontsize=9,
                va='bottom')


def legend(ax, handles, labels, ncol=2):
    ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, -.275), ncol=ncol, frameon=False, fontsize=8,
              handlelength=1.8, handletextpad=.45, columnspacing=.8, borderaxespad=0., labelspacing=.3)


def plot(results: Path, out: Path) -> None:
    folder = results/'composed'
    shown, routes, field = (np.load(folder/name) for name in
                            ('displayed_paths.npz', 'composed_paths.npz', 'landscape_T600.npz'))
    surface = field['degrees'], field['free_energy_kcal_mol']
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['STIXGeneral'], 'mathtext.fontset': 'stix',
                         'pdf.fonttype': 42})
    fig = plt.figure(figsize=(W_IN, H_IN))
    axes = [fig.add_axes([(LEFT+i*(S+GAP))/W_IN, BOTTOM/H_IN, S/W_IN, S/H_IN]) for i in range(3)]

    # (a) completion through three observed frames
    ax = axes[0]
    background(ax, *surface)
    ax.set_ylabel(r'$\psi$ (degrees)', fontsize=9, labelpad=1)
    path = shown['T600__completion__leap']
    observed = [round((len(path)-1)*f) for f in (.25, .5, .75)]
    line = draw(ax, path, RED, knots=True)
    endpoints(ax, path, observed)
    legend(ax, [line, symbol('o'), symbol('D'), symbol('s')], ['LEAP', 'Observed', 'Start', 'End'])
    title(ax, 'a', 'Completion')

    # (b) increasing preference weight for the shaded basin on the same context; frame dots show dwell time
    ax = axes[1]
    background(ax, *surface)
    ax.tick_params(labelleft=False)
    ax.add_patch(Circle((-60, -40), 30, facecolor='#f8bba3', edgecolor='white', lw=.6, ls=(0, (1.2, 1.6)), alpha=.5,
                        zorder=2))
    sweep = [(0, BLUE, DASH), (5, WHITE, DASHDOT), (20, RED, '-'), (50, PINK, DOT)]
    lines = [draw(ax, routes[f'preference_sweep_{lam}'], color, ls, lw=1.0, knots=True, zorder=5+lam/50)
             for lam, color, ls in sweep]
    endpoints(ax, routes['preference_sweep_0'])
    legend(ax, lines, [rf'$\lambda={lam}$' for lam, _, _ in sweep])
    title(ax, 'b', 'Preference weight')

    # (c) the bound |phi| <= 150 degrees composed with the preference, on the constraint context
    ax = axes[2]
    filled = background(ax, *surface)
    ax.tick_params(labelleft=False)
    for lo, hi in [(-180, -150), (150, 180)]:
        ax.axvspan(lo, hi, facecolor='#e8e9ea', alpha=.6, zorder=2, lw=0)
        ax.axvline(-150 if lo < 0 else 150, color='#f3f4f5', lw=.7, zorder=3)
    ax.add_patch(Rectangle((30, -180), 100, 180, fill=False, edgecolor='white', lw=.6, ls=(0, (1.2, 1.6)), alpha=.85,
                           zorder=3))
    ax.text(80, -150, r'$B$', ha='center', va='center', color='white', fontsize=8,
            path_effects=[pe.withStroke(linewidth=1.2, foreground='#324063')], zorder=8)
    ax.add_patch(Circle((-60, -40), 30, facecolor='#f8bba3', edgecolor='white', lw=.6, ls=(0, (1.2, 1.6)), alpha=.5,
                        zorder=2))
    rows = [('preference_only_20', WHITE, 'Preference only', DASHDOT), ('bound_preference_20', RED, 'Both', '-')]
    lines = [draw(ax, routes[key], color, ls, knots=True, zorder=5+(key == 'bound_preference_20'))
             for key, color, _, ls in rows]
    endpoints(ax, routes['original'])
    legend(ax, lines, [label for _, _, label, _ in rows], ncol=1)
    title(ax, 'c', 'Bound + preference')

    bar = fig.colorbar(filled, cax=fig.add_axes([(LEFT+3*S+2*GAP+CB_GAP)/W_IN, BOTTOM/H_IN, CB_W/W_IN, S/H_IN]),
                       ticks=[0, 3, 6, 9, 12])
    bar.set_label(r'Empirical $F$ (kcal mol$^{-1}$)', fontsize=8, labelpad=3)
    bar.ax.tick_params(labelsize=8, length=2, width=.5, pad=1.5)
    bar.outline.set_linewidth(.5)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=400, facecolor='white')
    plt.close(fig)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--results', type=Path, default=Path('results/alanine'))
    ap.add_argument('--out', type=Path, default=Path('results/figures/figure7.pdf'))
    args = ap.parse_args(argv)
    plot(args.results, args.out)
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()

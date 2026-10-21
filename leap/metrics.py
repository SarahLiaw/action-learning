"""Action metrics that remove only a per-context additive constant."""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch
from scipy import stats


@dataclass
class ActionCalibration:
    n_paths: int
    n_contexts: int
    gauge_rmse: float
    gauge_nrmse: float
    pairwise_delta_rmse: float
    pairwise_delta_nrmse: float          # NRMSE_dG in the paper
    centered_r2: float
    centered_truth_std: float
    calibration_slope_true_on_predicted: float
    pearson: float
    spearman: float
    kendall: float


def _vector(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError(f'expected a 1-D array, got {x.shape}')
    return x.astype(np.float64, copy=False)


def action_calibration(predicted, truth, context_ids) -> ActionCalibration:
    """Compare predicted and exact actions (or corrections) within each context."""
    pred, true, ctx = _vector(predicted), _vector(truth), _vector(context_ids)
    if not (pred.shape == true.shape == ctx.shape):
        raise ValueError('predicted, truth and context_ids must have equal shape')
    if not (np.isfinite(pred).all() and np.isfinite(true).all()):
        raise ValueError('actions contain non-finite values')
    aligned = pred.copy()
    pc, tc, pair_mse = [], [], []
    for c in np.unique(ctx):
        m = ctx == c
        if m.sum() < 2:
            continue
        p, t = pred[m], true[m]
        aligned[m] += t.mean()-p.mean()
        pc.append(p-p.mean())
        tc.append(t-t.mean())
        pair_mse.append(2.0*float(np.mean((pc[-1]-tc[-1])**2)))   # mean over ordered pairs
    if not pc:
        raise ValueError('every context has fewer than two paths')
    p, t = np.concatenate(pc), np.concatenate(tc)
    t_std = float(np.sqrt(np.mean(t**2)))
    gauge = float(np.sqrt(np.mean((aligned-true)**2)))
    pair = float(np.sqrt(np.mean(pair_mse)))
    pair_scale = float(np.sqrt(np.mean([2.0*float(np.mean(g**2)) for g in tc])))
    denom, t_ss, e_ss = float(p@p), float(t@t), float((p-t)@(p-t))
    return ActionCalibration(
        n_paths=int(pred.size), n_contexts=len(pc),
        gauge_rmse=gauge, gauge_nrmse=gauge/t_std if t_std > 0 else math.nan,
        pairwise_delta_rmse=pair, pairwise_delta_nrmse=pair/pair_scale if pair_scale > 0 else math.nan,
        centered_r2=1.0-e_ss/t_ss if t_ss > 0 else math.nan, centered_truth_std=t_std,
        calibration_slope_true_on_predicted=float(p@t/denom) if denom > 0 else math.nan,
        pearson=float(stats.pearsonr(p, t).statistic),
        spearman=float(stats.spearmanr(p, t).statistic),
        kendall=float(stats.kendalltau(p, t).statistic),
    )


def binary_auroc(scores, labels) -> float:
    """Tie-aware AUROC."""
    s, y = _vector(scores), _vector(labels).astype(np.int64)
    if not np.isin(y, (0, 1)).all():
        raise ValueError('labels must be binary')
    pos, neg = int(y.sum()), int(y.size-y.sum())
    if pos == 0 or neg == 0:
        return math.nan
    ranks = stats.rankdata(s, method='average')
    return (float(ranks[y == 1].sum())-pos*(pos+1)/2.0)/(pos*neg)


def within_context_auroc(scores, labels, context_ids) -> float:
    """Mean over contexts of the data-versus-reference AUROC."""
    s, y, c = _vector(scores), _vector(labels), _vector(context_ids)
    values = [binary_auroc(s[c == k], y[c == k]) for k in np.unique(c)]
    values = [v for v in values if not math.isnan(v)]
    return float(np.mean(values)) if values else math.nan

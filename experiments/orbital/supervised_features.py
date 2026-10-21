"""Oracle-supervised control on accumulated features (A, B).

a_k = e_k.s_k and b_k = |s_k|^2/2 with e_k = Sigma^{-1/2}(x_{k+1} - mu_0(x_k)) and
s_k = Sigma^{-1/2}(mu_+(x_k) - mu_0(x_k)); A = sum a_k, B = sum b_k. Models are fitted to the
exact correction on disjoint context folds, or on a fixed half split for candidate selection.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from scipy.stats import spearmanr

from leap import action_calibration

from .data import laws


def step_features(bundle: dict, batch: int = 16) -> tuple[np.ndarray, np.ndarray]:
    """Per-step (a_k, b_k): [n, N] each, from stored candidates in float64."""
    data, reference = laws(bundle['config'], float(bundle['epsilon']))
    paths = bundle['test_candidates'].to(torch.float64)*bundle['scales'].to(torch.float64)
    sd = torch.tensor(reference.std, dtype=torch.float64)
    a, b = [], []
    for chunk in paths.split(batch):
        cur = chunk[:, :-1]
        mu0 = reference.mean(cur, 0)
        e, s = (chunk[:, 1:]-mu0)/sd, (data.mean(cur, 1)-mu0)/sd
        a.append((e*s).sum(dim=(2, 3)))
        b.append(0.5*s.square().sum(dim=(2, 3)))
    return torch.cat(a).numpy(), torch.cat(b).numpy()


def closed_form(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Reduced symmetric correction log cosh A - B."""
    return np.logaddexp(A, -A)-math.log(2.0)-B


def _head() -> torch.nn.Sequential:
    return torch.nn.Sequential(torch.nn.Linear(2, 128), torch.nn.SiLU(), torch.nn.Linear(128, 128),
                               torch.nn.SiLU(), torch.nn.Linear(128, 1))


def _inputs(A, B, train: np.ndarray) -> torch.Tensor:
    ab = torch.tensor(np.stack([A, B], 1), dtype=torch.float32)
    return (ab-ab[train].mean(0))/ab[train].std(0).clamp_min(1e-8)


def fit_mlp(A, B, G, ctx, train: np.ndarray, epochs: int = 3000, lr: float = 1e-3, seed: int = 0) -> np.ndarray:
    """MLP on (A, B) minimizing the within-context centered squared error; predicts every path."""
    torch.manual_seed(seed)
    x, y, c = _inputs(A, B, train), torch.tensor(G, dtype=torch.float32), torch.tensor(ctx)
    tm = torch.tensor(train)
    model = _head()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-6)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    for _ in range(epochs):
        r = model(x).squeeze(-1)-y
        loss = 0.0
        for cc in c[tm].unique():
            rm = r[tm & (c == cc)]
            loss = loss+(rm-rm.mean()).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    with torch.no_grad():
        return model(x).squeeze(-1).numpy()


def untrained_mlp(A, B, train: np.ndarray, seed: int = 1234) -> np.ndarray:
    """Architecture-matched control with random weights."""
    torch.manual_seed(seed)
    model = _head()
    with torch.no_grad():
        return model(_inputs(A, B, train)).squeeze(-1).numpy()


def fit_additive(a, b, G, ctx, train: np.ndarray, bins: int = 40, ridge: float = 1e-3) -> np.ndarray:
    """Ridge on per-path counts of a_k in training-fold quantile bins, plus B."""
    edges = np.quantile(a[train].reshape(-1), np.linspace(0, 1, bins+1))
    H = np.stack([((a >= edges[i]) & (a < edges[i+1])).sum(1) for i in range(bins)], 1).astype(float)
    X = np.concatenate([b.sum(1)[:, None], H, np.ones((len(H), 1))], 1)
    Xc, yc = X.copy(), G.copy()
    for cc in np.unique(ctx):
        m = ctx == cc
        Xc[m] -= Xc[m].mean(0)
        yc[m] -= yc[m].mean()
    sc = Xc.std(0)+1e-12
    Xs = Xc/sc
    w = np.linalg.solve(Xs[train].T@Xs[train]+ridge*train.sum()*np.eye(Xs.shape[1]), Xs[train].T@yc[train])
    return (X/sc)@w


def _nrmse(pred, G, ctx) -> float:
    return action_calibration(pred, G, ctx).pairwise_delta_nrmse


def fold_study(bundle: dict, folds: int = 5, seed: int = 0, epochs: int = 3000) -> dict:
    """Additive and MLP fits held out by context fold, and the closed form."""
    a, b = step_features(bundle)
    A, B = a.sum(1), b.sum(1)
    G, ctx = bundle['test_oracle_ratio'].numpy(), bundle['test_context_ids'].numpy()
    pred_add, pred_mlp, fold_add, fold_mlp = np.zeros_like(G), np.zeros_like(G), [], []
    for f in np.array_split(np.random.default_rng(seed).permutation(np.unique(ctx)), folds):
        te = np.isin(ctx, f)
        pred_add[te] = fit_additive(a, b, G, ctx, ~te)[te]
        pred_mlp[te] = fit_mlp(A, B, G, ctx, ~te, epochs=epochs, seed=seed)[te]
        fold_add.append(_nrmse(pred_add[te], G[te], ctx[te]))
        fold_mlp.append(_nrmse(pred_mlp[te], G[te], ctx[te]))
    return {'num_steps': int(bundle['config']['num_steps']),
            'additive_pooled': _nrmse(pred_add, G, ctx), 'mlp_pooled': _nrmse(pred_mlp, G, ctx),
            'closed_form': _nrmse(closed_form(A, B), G, ctx),
            'additive_folds': fold_add, 'mlp_folds': fold_mlp,
            'mlp_fold_mean': float(np.mean(fold_mlp)), 'mlp_fold_sd': float(np.std(fold_mlp, ddof=1))}


def min_separation(bundle: dict) -> np.ndarray:
    """Closest approach between any two bodies over all frames of each candidate."""
    q = (bundle['test_candidates'].to(torch.float64)*bundle['scales']).numpy()[..., :3]
    d = np.linalg.norm(q[:, :, :, None, :]-q[:, :, None, :, :], axis=-1)
    i, j = np.triu_indices(q.shape[2], 1)
    return d[:, :, i, j].min(axis=(1, 2))


def selection_study(bundle: dict, penalty: float = 1.0, epochs: int = 3000) -> dict:
    """Fit on 12 contexts, select argmin G on the other 12, with and without a separation constraint."""
    a, b = step_features(bundle)
    A, B = a.sum(1), b.sum(1)
    G, ctx = bundle['test_oracle_ratio'].numpy(), bundle['test_context_ids'].numpy()
    uniq = np.unique(ctx)
    train_ctx = set(np.random.default_rng(0).choice(uniq, size=len(uniq)//2, replace=False).tolist())
    train = np.array([c in train_ctx for c in ctx])
    te = ~train
    sep = min_separation(bundle)
    d_min = float(np.median(sep[te]))
    C = np.maximum(0.0, d_min-sep)
    arms = {'accumulated_mlp': fit_mlp(A, B, G, ctx, train, epochs=epochs),
            'linear_A': A.copy(), 'untrained_mlp': untrained_mlp(A, B, train)}
    out = {'d_min': d_min, 'held_out_contexts': sorted(set(uniq.tolist())-train_ctx)}
    for name, s in arms.items():
        pct, picks, feas_pct = [], 0, []
        for c in np.unique(ctx[te]):
            m = (ctx == c) & te
            g, sm = G[m], s[m]
            pct.append(100.0*(g < g[int(np.argmin(sm))]).mean())
            i = int(np.argmin((sm-sm.mean())/(sm.std()+1e-12)+penalty*(C[m]/(C[m].std()+1e-12))))
            if sep[m][i] >= d_min:
                picks += 1
                feas_pct.append(100.0*(g[sep[m] >= d_min] < g[i]).mean())
        out[name] = {'spearman': float(spearmanr(s[te], G[te]).statistic), 'selection_percentile': float(np.mean(pct)),
                     'feasible_picks': picks, 'feasible_percentile': float(np.mean(feas_pct)) if feas_pct else None}
    return out

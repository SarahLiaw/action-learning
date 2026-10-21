"""Held-out evaluation: AUROC, correction calibration against the exact action, candidate regret."""
from __future__ import annotations

from dataclasses import asdict

import numpy as np
import torch

from leap import action_calibration, binary_auroc, within_context_auroc

from . import dynamics
from .models import GaussianMLE, score

EVAL_SEED = 90_001   # offset from the run seed for the evaluation references


def candidate_regret(predicted: torch.Tensor, truth: torch.Tensor, width: int) -> float:
    """Mean exact-action excess of choosing each context's lowest learned action."""
    p = predicted.numpy().astype(np.float64).reshape(-1, width)
    t = truth.numpy().astype(np.float64).reshape(-1, width)
    rows = np.arange(len(t))
    return float(np.mean(t[rows, p.argmin(1)]-t[rows, t.argmin(1)]))


@torch.no_grad()
def evaluate(models: dict, cfg, test: torch.Tensor, ref) -> tuple[dict, float | None]:
    """(metrics per model, exact-ratio AUROC) over test contexts of one data path and eval_K reference paths."""
    refs = ref.sample(test[:, 0], cfg.eval_K, torch.Generator().manual_seed(cfg.seed+EVAL_SEED))
    B, K, frames = refs.shape
    flat = refs.reshape(B*K, frames)
    pooled = torch.cat((torch.ones(B, dtype=torch.int64), torch.zeros(B*K, dtype=torch.int64)))
    candidates = torch.cat((test[:, None], refs), dim=1).reshape(B*(K+1), frames)
    contexts = torch.arange(B).repeat_interleave(K+1)
    is_data = (torch.arange(K+1) == 0).long().repeat(B)
    ref_log_prob = ref.log_prob(candidates)
    truth = -dynamics.log_prob(candidates, cfg) if cfg.exact else None
    metrics = {}
    for name, model in models.items():
        model.eval()
        # pooled over every data and reference path: the paper's AUROC
        m = {'heldout_auroc': binary_auroc(torch.cat((score(model, test, ref), score(model, flat, ref))), pooled)}
        action = model.action(candidates)
        ratio = -action-ref_log_prob
        m['within_context_auroc'] = within_context_auroc(ratio, is_data, contexts)
        # calibration needs the exact observed-path density: no hidden substeps, no observation noise
        m['correction'] = asdict(action_calibration(ratio, -truth-ref_log_prob, contexts)) if cfg.exact else None
        m['regret'] = candidate_regret(action, truth, K+1) if cfg.exact else None
        if isinstance(model, GaussianMLE):
            m['learned_diffusion'] = float(model.D)
        metrics[name] = m
    if not cfg.exact:
        return metrics, None
    both = torch.cat((test, flat))
    return metrics, binary_auroc(dynamics.log_prob(both, cfg)-ref.log_prob(both), pooled)

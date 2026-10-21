"""Scoring learned corrections against the exact mixture correction."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch

from leap import action_calibration, binary_auroc, within_context_auroc

from .models import NominalPhysics, build_model


def physics_for(bundle: dict) -> NominalPhysics:
    """Nominal simulator features for a bundle's bodies, scales and transition noise."""
    c = bundle['config']
    return NominalPhysics(bundle['masses_kg'], bundle['scales'], dt=float(c['dt_days']),
                          position_std=float(c['position_std_au']), velocity_std=float(c['velocity_std_au_per_day']))


@torch.no_grad()
def predict(model, bundle: dict, *, device: str = 'cpu', batch_size: int = 16) -> torch.Tensor:
    """G_theta on the test candidates (float64); Gaussian models use their likelihood ratio to N(0, I)."""
    model.eval().to(device)
    masses = bundle['masses_kg'].to(device=device, dtype=torch.float32)
    ids = bundle['body_ids'].to(device)
    score = getattr(model, 'correction', model)
    x = bundle['test_candidates']
    return torch.cat([score(x[i].to(device), masses, ids).cpu().to(torch.float64)
                      for i in torch.arange(len(x)).split(batch_size)])


def metrics(pred: torch.Tensor, bundle: dict) -> dict:
    """NRMSE_dG (the paper metric) plus calibration and discrimination diagnostics."""
    target, ids, labels = bundle['test_oracle_ratio'], bundle['test_context_ids'], bundle['test_labels']
    cal = action_calibration(pred, target, ids)
    return {'nrmse': cal.pairwise_delta_nrmse, 'calibration': asdict(cal),
            'within_context_auroc': within_context_auroc(pred, labels, ids),
            'auroc': binary_auroc(pred, labels), 'oracle_auroc': binary_auroc(target, labels)}


def load_transferred(checkpoint: Path, bundle: dict, method: str = 'leap_global', **arch):
    """Source weights with the target's nominal physics; bodies unseen in training use untrained embedding rows."""
    model = build_model(method, physics_for(bundle), **arch)
    state = {k: v for k, v in torch.load(checkpoint, map_location='cpu', weights_only=True).items()
             if 'innovation_module' not in k}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected or not all('innovation_module' in k for k in missing):
        raise RuntimeError(f'checkpoint mismatch: missing {missing}, unexpected {unexpected}')
    return model

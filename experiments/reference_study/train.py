"""Validation-selected AdamW fits of LEAP, transition NCE and Gaussian MLE."""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import torch

from leap import seed_everything

from .models import AdditiveAction, GaussianMLE, gaussian_nll, leap_loss, transition_nce_loss


@dataclass
class Fit:
    """The selected epoch, its validation loss and the per-epoch losses."""
    best_epoch: int = -1
    best_validation_loss: float = math.inf
    history: list[dict] = field(default_factory=list)


def _state(model) -> dict:
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def fit(model, train_paths, batch_loss, validation_loss, cfg, shuffle_seed: int) -> Fit:
    """AdamW with clipping on batch_loss(paths) -> (loss, logged loss); keeps the best validation epoch."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    shuffle = torch.Generator().manual_seed(shuffle_seed)
    result, best_state, stale = Fit(), _state(model), 0
    for epoch in range(cfg.epochs):
        model.train()
        logged = []
        for idx in torch.randperm(len(train_paths), generator=shuffle).split(cfg.batch_size):
            loss, value = batch_loss(train_paths[idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if not torch.isfinite(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)):
                raise FloatingPointError('non-finite gradient norm')
            optimizer.step()
            logged.append(float(value))
        model.eval()
        with torch.no_grad():
            val = float(validation_loss())
        result.history.append({'epoch': epoch, 'train_loss': sum(logged)/len(logged), 'validation_loss': val})
        if val < result.best_validation_loss-cfg.min_delta:
            result.best_epoch, result.best_validation_loss, best_state, stale = epoch, val, _state(model), 0
        else:
            stale += 1
        if stale >= cfg.patience:
            break
    model.load_state_dict(best_state)
    return result


def _fit_contrastive(objective, offset: int, cfg, data, ref):
    """LEAP or transition NCE plus the residual penalty; fresh references each batch, fixed for validation."""
    seed_everything(cfg.seed+offset+1)
    model = AdditiveAction(cfg)
    sampling = torch.Generator().manual_seed(cfg.seed+offset+211)

    def batch_loss(paths):
        nce = objective(model, paths, ref, cfg.K, sampling)
        return nce+cfg.residual_l2*model.residual(paths[:, :-1]).square().mean(), nce

    def validation_loss():
        return objective(model, data['val'], ref, cfg.K, torch.Generator().manual_seed(cfg.seed+offset+307))

    return model, fit(model, data['train'], batch_loss, validation_loss, cfg, cfg.seed+offset+101)


def _fit_gaussian(cfg, data):
    """Gaussian MLE on the per-transition negative log-likelihood."""
    seed_everything(cfg.seed+3001)
    model = GaussianMLE(cfg)

    def batch_loss(paths):
        loss = gaussian_nll(model, paths)
        return loss, loss

    return model, fit(model, data['train'], batch_loss, lambda: gaussian_nll(model, data['val']), cfg, cfg.seed+3000)


def train_all(cfg, data: dict, ref) -> tuple[dict, dict]:
    """({name: model}, {name: Fit}) for the three estimators; seed offsets follow the paper runs."""
    fitted = {'leap': _fit_contrastive(leap_loss, 1000, cfg, data, ref),
              'transition_nce': _fit_contrastive(transition_nce_loss, 2000, cfg, data, ref),
              'gaussian_mle': _fit_gaussian(cfg, data)}
    return {k: m for k, (m, _) in fitted.items()}, {k: f for k, (_, f) in fitted.items()}

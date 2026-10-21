"""Setwise NCE: identify each selected path among K terminal-window reference paths."""
from __future__ import annotations

import logging

import torch
from torch.utils.data import DataLoader, TensorDataset

from leap import setwise_nce

log = logging.getLogger(__name__)


def score(model, reference, paths: torch.Tensor) -> torch.Tensor:
    """Classifier logit G = -E_theta - log p_ref."""
    return -model.energy(paths)-reference.log_prob(paths)


def train(model, data: torch.Tensor, reference, *, epochs: int, batch_size: int, n_references: int,
          lr: float, weight_decay: float, eta_min: float, grad_clip: float) -> list[float]:
    """AdamW with cosine decay and gradient clipping; returns the mean loss of each epoch."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=eta_min)
    loader = DataLoader(TensorDataset(data), batch_size=batch_size, shuffle=True)
    losses = []
    for epoch in range(epochs):
        total = 0.0
        for (batch,) in loader:
            positive = score(model, reference, batch)
            negative = torch.stack([score(model, reference, reference.sample(len(batch)))
                                    for _ in range(n_references)], 1)
            loss = setwise_nce(positive, negative).loss
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            total += loss.item()
        sched.step()
        losses.append(total/len(loader))
        if epoch % 10 == 0 or epoch == epochs-1:
            log.info('epoch %4d  loss %.6f', epoch, losses[-1])
    return losses

"""Learning the path log ratio with the setwise objective (Eq. 4, Algorithm 1)."""
from __future__ import annotations

import torch

from .objective import setwise_nce


def fit(model: torch.nn.Module, paths: torch.Tensor, sample_reference, *, k: int = 8, epochs: int = 100,
        batch_size: int = 64, lr: float = 1e-3, weight_decay: float = 0.0,
        generator: torch.Generator | None = None) -> list[float]:
    """Train model (paths [..., N+1, d] -> scores [...]) to pick each observed path out of k reference paths.

    paths: [B, N+1, d]. sample_reference(batch, k) returns [b, k, N+1, d], k reference paths in the context of each
    path in the batch (same start, endpoints or selection event). Returns the mean loss of each epoch.
    """
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    history = []
    for _ in range(epochs):
        total = torch.zeros((), device=paths.device)
        for idx in torch.randperm(len(paths), generator=generator).split(batch_size):
            batch = paths[idx]
            loss = setwise_nce(model(batch), model(sample_reference(batch, k))).loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.detach()*len(idx)
        history.append(total.item()/len(paths))
    return history

"""LEAP training objective and reference centering."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class SetwiseOutput:
    loss: torch.Tensor
    accuracy: torch.Tensor
    margin: torch.Tensor


def setwise_nce(positive: torch.Tensor, negative: torch.Tensor) -> SetwiseOutput:
    """Identify one data path among K reference paths of the same context.

    positive: [...], negative: [..., K]. A per-context offset cancels in the softmax.
    """
    if negative.shape[:-1] != positive.shape or negative.shape[-1] < 1:
        raise ValueError(f'negative must be positive.shape + (K,), got {tuple(negative.shape)}')
    logits = torch.cat((positive.unsqueeze(-1), negative), dim=-1).reshape(-1, negative.shape[-1]+1)
    target = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
    return SetwiseOutput(
        loss=F.cross_entropy(logits, target),
        accuracy=(logits.argmax(-1) == 0).float().mean().detach(),
        margin=(positive-negative.logsumexp(-1)).mean().detach(),
    )


def center(correction: torch.Tensor, reference_correction: torch.Tensor) -> torch.Tensor:
    """G~ = G - E_ref[G]: fixes the per-context constant left free by the objective.

    correction: [..., P], reference_correction: [..., M], both for the same contexts.
    """
    if correction.shape[:-1] != reference_correction.shape[:-1]:
        raise ValueError(f'context shapes differ: {tuple(correction.shape)} vs {tuple(reference_correction.shape)}')
    return correction-reference_correction.mean(-1, keepdim=True)


def action(reference_log_prob: torch.Tensor, centered_correction: torch.Tensor) -> torch.Tensor:
    """S = -log p_ref - G~."""
    return -reference_log_prob-centered_correction

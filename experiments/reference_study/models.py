"""The three estimators: LEAP and transition NCE share an additive action; Gaussian MLE."""
from __future__ import annotations

import math

import torch
from torch import nn

from leap import setwise_nce

from . import dynamics


def _mlp(hidden: int, depth: int) -> nn.Sequential:
    layers = [nn.Linear(1, hidden), nn.SiLU()]
    for _ in range(depth-1):
        layers += [nn.Linear(hidden, hidden), nn.SiLU()]
    return nn.Sequential(*layers)


def _zero_head(hidden: int) -> nn.Linear:
    head = nn.Linear(hidden, 1)
    nn.init.zeros_(head.weight)
    nn.init.zeros_(head.bias)
    return head


class AdditiveAction(nn.Module):
    """S = sum_k Gaussian cost of x_{k+1} under drift b_ref + u_theta and the known D, + dt r_theta(x_k)."""

    def __init__(self, cfg):
        super().__init__()
        self.dt, self.D, self.a_ref = cfg.dt, cfg.D_data, cfg.a_ref
        self.features = _mlp(cfg.hidden, cfg.depth)
        self.drift_head = _zero_head(cfg.hidden)
        self.residual_head = _zero_head(cfg.hidden)

    # drift and residual each run the feature net; sharing one pass changes float32 gradients
    def drift(self, x: torch.Tensor) -> torch.Tensor:
        return dynamics.drift(x, self.a_ref)+self.drift_head(self.features(x.unsqueeze(-1))).squeeze(-1)

    def residual(self, x: torch.Tensor) -> torch.Tensor:
        return self.residual_head(self.features(x.unsqueeze(-1))).squeeze(-1)

    def step_action(self, x: torch.Tensor, x_next: torch.Tensor) -> torch.Tensor:
        variance = 2.0*self.D*self.dt
        innovation = x_next-x-self.drift(x)*self.dt
        return 0.5*(innovation.square()/variance+math.log(2.0*math.pi*variance))+self.dt*self.residual(x)

    def action(self, paths: torch.Tensor) -> torch.Tensor:
        return self.step_action(paths[..., :-1], paths[..., 1:]).sum(-1)


class GaussianMLE(nn.Module):
    """Euler likelihood with drift b_ref + u_theta and a learned scalar diffusion, initialized at D_ref."""

    def __init__(self, cfg):
        super().__init__()
        self.dt, self.a_ref = cfg.dt, cfg.a_ref
        self.features = _mlp(cfg.hidden, cfg.depth)
        self.drift_head = _zero_head(cfg.hidden)
        self.log_D = nn.Parameter(torch.tensor(math.log(cfg.D_ref), dtype=torch.float32))

    @property
    def D(self) -> torch.Tensor:
        return self.log_D.clamp(math.log(1e-5), math.log(10.0)).exp()

    def drift(self, x: torch.Tensor) -> torch.Tensor:
        return dynamics.drift(x, self.a_ref)+self.drift_head(self.features(x.unsqueeze(-1))).squeeze(-1)

    def step_log_prob(self, x: torch.Tensor, x_next: torch.Tensor) -> torch.Tensor:
        variance = 2.0*self.D*self.dt
        innovation = x_next-x-self.drift(x)*self.dt
        return -0.5*(innovation.square()/variance+torch.log(2.0*math.pi*variance))

    def action(self, paths: torch.Tensor) -> torch.Tensor:
        return -self.step_log_prob(paths[..., :-1], paths[..., 1:]).sum(-1)


def score(model, paths: torch.Tensor, ref) -> torch.Tensor:
    """Learned log-ratio G = -S_theta - log p_ref of whole paths."""
    return -model.action(paths)-ref.log_prob(paths)


def step_score(model: AdditiveAction, x: torch.Tensor, x_next: torch.Tensor, ref) -> torch.Tensor:
    """Learned log-ratio of single transitions."""
    return -model.step_action(x, x_next)-ref.step_log_prob(x, x_next)


def leap_loss(model: AdditiveAction, paths: torch.Tensor, ref, K: int, generator) -> torch.Tensor:
    """Setwise path NCE: each data path against K reference paths from its own x_0."""
    candidates = torch.cat((paths[:, None], ref.sample(paths[:, 0], K, generator)), dim=1)
    scores = score(model, candidates.reshape(-1, paths.shape[-1]), ref).reshape(-1, K+1)
    return setwise_nce(scores[:, 0], scores[:, 1:]).loss


def transition_nce_loss(model: AdditiveAction, paths: torch.Tensor, ref, K: int, generator) -> torch.Tensor:
    """Setwise NCE per transition: the observed x_{k+1} against K reference draws from the same x_k."""
    x, x_next = paths[:, :-1], paths[:, 1:]
    negatives = ref.sample_next(x, K, generator)
    positive = step_score(model, x, x_next, ref)
    return setwise_nce(positive, step_score(model, x.unsqueeze(-1).expand_as(negatives), negatives, ref)).loss


def gaussian_nll(model: GaussianMLE, paths: torch.Tensor) -> torch.Tensor:
    """Mean negative log-likelihood per transition."""
    return -model.step_log_prob(paths[:, :-1], paths[:, 1:]).mean()

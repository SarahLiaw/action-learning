"""Conditional DDPM over missing frames, adapted from GeoTDM (MIT, see LICENSE).

Kept: linear beta schedule, epsilon prediction, fixed-large variance, MSE loss, learned reference frame.
"""
from __future__ import annotations

import numpy as np
import torch

from .egtn import EGTN


def _extract(arr: np.ndarray, t: torch.Tensor, shape) -> torch.Tensor:
    res = torch.from_numpy(arr).to(device=t.device)[t].float()
    while len(res.shape) < len(shape):
        res = res[..., None]
    return res.expand(shape)


def _sum_flat(x: torch.Tensor) -> torch.Tensor:
    return x.sum(dim=list(range(1, x.ndim)))


def normal_kl(mean1, logvar1, mean2, logvar2):
    return 0.5*(-1.0+logvar2-logvar1+torch.exp(logvar1-logvar2)+((mean1-mean2)**2)*torch.exp(-logvar2))


class GeoTDM:
    """Diffusion over [BN, 3, T_missing] positions relative to a learned per-frame reference."""

    def __init__(self, network: EGTN, num_timesteps: int = 1000):
        self.net = network
        self.ref_frame_cache = None
        scale = 1000/num_timesteps
        betas = np.linspace(scale*0.0001, scale*0.02, num_timesteps, dtype=np.float64)
        self.betas, self.num_timesteps = betas, num_timesteps
        alphas = 1.0-betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0-self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0/self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0/self.alphas_cumprod-1)
        self.posterior_variance = betas*(1.0-self.alphas_cumprod_prev)/(1.0-self.alphas_cumprod)
        self.posterior_log_variance_clipped = np.log(np.append(self.posterior_variance[1], self.posterior_variance[1:]))
        self.posterior_mean_coef1 = betas*np.sqrt(self.alphas_cumprod_prev)/(1.0-self.alphas_cumprod)
        self.posterior_mean_coef2 = (1.0-self.alphas_cumprod_prev)*np.sqrt(alphas)/(1.0-self.alphas_cumprod)

    def reference_frame(self, kw: dict) -> torch.Tensor:
        """Learned reference positions; cached until clear_reference_frame (upstream semantics)."""
        if self.ref_frame_cache is None:
            self.ref_frame_cache = self.net.reference_frame(kw['h'], kw['edge_index'], kw['edge_attr'],
                                                            kw['cond_mask'], kw['x_given'])
        return self.ref_frame_cache

    def clear_reference_frame(self) -> None:
        self.ref_frame_cache = None

    def _denoise(self, x, t, kw):
        return self.net(x, t, kw['h'], kw['edge_index'], kw['edge_attr'], kw['cond_mask'], kw['x_given'])

    def q_sample(self, x0, t, noise):
        return _extract(self.sqrt_alphas_cumprod, t, x0.shape)*x0+_extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape)*noise

    def q_posterior(self, x0, xt, t):
        mean = _extract(self.posterior_mean_coef1, t, xt.shape)*x0+_extract(self.posterior_mean_coef2, t, xt.shape)*xt
        return mean, _extract(self.posterior_log_variance_clipped, t, xt.shape)

    def training_loss(self, x0: torch.Tensor, kw: dict) -> torch.Tensor:
        """Per-node epsilon MSE at one diffusion time per graph: [BN]."""
        noise = torch.randn_like(x0)
        t = torch.randint(0, self.num_timesteps, size=(int(kw['batch'].max()+1),)).to(x0.device)[kw['batch']]
        self.clear_reference_frame()
        x_ref = self.reference_frame(kw)
        self.clear_reference_frame()
        xt = self.q_sample(x0-x_ref, t, noise)+x_ref
        return ((noise-self._denoise(xt, t, kw))**2).mean(dim=[1, 2])

    def vb_term(self, x0: torch.Tensor, xt: torch.Tensor, t: torch.Tensor, kw: dict) -> torch.Tensor:
        """KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t)) summed per node (decoder NLL at t = 0): [BN]."""
        true_mean, true_logvar = self.q_posterior(x0, xt, t)
        x_ref = self.reference_frame(kw)
        x = xt+x_ref
        eps = self._denoise(x, t, kw)
        x = x-x_ref
        logvar = _extract(np.log(np.append(self.posterior_variance[1], self.betas[1:])), t, x.shape)
        pred_x0 = _extract(self.sqrt_recip_alphas_cumprod, t, x.shape)*x-_extract(self.sqrt_recipm1_alphas_cumprod, t, x.shape)*eps
        mean, _ = self.q_posterior(pred_x0, x, t)
        kl = _sum_flat(normal_kl(true_mean, true_logvar, mean, logvar))
        batch, nodes = kw['batch'], kw['num_nodes']
        log_scales = 0.5*logvar
        n_dim = (nodes*x0.size(1)*x0.size(2))[batch]
        constants = n_dim*(log_scales[:, 0, 0]+0.5*np.log(2*np.pi))/nodes[batch]
        decoder_nll = constants+_sum_flat(0.5*((x0-mean)**2)/torch.exp(2*log_scales))
        return torch.where(t == 0, decoder_nll, kl)

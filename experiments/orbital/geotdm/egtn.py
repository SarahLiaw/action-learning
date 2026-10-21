"""Equivariant graph temporal network (EGTN), adapted from GeoTDM (MIT, see LICENSE).

Kept: the configuration used here (SiLU, normalized coordinate updates, learned reference frame).
torch_scatter is replaced by native scatter_add; observed and missing frames are re-interleaved
chronologically (identical to upstream for the frame-0 condition used here).
"""
from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

EPS = 1e-8


def merge_time(x: torch.Tensor) -> torch.Tensor:
    """[m, d, t] -> [m*t, d]."""
    return x.permute(0, 2, 1).reshape(-1, x.shape[1])


def separate_time(x: torch.Tensor, t: int) -> torch.Tensor:
    """[m*t, d] -> [m, d, t]."""
    return x.reshape(-1, t, x.shape[-1]).permute(0, 2, 1)


def scatter(src: torch.Tensor, index: torch.Tensor, size: int, reduce: str) -> torch.Tensor:
    """Sum or mean of src rows grouped by index (dim 0)."""
    out = torch.zeros((size, *src.shape[1:]), dtype=src.dtype, device=src.device)
    out.scatter_add_(0, index.reshape(-1, *((1,)*(src.ndim-1))).expand_as(src), src)
    if reduce == 'sum':
        return out
    count = torch.zeros(size, dtype=src.dtype, device=src.device)
    count.scatter_add_(0, index, torch.ones_like(index, dtype=src.dtype))
    return out/count.clamp_min(1).reshape(size, *((1,)*(src.ndim-1)))


def timestep_embedding(t: torch.Tensor, dim: int, max_positions: int = 10000) -> torch.Tensor:
    half = dim//2
    emb = math.log(max_positions)/(half-1)
    emb = torch.exp(torch.arange(half, dtype=torch.float32, device=t.device)*-emb)
    emb = t.float()[:, None]*emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    return F.pad(emb, (0, 1), mode='constant') if dim % 2 == 1 else emb


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, act: nn.Module, last_act: bool = False):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden), act, nn.Linear(hidden, out_dim)]
        self.actions = nn.Sequential(*layers, *([act] if last_act else []))

    def forward(self, x):
        return self.actions(x)


class ESLayer(nn.Module):
    """Equivariant message passing over bodies, per frame."""

    def __init__(self, hidden: int, edge_dim: int, act: nn.Module):
        super().__init__()
        self.edge_mlp = MLP(2*hidden+1+edge_dim, hidden, hidden, act, last_act=True)
        self.coord_mlp = MLP(hidden, hidden, 1, act)
        self.node_mlp = MLP(2*hidden, hidden, hidden, act)

    def forward(self, x, h, edge_index, edge_attr):
        row, col = edge_index[0], edge_index[1]
        t = x.size(-1)
        x_ij = x[row]-x[col]
        r2 = (x_ij**2).sum(dim=1, keepdim=True)
        m_ij = self.edge_mlp(merge_time(torch.cat((h[row], h[col], r2, edge_attr), dim=1)))
        m_i = scatter(separate_time(m_ij, t), row, x.size(0), 'sum')
        h = separate_time(self.node_mlp(torch.cat((merge_time(h), merge_time(m_i)), dim=-1)), t)
        coord = self.coord_mlp(m_ij)/((merge_time(r2)+EPS).sqrt()+1)
        return x+scatter(separate_time(coord, t)*x_ij, row, x.size(0), 'mean'), h


class ETLayer(nn.Module):
    """Equivariant attention over frames with relative-time embeddings."""

    def __init__(self, hidden: int, act: nn.Module, time_emb_dim: int):
        super().__init__()
        self.q_mlp = MLP(hidden, hidden, hidden, act)
        self.k_mlp = MLP(hidden, hidden, hidden, act)
        self.v_mlp = MLP(hidden, hidden, hidden, act)
        self.x_mlp = MLP(hidden, hidden, 1, act)
        self.time_emb = nn.Linear(time_emb_dim, hidden)
        self.time_emb_dim = time_emb_dim

    def forward(self, x, h):
        t = x.size(-1)
        idx = torch.arange(t).to(x)
        rel = self.time_emb(timestep_embedding((idx.unsqueeze(-1)-idx.unsqueeze(-2)).view(-1), self.time_emb_dim))
        rel = rel.view(t, t, -1).permute(2, 0, 1).unsqueeze(0).repeat(x.size(0), 1, 1, 1)
        q, k, v = self.q_mlp(merge_time(h)), self.k_mlp(merge_time(h)), self.v_mlp(merge_time(h))
        v_x = separate_time(self.x_mlp(v), t)
        k_ts = separate_time(k, t).unsqueeze(-2).repeat(1, 1, t, 1)+rel
        v_ts = separate_time(v, t).unsqueeze(-2).repeat(1, 1, t, 1)+rel
        alpha = F.softmax(torch.einsum('nth,nhts->nts', separate_time(q, t).transpose(-1, -2), k_ts), dim=-1)
        h = h+torch.einsum('nts,nhts->nht', alpha, v_ts)
        x_ts = x.unsqueeze(-1)-x.unsqueeze(-2)
        return x+(alpha.unsqueeze(1)*x_ts*v_x.unsqueeze(-2)).sum(dim=-1), h


class EGTN(nn.Module):
    """Denoiser over missing frames, conditioned on observed frames; also predicts the reference frame."""

    def __init__(self, n_layers: int, node_dim: int, edge_dim: int, hidden: int, time_emb_dim: int,
                 n_layers_ref: int, num_w: int):
        super().__init__()
        act = nn.SiLU()
        self.s_modules, self.t_modules = nn.ModuleList(), nn.ModuleList()
        self.time_emb_dim = time_emb_dim
        self.input_linear = nn.Linear(node_dim+time_emb_dim, hidden)
        for _ in range(n_layers):
            self.s_modules.append(ESLayer(hidden, edge_dim, act))
            self.t_modules.append(ETLayer(hidden, act, time_emb_dim))
        self.input_linear_ref = nn.Linear(node_dim, hidden)
        self.s_modules_ref, self.t_modules_ref = nn.ModuleList(), nn.ModuleList()
        for _ in range(n_layers_ref):
            self.s_modules_ref.append(ESLayer(hidden, edge_dim, act))
            self.t_modules_ref.append(ETLayer(hidden, act, time_emb_dim))
        self.ws = nn.Parameter(torch.ones(num_w))

    def reference_frame(self, h, edge_index, edge_attr, cond_mask, x_given) -> torch.Tensor:
        """Learned per-frame reference positions from the observed frames: [BN, 3, S]."""
        x = x_given[..., cond_mask.view(-1).bool()]
        t = x.size(-1)
        h = separate_time(self.input_linear_ref(merge_time(h.unsqueeze(-1).repeat(1, 1, t))), t)
        edge_attr = edge_attr.unsqueeze(-1).repeat(1, 1, t)
        for s, tm in zip(self.s_modules_ref, self.t_modules_ref):
            x, h = s(x, h, edge_index, edge_attr)
            x, h = tm(x, h)
        wts = h.mean(dim=1, keepdim=True).unsqueeze(-1)*self.ws.view(1, 1, -1).repeat(h.size(0), 1, 1).unsqueeze(-2)
        wts[:, :, -1, :] = 1-wts[:, :, :-1, :].sum(dim=-2)
        return (x.unsqueeze(-1)*wts).sum(dim=-2)

    def forward(self, x, diffusion_t, h, edge_index, edge_attr, cond_mask, x_given):
        """Predicted noise on the missing frames: [BN, 3, T_missing]."""
        cond = cond_mask.view(-1).bool()
        x_input = x
        full = torch.empty(*x.shape[:-1], cond.numel(), dtype=x.dtype, device=x.device)
        full[..., cond] = x_given[..., cond]
        full[..., ~cond] = x
        x, t = full, cond.numel()
        emb = timestep_embedding(diffusion_t, self.time_emb_dim).unsqueeze(-1).repeat(1, 1, t)
        h = torch.cat((h.unsqueeze(-1).repeat(1, 1, t), emb), dim=1)
        h = separate_time(self.input_linear(merge_time(h)), t)
        edge_attr = edge_attr.unsqueeze(-1).repeat(1, 1, t)
        for s, tm in zip(self.s_modules, self.t_modules):
            x, h = s(x, h, edge_index, edge_attr)
            x, h = tm(x, h)
        return x[..., ~cond]-x_input

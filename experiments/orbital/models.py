"""Path scorers for the orbital benchmark.

Every scorer reads normalized paths [B, N+1, M, 6]; the latent mass regime is never an input.
Module names match the original checkpoints.
"""
from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from .dynamics import verlet

PROBE = 1e-4          # relative mass step of the finite-difference sensitivity


class NominalPhysics(nn.Module):
    """Standardized residuals e_k and relative-mass sensitivities J_k under the nominal simulator."""

    def __init__(self, masses_kg: torch.Tensor, scales: torch.Tensor, *, dt: float,
                 position_std: float, velocity_std: float):
        super().__init__()
        self.register_buffer('physical_masses_kg', masses_kg.detach().to(torch.float64))
        self.register_buffer('coordinate_scales', scales.detach().to(torch.float64))
        std = torch.tensor([position_std]*3+[velocity_std]*3, dtype=torch.float64)
        self.register_buffer('scaled_std', std/self.coordinate_scales)
        self.dt = float(dt)

    def _mean(self, current: torch.Tensor, masses: torch.Tensor) -> torch.Tensor:
        physical = current.to(torch.float64)*self.coordinate_scales
        return verlet(physical, masses, self.dt)/self.coordinate_scales

    def residual(self, paths: torch.Tensor) -> torch.Tensor:
        """e_k: [B, N, M, 6]."""
        nominal = self._mean(paths[:, :-1], self.physical_masses_kg)
        return ((paths[:, 1:].to(torch.float64)-nominal)/self.scaled_std).to(paths.dtype)

    def features(self, paths: torch.Tensor, bodies=None):
        """(e_k, J_k): J_k is [B, N, M, 6] for the Sun, or [B, N, M, 6, P] for a list of bodies."""
        current = paths[:, :-1]
        nominal = self._mean(current, self.physical_masses_kg)
        e = (paths[:, 1:].to(torch.float64)-nominal)/self.scaled_std
        sens = []
        for i in ([0] if bodies is None else bodies):
            m = self.physical_masses_kg.clone()
            m[i] *= 1.0+PROBE
            sens.append((self._mean(current, m)-nominal)/PROBE/self.scaled_std)
        j = sens[0] if bodies is None else torch.stack(sens, dim=-1)
        return e.to(paths.dtype), j.to(paths.dtype)


def accumulated_summary(e: torch.Tensor, j: torch.Tensor) -> torch.Tensor:
    """(Z, Z^2, log(1+I), mean_k gamma_k^2/I_k) with gamma_k = e_k.J_k, I_k = |J_k|^2, Z = sum gamma / sqrt(I)."""
    info = j.square().sum(dim=(2, 3))
    gamma = (e*j).sum(dim=(2, 3))
    total = info.sum(dim=1).clamp_min(1e-12)
    z = gamma.sum(dim=1)/total.sqrt()
    local = (gamma.square()/info.clamp_min(1e-12)).mean(dim=1)
    return torch.stack((z, z.square(), torch.log1p(total), local), dim=-1)


def _mlp(in_dim: int, hidden: int, out_dim: int, depth: int = 2) -> nn.Sequential:
    layers, width = [], in_dim
    for _ in range(depth):
        layers += [nn.Linear(width, hidden), nn.SiLU()]
        width = hidden
    return nn.Sequential(*layers, nn.Linear(width, out_dim))


def _log_mass(masses: torch.Tensor, batch: int, steps: int, like: torch.Tensor) -> torch.Tensor:
    """Per-path centered log masses: [B, steps, M, 1]."""
    log_mass = masses.unsqueeze(0).expand(batch, -1).to(like).clamp_min(1e-30).log()
    log_mass = log_mass-log_mass.mean(dim=1, keepdim=True)
    return log_mass[:, None, :, None].expand(-1, steps, -1, -1)


class GraphMessageLayer(nn.Module):
    """All-pairs message passing over bodies, in frame chunks."""

    def __init__(self, node_dim: int, hidden: int, chunk: int):
        super().__init__()
        self.chunk = chunk
        self.edge_mlp = _mlp(9, hidden, hidden)        # dq, dv, log|dq|, receiver and sender log mass
        self.node_mlp = _mlp(node_dim, hidden, hidden)
        self.update_mlp = _mlp(2*hidden, hidden, hidden, depth=1)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, state: torch.Tensor, node_input: torch.Tensor, log_mass: torch.Tensor) -> torch.Tensor:
        """state [F, M, 6], node_input [F, M, D], log_mass [F, M, 1] -> [F, M, H]."""
        frames, bodies, _ = state.shape
        node = self.node_mlp(node_input)
        diag = torch.eye(bodies, dtype=torch.bool, device=state.device)
        scale = math.sqrt(max(bodies-1, 1))
        out = []
        for a in range(0, frames, self.chunk):
            b = min(a+self.chunk, frames)
            q, v = state[a:b, :, :3], state[a:b, :, 3:]
            dq = q[:, None, :, :]-q[:, :, None, :]                 # receiver i, sender j
            dv = v[:, None, :, :]-v[:, :, None, :]
            r = dq.square().sum(dim=-1, keepdim=True).clamp_min(1e-12).sqrt()
            edge = torch.cat((dq, dv, r.log(),
                              log_mass[a:b, :, None, :].expand(-1, -1, bodies, -1),
                              log_mass[a:b, None, :, :].expand(-1, bodies, -1, -1)), dim=-1)
            messages = self.edge_mlp(edge).masked_fill(diag[None, :, :, None], 0.0)
            update = self.update_mlp(torch.cat((node[a:b], messages.sum(dim=2)/scale), dim=-1))
            out.append(self.norm(node[a:b]+update))
        return torch.cat(out, dim=0)


class TransitionEncoder(nn.Module):
    """Per-transition graph features from (x_k, e_k, J_k, log mass, time, body identity)."""

    def __init__(self, physics: NominalPhysics, hidden: int = 96, body_embed: int = 16,
                 max_bodies: int = 64, chunk: int = 64):
        super().__init__()
        self.hidden = hidden
        self.innovation_module = physics
        self.body_embedding = nn.Embedding(max_bodies, body_embed)
        self.graph = GraphMessageLayer(21+body_embed, hidden, chunk)

    def forward(self, paths, masses, body_ids, e, j) -> torch.Tensor:
        """[B, N, M, H]."""
        batch, frames, bodies, _ = paths.shape
        steps = frames-1
        current = paths[:, :-1]
        log_mass = _log_mass(masses, batch, steps, paths)
        embedding = self.body_embedding(body_ids.unsqueeze(0).expand(batch, -1))[:, None].expand(-1, steps, -1, -1)
        time = torch.linspace(0.0, 1.0, steps, device=paths.device, dtype=paths.dtype)
        time = time[None, :, None, None].expand(batch, -1, bodies, -1)
        node = torch.cat((current, e, log_mass, time, 1.0-time, embedding, j), dim=-1)
        out = self.graph(current.reshape(batch*steps, bodies, 6), node.reshape(batch*steps, bodies, -1),
                         log_mass.reshape(batch*steps, bodies, 1))
        return out.reshape(batch, steps, bodies, self.hidden)


class StateEncoder(nn.Module):
    """Graph features of states only (no next state), optionally with the terminal state."""

    def __init__(self, hidden: int = 96, body_embed: int = 16, max_bodies: int = 64,
                 chunk: int = 64, endpoint_aware: bool = False):
        super().__init__()
        self.hidden = hidden
        self.endpoint_aware = endpoint_aware
        self.body_embedding = nn.Embedding(max_bodies, body_embed)
        self.graph = GraphMessageLayer(9+body_embed+(6 if endpoint_aware else 0), hidden, chunk)

    def forward(self, states, masses, body_ids, endpoint=None) -> torch.Tensor:
        """states [B, T, M, 6] -> [B, T, M, H]."""
        batch, steps, bodies, _ = states.shape
        log_mass = _log_mass(masses, batch, steps, states)
        embedding = self.body_embedding(body_ids.unsqueeze(0).expand(batch, -1))[:, None].expand(-1, steps, -1, -1)
        time = torch.linspace(0.0, 1.0, steps, device=states.device, dtype=states.dtype)
        time = time[None, :, None, None].expand(batch, -1, bodies, -1)
        pieces = (states, log_mass, time, 1.0-time, embedding)
        if self.endpoint_aware:
            pieces += (endpoint[:, None]-states,)
        out = self.graph(states.reshape(batch*steps, bodies, 6), torch.cat(pieces, dim=-1).reshape(batch*steps, bodies, -1),
                         log_mass.reshape(batch*steps, bodies, 1))
        return out.reshape(batch, steps, bodies, self.hidden)


class TemporalResidualBlock(nn.Module):
    def __init__(self, hidden: int, dilation: int):
        super().__init__()
        self.conv = nn.Conv1d(hidden, hidden, kernel_size=3, padding=dilation, dilation=dilation)
        self.gate = nn.Conv1d(hidden, hidden, kernel_size=3, padding=dilation, dilation=dilation)
        self.proj = nn.Conv1d(hidden, hidden, kernel_size=1)
        self.norm = nn.GroupNorm(1, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        update = self.conv(x)*torch.sigmoid(self.gate(x))
        return self.norm(x+self.proj(F.silu(update)))


class LeapModel(nn.Module):
    """G_theta = local (sum of transition corrections) and/or global (path score) terms.

    summary: 'solar' adds the accumulated physics summary, 'discover' the same summary for every
    body mass pooled permutation-invariantly, None neither.
    """

    def __init__(self, physics: NominalPhysics, *, local: bool, path: bool, summary: str | None = 'solar',
                 hidden: int = 96, body_embed: int = 16, max_bodies: int = 64, chunk: int = 64,
                 dilations=(1, 2, 4)):
        super().__init__()
        self.has_local, self.has_path, self.summary = local, path, summary if path else None
        self.encoder = TransitionEncoder(physics, hidden, body_embed, max_bodies, chunk)
        if local:
            self.local_head = _mlp(hidden, hidden, 1, depth=1)
        if path:
            self.temporal = nn.Sequential(*(TemporalResidualBlock(hidden, d) for d in dilations))
            self.global_head = _mlp(2*hidden, hidden, 1)
            if self.summary == 'solar':
                self.physics_global_head = _mlp(4, hidden, 1)
            elif self.summary == 'discover':
                self.parameter_encoder = _mlp(4, hidden, hidden)
                self.parameter_pool_head = _mlp(2*hidden, hidden, 1)
        # near-zero initial scores keep finite-K NCE unsaturated
        for name in ('local_head', 'global_head', 'physics_global_head', 'parameter_pool_head'):
            if hasattr(self, name):
                nn.init.normal_(getattr(self, name)[-1].weight, std=1e-3)

    @property
    def physics(self) -> NominalPhysics:
        return self.encoder.innovation_module

    def transition_scores(self, paths, masses, body_ids) -> torch.Tensor:
        """Per-transition corrections: [B, N]."""
        e, j = self.physics.features(paths)
        return self.local_head(self.encoder(paths, masses, body_ids, e, j)).squeeze(-1).sum(dim=2)

    def components(self, paths, masses, body_ids) -> dict[str, torch.Tensor]:
        e, j = self.physics.features(paths)
        h = self.encoder(paths, masses, body_ids, e, j)
        batch, steps, bodies, hidden = h.shape
        out = {}
        if self.has_local:
            out['local'] = self.local_head(h).squeeze(-1).sum(dim=2).sum(dim=1)
        if self.has_path:
            t = self.temporal(h.permute(0, 2, 3, 1).reshape(batch*bodies, hidden, steps))
            t = t.reshape(batch, bodies, hidden, steps)
            pooled = torch.cat((t.mean(dim=(1, 3))*math.sqrt(bodies*steps), t.amax(dim=(1, 3))), dim=-1)
            score = self.global_head(pooled).squeeze(-1)
            if self.summary == 'discover':
                e_all, j_all = self.physics.features(paths, list(range(self.physics.physical_masses_kg.numel())))
                p = self.parameter_encoder(accumulated_summary(e_all.unsqueeze(-1), j_all))
                score = score+self.parameter_pool_head(torch.cat((p.mean(dim=1), p.amax(dim=1)), dim=-1)).squeeze(-1)
            elif self.summary == 'solar':
                score = score+self.physics_global_head(accumulated_summary(e, j)).squeeze(-1)
            out['global'] = score
        return out

    def forward(self, paths, masses, body_ids) -> torch.Tensor:
        return sum(self.components(paths, masses, body_ids).values())


def _zero_init(head: nn.Sequential) -> nn.Sequential:
    nn.init.normal_(head[-1].weight, std=1e-3)
    nn.init.zeros_(head[-1].bias)
    return head


class PathScorer(nn.Module):
    """Body-pooled transition features aggregated over time by a learned sequence model."""

    has_local, has_path = False, True

    def __init__(self, physics: NominalPhysics, *, aggregator: str, summary: bool = False,
                 hidden: int = 96, body_embed: int = 16, max_bodies: int = 64, chunk: int = 64):
        super().__init__()
        self.aggregator, self.use_summary = aggregator, summary
        self.encoder = TransitionEncoder(physics, hidden, body_embed, max_bodies, chunk)
        if aggregator == 'transformer':
            if summary:
                self.physics_head = _zero_init(_mlp(4, hidden, 1))
            layer = nn.TransformerEncoderLayer(hidden, 4, dim_feedforward=4*hidden, batch_first=True,
                                               norm_first=True, dropout=0.0)
            self.transformer = nn.TransformerEncoder(layer, num_layers=3)
        elif aggregator == 'gru':
            self.gru = nn.GRU(hidden, hidden, num_layers=2, batch_first=True, bidirectional=True)
        elif aggregator == 'deepset':
            self.step_mlp = _mlp(hidden, hidden, hidden)
        self.head = _zero_init(_mlp(2*hidden, hidden, 1))

    @property
    def physics(self) -> NominalPhysics:
        return self.encoder.innovation_module

    def _aggregate(self, s: torch.Tensor) -> torch.Tensor:
        n = s.shape[1]
        if self.aggregator == 'transformer':
            pos = torch.arange(n, device=s.device, dtype=s.dtype).unsqueeze(1)/max(n-1, 1)
            freq = torch.arange(s.shape[-1]//2, device=s.device, dtype=s.dtype)
            angle = pos*(2.0*math.pi)*(2.0**freq)
            h = self.transformer(s+torch.cat((angle.sin(), angle.cos()), dim=-1).unsqueeze(0))
            pooled = torch.cat((h.mean(dim=1)*math.sqrt(n), h.amax(dim=1)), dim=-1)
        elif self.aggregator == 'gru':
            pooled = self.gru(s)[0].mean(dim=1)*math.sqrt(n)
        elif self.aggregator == 'deepset':
            h = self.step_mlp(s)
            pooled = torch.cat((h.mean(dim=1)*math.sqrt(n), h.amax(dim=1)), dim=-1)
        else:
            pooled = torch.cat((s[:, 0], s[:, -1]), dim=-1)
        return self.head(pooled).squeeze(-1)

    def components(self, paths, masses, body_ids) -> dict[str, torch.Tensor]:
        e, j = self.physics.features(paths)
        score = self._aggregate(self.encoder(paths, masses, body_ids, e, j).mean(dim=2))
        if self.use_summary:
            score = score+self.physics_head(accumulated_summary(e, j)).squeeze(-1)
        return {'global': score}

    def forward(self, paths, masses, body_ids) -> torch.Tensor:
        return self.components(paths, masses, body_ids)['global']


class GaussianMLE(nn.Module):
    """Diagonal Gaussian likelihood of e_k given x_k (and optionally a causal GRU history)."""

    def __init__(self, physics: NominalPhysics, *, history: bool = False, endpoint_aware: bool = False,
                 hidden: int = 96, body_embed: int = 16, max_bodies: int = 64, chunk: int = 64):
        super().__init__()
        self.history = history
        self.innovation_module = physics
        self.encoder = StateEncoder(hidden, body_embed, max_bodies, chunk, endpoint_aware)
        if history:
            self.gru = nn.GRU(hidden, hidden, batch_first=True)
            self.innovation_history_projection = nn.Linear(6, hidden)
        self.mean_head = _mlp(hidden, hidden, 6, depth=1)
        self.log_std_head = _mlp(hidden, hidden, 6, depth=1)
        for head in (self.mean_head, self.log_std_head):
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

    def distribution(self, paths, masses, body_ids):
        """Predicted mean and log std of e_k: [B, N, M, 6] each."""
        batch, frames, bodies, _ = paths.shape
        endpoint = paths[:, -1] if self.encoder.endpoint_aware else None
        h = self.encoder(paths[:, :-1], masses, body_ids, endpoint)
        if self.history:
            e = self.innovation_module.residual(paths)
            past = torch.zeros_like(e)
            past[:, 1:] = e[:, :-1]
            h = h+self.innovation_history_projection(past)
            h, _ = self.gru(h.permute(0, 2, 1, 3).reshape(batch*bodies, frames-1, -1))
            h = h.reshape(batch, bodies, frames-1, -1).permute(0, 2, 1, 3)
        return self.mean_head(h), self.log_std_head(h).clamp(-16.0, 2.0)

    def forward(self, paths, masses, body_ids) -> torch.Tensor:
        """log p_theta(e_1..e_N | x_0) per path."""
        mean, log_std = self.distribution(paths, masses, body_ids)
        z = (self.innovation_module.residual(paths)-mean)*torch.exp(-log_std)
        return (-0.5*(z.square()+2.0*log_std+math.log(2.0*math.pi))).sum(dim=(2, 3)).sum(dim=1)

    def correction(self, paths, masses, body_ids) -> torch.Tensor:
        """G = log p_theta(e) - log N(e; 0, I), accumulated in float64."""
        mean, log_std = self.distribution(paths, masses, body_ids)
        e, mean, log_std = self.innovation_module.residual(paths).double(), mean.double(), log_std.double()
        return (-0.5*(((e-mean)*(-log_std).exp()).square()-e.square()+2*log_std)).sum(dim=(1, 2, 3))


# name: (constructor kwargs, objective)
METHODS = {
    'leap_local': (dict(kind='leap', local=True, path=False), 'path'),
    'leap_global': (dict(kind='leap', local=False, path=True, summary='solar'), 'path'),
    'leap_hybrid': (dict(kind='leap', local=True, path=True, summary='solar'), 'path'),
    'leap_global_discover': (dict(kind='leap', local=False, path=True, summary='discover'), 'path'),
    'transition_nce': (dict(kind='leap', local=True, path=False), 'transition'),
    'temporal_conv': (dict(kind='leap', local=False, path=True, summary=None), 'path'),
    'hybrid_conv': (dict(kind='leap', local=True, path=True, summary=None), 'path'),
    'transformer': (dict(kind='scorer', aggregator='transformer'), 'path'),
    'gru': (dict(kind='scorer', aggregator='gru'), 'path'),
    'deepset': (dict(kind='scorer', aggregator='deepset'), 'path'),
    'endpoints': (dict(kind='scorer', aggregator='endpoints'), 'path'),
    'categorical_nre_summary': (dict(kind='scorer', aggregator='transformer', summary=True), 'path'),
    'binary_nre': (dict(kind='scorer', aggregator='transformer'), 'binary'),
    'balanced_nre': (dict(kind='scorer', aggregator='transformer'), 'balanced'),
    'binary_nre_summary': (dict(kind='scorer', aggregator='transformer', summary=True), 'binary'),
    'gaussian_mle': (dict(kind='gaussian'), 'mle'),
    'endpoint_gaussian_mle': (dict(kind='gaussian', endpoint_aware=True), 'mle'),
    'history_gaussian_mle': (dict(kind='gaussian', history=True), 'mle'),
}


def build_model(method: str, physics: NominalPhysics, **arch) -> nn.Module:
    """arch: hidden, body_embed, max_bodies, chunk."""
    kw = dict(METHODS[method][0])
    kind = kw.pop('kind')
    cls = {'leap': LeapModel, 'scorer': PathScorer, 'gaussian': GaussianMLE}[kind]
    return cls(physics, **kw, **arch)

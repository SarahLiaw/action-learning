"""Training with fresh, context-matched data and reference simulations every epoch."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from leap import seed_everything, setwise_nce, write_json
from leap.objective import SetwiseOutput

from .dynamics import SolarMassMixture

BALANCE = 100.0        # balanced-NRE penalty weight (Delaunoy et al. 2022)


@dataclass
class TrainConfig:
    epochs: int = 100
    batch_size: int = 8
    K: int = 4
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    patience: int = 20
    global_l2: float = 1e-5          # applied to the path score only
    seed: int = 11
    device: str = 'cuda'
    log_every: int = 5


def simulate(law: SolarMassMixture, scales: torch.Tensor, x0: torch.Tensor, steps: int, generator,
             copies: int = 1) -> torch.Tensor:
    """copies paths per scaled context x0 [B, M, 6]: float64 physics, float32 output with x0 kept exactly."""
    scale = scales.to(device=x0.device, dtype=torch.float64)
    rep = lambda t: t[:, None].expand(-1, copies, -1, -1).reshape(-1, *x0.shape[1:])
    paths = (law.rollout(rep(x0.to(torch.float64)*scale), steps, generator)/scale).to(torch.float32)
    paths[:, 0] = rep(x0)
    return paths


def ratio_loss(model, objective: str, positive: torch.Tensor, negative: torch.Tensor, masses, body_ids) -> SetwiseOutput:
    """positive [B, T, M, 6] against its K context-matched references [B, K, T, M, 6]."""
    batch, frames = positive.shape[:2]
    k = negative.shape[1]
    flat = torch.cat((positive[:, None], negative), dim=1).reshape(batch*(k+1), *positive.shape[1:])
    if objective == 'transition':
        s = model.transition_scores(flat, masses, body_ids).reshape(batch, k+1, frames-1).permute(0, 2, 1)
        return setwise_nce(s[..., 0], s[..., 1:])
    s = model(flat, masses, body_ids).reshape(batch, k+1)
    if objective == 'path':
        return setwise_nce(s[:, 0], s[:, 1:])
    pos, neg = s[:, 0], s[:, 1:].reshape(-1)
    loss = 0.5*(F.binary_cross_entropy_with_logits(pos, torch.ones_like(pos))
                + F.binary_cross_entropy_with_logits(neg, torch.zeros_like(neg)))
    if objective == 'balanced':
        loss = loss+BALANCE*(torch.sigmoid(pos).mean()+torch.sigmoid(neg).mean()-1.0).square()
    accuracy = (pos > 0).float().mean()*0.5+(neg <= 0).float().mean()*0.5
    return SetwiseOutput(loss=loss, accuracy=accuracy, margin=(pos.mean()-neg.mean()).detach())


def _step(model, loss, opt, clip: float, epoch: int) -> None:
    opt.zero_grad(set_to_none=True)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    if not torch.isfinite(norm):
        raise FloatingPointError(f'non-finite gradient norm at epoch {epoch}')
    opt.step()


def _setup(model, bundle, cfg: TrainConfig):
    seed_everything(cfg.seed)
    device = torch.device(cfg.device)
    model.to(device)
    masses = bundle['masses_kg'].to(device=device, dtype=torch.float32)
    return device, masses, bundle['body_ids'].to(device), torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)


def _snapshot(model) -> dict:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def train_ratio(model, objective: str, bundle, data: SolarMassMixture, reference: SolarMassMixture,
                cfg: TrainConfig, out: Path) -> list[dict]:
    """Contrastive training; keeps the best-validation state (saved as best_model.pt)."""
    out.mkdir(parents=True, exist_ok=True)
    device, masses, ids, opt = _setup(model, bundle, cfg)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs, eta_min=cfg.learning_rate*0.02)
    shuffle = torch.Generator(device='cpu').manual_seed(cfg.seed+17)
    sim = torch.Generator(device=device).manual_seed(cfg.seed+31)
    train, val, scales = bundle['train_paths'], bundle['validation_paths'], bundle['scales']
    steps, k = train.shape[1]-1, cfg.K
    l2 = cfg.global_l2 if model.has_path else 0.0

    def pair(x0, gen):
        pos = simulate(data, scales, x0, steps, gen)
        return pos, simulate(reference, scales, pos[:, 0], steps, gen, k).reshape(len(x0), k, *pos.shape[1:])

    history, best, best_state, stale = [], float('inf'), None, 0
    for epoch in range(cfg.epochs):
        model.train()
        t0, tot, n = time.perf_counter(), [0.0, 0.0, 0.0], 0
        for idx in torch.randperm(len(train), generator=shuffle).split(cfg.batch_size):
            pos, neg = pair(train[idx, 0].to(device), sim)
            res = ratio_loss(model, objective, pos, neg, masses, ids)
            loss = res.loss
            if l2 > 0.0:
                loss = loss+l2*model.components(pos, masses, ids)['global'].square().mean()
            _step(model, loss, opt, cfg.grad_clip, epoch)
            tot = [tot[0]+float(res.loss.detach()), tot[1]+float(res.accuracy), tot[2]+float(res.margin)]
            n += 1
        sched.step()
        rec = {'epoch': epoch, 'train_loss': tot[0]/n, 'train_accuracy': tot[1]/n, 'train_margin': tot[2]/n,
               'learning_rate': float(sched.get_last_lr()[0]), 'epoch_seconds': time.perf_counter()-t0}
        model.eval()
        vgen = torch.Generator(device=device).manual_seed(cfg.seed+49_919)
        vl = va = vn = 0.0
        with torch.no_grad():
            for a in range(0, len(val), cfg.batch_size):
                pos, neg = pair(val[a:a+cfg.batch_size, 0].to(device), vgen)
                res = ratio_loss(model, objective, pos, neg, masses, ids)
                vl, va, vn = vl+float(res.loss)*len(pos), va+float(res.accuracy)*len(pos), vn+len(pos)
        rec.update(validation_loss=vl/vn, validation_accuracy=va/vn)
        if rec['validation_loss'] < best:
            best, best_state, stale = rec['validation_loss'], _snapshot(model), 0
            torch.save(best_state, out/'best_model.pt')
        else:
            stale += 1
        history.append(rec)
        if epoch % cfg.log_every == 0 or epoch == cfg.epochs-1:
            print(f"[epoch {epoch:04d}] loss={rec['train_loss']:.4f} acc={rec['train_accuracy']:.3f} "
                  f"val={rec['validation_loss']:.4f} s={rec['epoch_seconds']:.1f}", flush=True)
        if stale >= cfg.patience:
            break
    model.load_state_dict(best_state)
    write_json(out/'history.json', history)
    return history


def train_mle(model, bundle, data: SolarMassMixture, cfg: TrainConfig, out: Path) -> list[dict]:
    """Maximum likelihood on fresh data paths; keeps the best-validation state."""
    out.mkdir(parents=True, exist_ok=True)
    device, masses, ids, opt = _setup(model, bundle, cfg)
    shuffle = torch.Generator(device='cpu').manual_seed(cfg.seed+71)
    sim = torch.Generator(device=device).manual_seed(cfg.seed+31)
    train, val, scales = bundle['train_paths'], bundle['validation_paths'], bundle['scales']
    steps = train.shape[1]-1
    history, best, best_state, stale = [], float('inf'), None, 0
    for epoch in range(cfg.epochs):
        model.train()
        t0, tot, n = time.perf_counter(), 0.0, 0
        for idx in torch.randperm(len(train), generator=shuffle).split(cfg.batch_size):
            paths = simulate(data, scales, train[idx, 0].to(device), steps, sim)
            loss = -model(paths, masses, ids).mean()
            _step(model, loss, opt, cfg.grad_clip, epoch)
            tot, n = tot+float(loss.detach())*len(paths), n+len(paths)
        model.eval()
        vgen = torch.Generator(device=device).manual_seed(cfg.seed+90_031)
        vl = vn = 0.0
        with torch.no_grad():
            for a in range(0, len(val), cfg.batch_size):
                paths = simulate(data, scales, val[a:a+cfg.batch_size, 0].to(device), steps, vgen)
                vl, vn = vl+float(-model(paths, masses, ids).mean())*len(paths), vn+len(paths)
        rec = {'epoch': epoch, 'train_nll': tot/n, 'validation_nll': vl/vn, 'epoch_seconds': time.perf_counter()-t0}
        if rec['validation_nll'] < best:
            best, best_state, stale = rec['validation_nll'], _snapshot(model), 0
            torch.save(best_state, out/'best_model.pt')
        else:
            stale += 1
        history.append(rec)
        if epoch % cfg.log_every == 0 or epoch == cfg.epochs-1:
            print(f"[epoch {epoch:04d}] nll={rec['train_nll']:.4f} val={rec['validation_nll']:.4f}", flush=True)
        if stale >= cfg.patience:
            break
    model.load_state_dict(best_state)
    write_json(out/'history.json', history)
    return history

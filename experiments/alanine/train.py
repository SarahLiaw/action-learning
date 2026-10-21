"""Validation-selected training of the actions (LEAP, binary NRE, velocity) and the conditional generators."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from leap.objective import setwise_nce
from leap.utils import write_json
from .data import delta, embed, observations
from .generators import GENERATORS, TemporalNet, conditioning, cosine_alphas, generator_loss
from .models import PathAction, offset_net
from .reference import cached_bank


def _logged(step, steps, every):
    return step % every == 0 or step+1 == steps


def train_velocity(model, train, val, cfg, out):
    """Regress wrapped one-step increments / dt with the action's drift network."""
    dt, dev = model.dt, train['paths_ab'].device
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    points = lambda d: d['paths_ab'][:, :-1][d['masks_ab'].bool()]
    speed = lambda d: delta(d['paths_ab'][:, :-1], d['paths_ab'][:, 1:])[d['masks_ab'].bool()]/dt
    tx, tv, vx, vv = points(train), speed(train), points(val), speed(val)
    best, hist = float('inf'), []
    for step in range(cfg['steps']):
        ii = torch.randint(len(tx), (cfg['velocity_batch'],), device=dev)
        loss = F.mse_loss(model.drift(tx[ii]), tv[ii])
        opt.zero_grad(); loss.backward(); opt.step()
        if _logged(step, cfg['steps'], cfg['eval_every']):
            with torch.no_grad():
                vl = F.mse_loss(model.drift(vx), vv).item()
            hist.append({'step': step, 'loss': loss.item(), 'validation': vl})
            if vl < best:
                best = vl
                torch.save(model.state_dict(), out/'velocity.pt')
    write_json(out/'velocity_history.json', hist)


def train_contrastive(method, original, train, val, banks, cfg, out, seed):
    """Setwise NCE (LEAP, K references) or binary NRE with a learned initial-state offset, on G = -S - log q."""
    dev = train['paths_ab'].device
    torch.manual_seed(seed)
    model = copy.deepcopy(original)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    offset = offset_net().to(dev)
    if method == 'binary_nre':
        opt.add_param_group({'params': offset.parameters()})

    def scores(split, idx, cols):
        data, bank, logq = split
        neg, nm, nq = (bank[k][idx[:, None], cols] for k in ('paths', 'masks', 'logq'))
        gp = -model(data['paths_ab'][idx], data['masks_ab'][idx])-logq[idx]
        gn = -model(neg.flatten(0, 1), nm.flatten(0, 1)).reshape(len(idx), -1)-nq
        return gp, gn

    def objective(data, idx, gp, gn):
        if method == 'leap':
            return setwise_nce(gp, gn).loss
        shift = offset(embed(data['paths_ab'][idx, 0])).squeeze(-1)
        return F.softplus(-gp-shift).mean()+F.softplus(gn+shift[:, None]).mean()

    best, hist = float('inf'), []
    n, nv = len(train['paths_ab']), len(val['paths_ab'])
    for step in range(cfg['steps']):
        idx = torch.randint(n, (cfg['batch'],), device=dev)
        cols = torch.randint(cfg['bank'], (cfg['batch'], cfg['negatives']), device=dev)
        loss = objective(train, idx, *scores(banks['train'], idx, cols))
        points = train['paths_ab'][idx, :-1][train['masks_ab'][idx].bool()]
        penalty = cfg['residual_penalty']*model.residual(points).square().mean()
        opt.zero_grad(); (loss+penalty).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['clip'])
        opt.step()
        if _logged(step, cfg['steps'], cfg['eval_every']):
            with torch.no_grad():
                vi = torch.arange(nv, device=dev)
                vc = torch.arange(cfg['validation_references'], device=dev)[None].expand(nv, -1)
                vl = objective(val, vi, *scores(banks['validation'], vi, vc)).item()
            hist.append({'step': step, 'loss': loss.item(), 'residual_penalty': penalty.item(), 'validation': vl})
            if vl < best:
                best = vl
                torch.save(model.state_dict(), out/f'{method}.pt')
                torch.save(offset.state_dict(), out/f'{method}_offset.pt')
    write_json(out/f'{method}_history.json', hist)


def train_actions(train, val, ref, cfg, out, seed):
    """Untrained control, velocity regression, then LEAP and binary NRE from the same initialization."""
    out = Path(out)
    diffusion, dt, temperature = float(train['D_eff'][0]), float(train['dt_ps'][0]), float(train['temperature_k'][0])
    torch.manual_seed(seed)
    original = PathAction(diffusion, dt, temperature).to(train['paths_ab'].device)
    torch.save(original.state_dict(), out/'untrained.pt')
    velocity = PathAction(diffusion, dt, temperature, residual=False).to(train['paths_ab'].device)
    velocity.drift_net.load_state_dict(original.drift_net.state_dict())
    train_velocity(velocity, train, val, cfg, out)
    banks = {}
    for split, data, count, offset in (('train', train, cfg['bank'], 1000), ('validation', val, cfg['validation_references'], 2000)):
        refs = cached_bank(out, data, ref, count, seed+offset)
        with torch.no_grad():
            banks[split] = (data, refs, ref.log_prob(data['paths_ab'], data['masks_ab']))
    for method in ('leap', 'binary_nre'):
        train_contrastive(method, original, train, val, banks, cfg, out, seed)


def train_generators(train, val, ref, cfg, out, seed, steps, resume_from=None):
    """Both generators on randomly alternating completion/endpoint masks; resume_from continues saved fits."""
    out = Path(out)
    dev = train['paths_ab'].device
    abar = cosine_alphas(dev)
    tasks = {}
    for split, d in (('train', train), ('validation', val)):
        tasks[split] = []
        for mode in ('completion', 'endpoints'):
            obs = observations(d['paths_ab'], d['masks_ab'], mode)
            tasks[split].append((conditioning(d['paths_ab'], d['masks_ab'], obs, ref), (~obs).float()))

    def loss_for(model, method, d, ii, split):
        cond, free = tasks[split][int(torch.randint(2, ()).item())]
        return generator_loss(model, method, embed(d['paths_ab'][ii]), cond[ii], free[ii], abar)

    for method in GENERATORS:
        torch.manual_seed(seed+20)
        model = TemporalNet().to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
        best, hist, first = float('inf'), [], 0
        if resume_from is not None:
            model.load_state_dict(torch.load(resume_from/f'{method}.pt', map_location=dev, weights_only=True))
            hist = json.loads((resume_from/f'{method}_history.json').read_text())
            best, first = min(r['validation'] for r in hist), hist[-1]['step']+1
            torch.save(model.state_dict(), out/f'{method}.pt')
        for step in range(steps):
            ii = torch.randint(len(train['paths_ab']), (cfg['batch'],), device=dev)
            loss = loss_for(model, method, train, ii, 'train')
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['clip'])
            opt.step()
            if _logged(step, steps, cfg['eval_every']):
                with torch.random.fork_rng(devices=[dev.index or 0] if dev.type == 'cuda' else []):
                    torch.manual_seed(cfg['validation_seed'])
                    with torch.no_grad():
                        vi = torch.arange(len(val['paths_ab']), device=dev)
                        vl = torch.stack([loss_for(model, method, val, vi, 'validation') for _ in range(cfg['validation_repeats'])]).mean().item()
                hist.append({'step': step+first, 'loss': loss.item(), 'validation': vl})
                if vl < best:
                    best = vl
                    torch.save(model.state_dict(), out/f'{method}.pt')
        write_json(out/f'{method}_history.json', hist)

"""Paired GeoTDM models on data and reference paths; their loss differences score paths (Table 12).

Positions only, conditioned on frame 0. Scores use common random numbers across the pair.
"""
from __future__ import annotations

from dataclasses import asdict

import torch

from leap import action_calibration, binary_auroc, seed_everything, write_json

from .. import data as D
from ..train import simulate
from .diffusion import GeoTDM
from .egtn import EGTN


def condition_mask(frames: int, device) -> torch.Tensor:
    mask = torch.zeros(frames, dtype=torch.bool, device=device)
    mask[:1] = True
    return mask.view(1, 1, frames)


def graph(batch: int, masses: torch.Tensor, device) -> dict:
    """Fully connected bodies; node feature = standardized log mass; zero edge features."""
    m = masses.numel()
    nodes = torch.arange(m, device=device)
    row, col = nodes[:, None].expand(-1, m).reshape(-1), nodes[None, :].expand(m, -1).reshape(-1)
    base = torch.stack((row[row != col], col[row != col]), dim=0)
    e = base.shape[1]
    offsets = (torch.arange(batch, device=device)*m).view(batch, 1, 1)
    edge_index = (base.view(1, 2, e)+offsets).permute(1, 0, 2).reshape(2, batch*e)
    log_mass = masses.to(device=device, dtype=torch.float32).log()
    log_mass = (log_mass-log_mass.mean())/log_mass.std().clamp_min(1e-6)
    return {'h': log_mass.view(1, m, 1).expand(batch, -1, -1).reshape(batch*m, 1), 'edge_index': edge_index,
            'edge_attr': torch.zeros(batch*e, 1, device=device),
            'batch': torch.arange(batch, device=device).repeat_interleave(m)}


def positions(paths: torch.Tensor) -> torch.Tensor:
    """[B, T, M, 6] -> [B*M, 3, T]."""
    return paths[..., :3].permute(0, 2, 3, 1).reshape(paths.shape[0]*paths.shape[2], 3, paths.shape[1])


def build(cfg: dict, mask: torch.Tensor, device) -> tuple[GeoTDM, EGTN]:
    net = EGTN(cfg['n_layers'], 1, 1, cfg['hidden'], cfg['time_emb_dim'], cfg['n_layers_ref'],
               int((~mask.view(-1)).sum())).to(device)
    return GeoTDM(net, cfg['timesteps']), net


def batch_loss(model: GeoTDM, paths, masses, mask) -> torch.Tensor:
    """Per-path epsilon MSE averaged over bodies: [B]."""
    b, _, m, _ = paths.shape
    x = positions(paths)
    kw = {**graph(b, masses, x.device), 'cond_mask': mask, 'x_given': x}
    return model.training_loss(x[..., ~mask.view(-1)], kw).reshape(b, m).mean(dim=1)


def train_one(cfg: dict, bundle: dict, law, seed: int, device, masses, mask):
    """Fresh paths from `law` every epoch; last-epoch weights."""
    model, net = build(cfg, mask, device)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg['learning_rate'])
    x0, steps = bundle['train_paths'][:, 0], bundle['train_paths'].shape[1]-1
    gen = torch.Generator(device=device).manual_seed(seed+7717)
    history = []
    for epoch in range(cfg['epochs']):
        paths = simulate(law, bundle['scales'], x0.to(device), steps, gen)
        order = torch.randperm(len(paths), generator=torch.Generator().manual_seed(seed+epoch))
        tot, n = 0.0, 0
        for a in range(0, len(paths), cfg['batch_size']):
            loss = batch_loss(model, paths[order[a:a+cfg['batch_size']]], masses, mask).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            tot, n = tot+float(loss.detach()), n+1
        history.append(tot/max(n, 1))
    return model, net, history


def _reseed(seed: int, device) -> None:
    torch.manual_seed(seed)
    if torch.device(device).type == 'cuda':
        torch.cuda.manual_seed(seed)


def simple_scores(model: GeoTDM, paths, masses, mask, draws: int, seed: int, batch_size: int) -> torch.Tensor:
    """Mean epsilon MSE per path over `draws` (time, noise) draws shared across models."""
    out = []
    for batch in paths.split(batch_size):
        acc = torch.zeros(len(batch), device=batch.device, dtype=torch.float64)
        for d in range(draws):
            _reseed(seed+d, batch.device)
            with torch.no_grad():
                acc += batch_loss(model, batch, masses, mask).double()
        out.append(acc/draws)
    return torch.cat(out)


def vb_scores(model: GeoTDM, paths, masses, mask, draws: int, seed: int, batch_size: int) -> torch.Tensor:
    """T * L_t averaged over draws, as scored in the paper.

    As in the paper's run: targets are diffused without the reference-frame shift, t is drawn per
    node, and the reference frame computed on the first batch is reused (cached) for all batches.
    """
    T = model.num_timesteps
    out = []
    for batch in paths.split(batch_size):
        bs = len(batch)
        x = positions(batch)
        kw = {**graph(bs, masses, x.device), 'cond_mask': mask, 'x_given': x,
              'num_nodes': torch.full((bs,), masses.numel(), device=x.device, dtype=torch.long)}
        target = x[..., ~mask.view(-1)]
        acc = torch.zeros(bs, device=batch.device, dtype=torch.float64)
        for d in range(draws):
            _reseed(seed+d, batch.device)
            t = torch.randint(0, T, (target.shape[0],), device=target.device)
            xt = model.q_sample(target, t, torch.randn_like(target))
            with torch.no_grad():
                term = model.vb_term(target, xt, t, kw)
            acc += (float(T)*term).reshape(bs, -1).mean(dim=1).double()
        out.append(acc/draws)
    return torch.cat(out)


def affine_error(score: torch.Tensor, target: torch.Tensor) -> float:
    """RelErr_aff: residual of one global least-squares scale and offset, over the uncentered target norm."""
    y, s = target.to(torch.float64), score.to(torch.float64)
    sc, yc = s-s.mean(), y-y.mean()
    slope = float((sc*yc).sum()/sc.pow(2).sum().clamp_min(1e-30))
    fitted = slope*s+(y.mean()-slope*s.mean())
    return float(((fitted-y).pow(2).sum()/y.pow(2).sum()).sqrt())


def run_geotdm(cfg: dict, args, seed: int) -> dict:
    device = args.device
    seed_everything(seed)
    bundle = D.load_bundle(args.bundles/cfg['bundle'])
    masses = bundle['masses_kg'].to(device)
    mask = condition_mask(bundle['train_paths'].shape[1], device)
    data, reference = D.laws(bundle['config'], float(bundle['epsilon']))
    model_p, net_p, hist_p = train_one(cfg, bundle, data, seed, device, masses, mask)
    model_q, net_q, hist_q = train_one(cfg, bundle, reference, seed, device, masses, mask)
    run = args.out/cfg['name']/f'seed{seed}'
    run.mkdir(parents=True, exist_ok=True)
    torch.save({'data': net_p.state_dict(), 'ref': net_q.state_dict()}, run/'diffusion_models.pt')
    x, s, bs = bundle['test_candidates'].to(device), seed+991, cfg['eval_batch_size']
    l_p = simple_scores(model_p, x, masses, mask, cfg['draws'], s, bs)
    l_q = simple_scores(model_q, x, masses, mask, cfg['draws'], s, bs)
    v_p = vb_scores(model_p, x, masses, mask, cfg['draws'], s, bs)
    v_q = vb_scores(model_q, x, masses, mask, cfg['draws'], s, bs)
    simple, vb = (l_q-l_p).cpu(), (v_q-v_p).cpu()
    torch.save({'simple': simple, 'variational': vb}, run/'scores.pt')
    target, ids, labels = bundle['test_oracle_ratio'], bundle['test_context_ids'], bundle['test_labels']
    cal, cal_vb = action_calibration(simple, target, ids), action_calibration(vb, target, ids)
    result = {'config': cfg['name'], 'paper': cfg.get('paper'), 'seed': seed,
              'simple': {'nrmse': cal.pairwise_delta_nrmse, 'auroc': binary_auroc(simple, labels),
                         'calibration': asdict(cal)},
              'variational': {'relerr_aff': affine_error(vb, target), 'auroc': binary_auroc(vb, labels),
                              'spearman': cal_vb.spearman, 'calibration': asdict(cal_vb)},
              'oracle_auroc': binary_auroc(target, labels), 'train_loss': {'data': hist_p, 'reference': hist_q}}
    write_json(run/'result.json', result)
    print(f"{cfg['name']} seed {seed}: simple NRMSE {cal.pairwise_delta_nrmse:.4f}, "
          f"VB RelErr_aff {result['variational']['relerr_aff']:.4f}", flush=True)
    return result

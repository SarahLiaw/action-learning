"""Self-contained checks of the reference study: configs, densities, objectives, a pinned run, CLI and report."""
from dataclasses import asdict, replace
import json
import math

import numpy as np
import pytest
import torch

import leap
from experiments.reference_study import dynamics, models, report, sweeps
from experiments.reference_study import run as cli
from experiments.reference_study.reference import Reference

BASE = sweeps.load(sweeps.CONFIGS/'default.json')[0][1]   # paper default, seed 17
TINY = replace(BASE, n_train=24, n_val=12, n_test=10, n_steps=8, K=3, eval_K=3, epochs=4, batch_size=8, hidden=16,
               patience=2)
# The paper's original implementation on replace(TINY, a_ref=0.5); bitwise equal on the original CPU setup.
PINNED = {
    'leap': {'heldout_auroc': 0.8366666666666667, 'within_context_auroc': 0.8333333333333333,
             'nrmse': 0.29164347991597167, 'regret': 0.0, 'best_validation_loss': 1.1647659540176392},
    'transition_nce': {'heldout_auroc': 0.8366666666666667, 'within_context_auroc': 0.8333333333333333,
                       'nrmse': 0.30661189477105677, 'regret': 0.0, 'best_validation_loss': 1.318058967590332},
    'gaussian_mle': {'heldout_auroc': 0.7666666666666667, 'within_context_auroc': 0.8, 'nrmse': 0.9645732704082256,
                     'regret': 0.0, 'best_validation_loss': -0.8768908977508545,
                     'learned_diffusion': 0.19543562829494476},
}
PINNED_ORACLE_AUROC = 0.8633333333333333


def test_configs_define_the_paper_runs():
    members = sweeps.members(sorted(sweeps.CONFIGS.glob('*.json')))
    settings = {name for name, _ in members}
    assert len(members) == 78 and len(settings) == 26   # shared default settings run once
    assert settings == {r['setting'] for panels in report.build({}).values() for p in panels for r in p['rows']}
    assert all(cfg.n_steps*cfg.dt == pytest.approx(0.96) for _, cfg in members)


def test_densities_are_euler():
    cfg = replace(TINY, a_ref=0.5)
    ref = Reference.of(cfg)
    paths = ref.sample(torch.linspace(-1.0, 1.0, 4, dtype=torch.float64), 3, torch.Generator().manual_seed(0))
    paths = paths.reshape(12, -1)
    assert torch.allclose(ref.log_prob(paths), leap.em_log_prob(paths[..., None], ref.drift, cfg.dt, cfg.D_ref))
    assert torch.allclose(dynamics.log_prob(paths, cfg),
                          leap.em_log_prob(paths[..., None], dynamics.drift, cfg.dt, cfg.D_data))


def test_simulated_observation_laws():
    cfg = replace(TINY, n_steps=24)
    clean = dynamics.simulate(4000, cfg, 0)
    innovation = clean[:, 1:]-clean[:, :-1]-dynamics.drift(clean[:, :-1])*cfg.dt
    assert innovation.var().item() == pytest.approx(2*cfg.D_data*cfg.dt, rel=0.02)
    noise = dynamics.simulate(4000, replace(cfg, obs_noise=0.05), 0)-clean   # same latent paths
    assert torch.equal(noise[:, 0], torch.zeros(4000)) and noise[:, 1:].std().item() == pytest.approx(0.05, rel=0.02)
    hidden = dynamics.simulate(4000, replace(cfg, substeps=4), 0)
    assert hidden.shape == clean.shape and torch.equal(hidden[:, 0], clean[:, 0])


def test_objectives_at_initialization():
    """Zero heads and D_ref = D_data: the learned ratio is zero and both NCE losses equal log(K+1)."""
    cfg = replace(TINY, D_ref=TINY.D_data, a_ref=0.5)
    paths, ref = dynamics.simulate(16, cfg, 0), Reference.of(cfg)
    model, generator = models.AdditiveAction(cfg), torch.Generator().manual_seed(0)
    assert torch.equal(models.score(model, paths, ref), torch.zeros(16))
    assert models.leap_loss(model, paths, ref, 3, generator).item() == pytest.approx(math.log(4))
    assert models.transition_nce_loss(model, paths, ref, 3, generator).item() == pytest.approx(math.log(4))
    assert torch.allclose(-models.GaussianMLE(cfg).action(paths), ref.log_prob(paths))


def test_tiny_run_matches_original_values(tmp_path):
    record = cli.run('tiny', replace(TINY, a_ref=0.5), tmp_path)
    for method, expected in PINNED.items():
        m = record['metrics'][method]
        got = {'heldout_auroc': m['heldout_auroc'], 'within_context_auroc': m['within_context_auroc'],
               'nrmse': m['correction']['pairwise_delta_nrmse'], 'regret': m['regret'],
               'best_validation_loss': record['training'][method]['best_validation_loss'],
               'learned_diffusion': m.get('learned_diffusion')}
        for key, value in expected.items():
            np.testing.assert_allclose(got[key], value, rtol=1e-5, atol=1e-7, err_msg=f'{method} {key}')
    np.testing.assert_allclose(record['oracle_heldout_auroc'], PINNED_ORACLE_AUROC, rtol=1e-5)
    assert set(torch.load(tmp_path/'models.pt')) == set(PINNED)


def test_report_cells_and_bold():
    s = lambda mean, sd: {'mean': mean, 'sd': sd, 'n': 3}
    assert report.cell(s(0.66412, 0.13627)) == r'$0.6641\pm0.1363$'
    assert report.cell(s(0.99961, 0.00002)) == r'$0.9996\pm2.0\times10^{-5}$'
    stats = {'leap': s(0.3, 0.1), 'transition_nce': s(0.20004, 0.1), 'gaussian_mle': s(0.19996, 0.1)}
    assert report.latex_row('x', stats, higher=False) == \
        r'x & $0.3000\pm0.1000$ & $\mathbf{0.2000\pm0.1000}$ & $\mathbf{0.2000\pm0.1000}$ \\'
    assert report.latex_row('x', stats, higher=True) == \
        r'x & $\mathbf{0.3000\pm0.1000}$ & $0.2000\pm0.1000$ & $0.2000\pm0.1000$ \\'


def test_cli_runs_and_reports_a_sweep(tmp_path, capsys):
    (tmp_path/'default.json').write_text(json.dumps(
        {'seeds': [17, 29], 'config': {k: v for k, v in asdict(TINY).items() if k != 'seed'}}))
    (tmp_path/'count.json').write_text(json.dumps({'base': 'default.json', 'settings': [{'K': 1}, {'K': 3}]}))
    args = ['--config', str(tmp_path/'count.json'), '--out', str(tmp_path/'results')]
    cli.main(args)
    assert len(list((tmp_path/'results').glob('*/seed*/record.json'))) == 4
    cli.main(args)
    assert capsys.readouterr().out.count('record exists') == 4
    report.main(['--results', str(tmp_path/'results'), '--out', str(tmp_path/'report')])
    summary = json.loads((tmp_path/'report'/'summary.json').read_text())
    rows = {r['setting']: r for p in summary['tables']['sensitivity_budget'] for r in p['rows']}
    assert rows['K=1']['stats']['leap']['n'] == 2 and rows['default']['stats']['gaussian_mle']['n'] == 2
    assert rows['K=2']['stats']['leap'] is None and 'K=2' in summary['missing']
    assert (tmp_path/'report'/'tables.tex').read_text().count(r'\begin{table}') == 3

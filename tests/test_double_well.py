"""Self-contained checks of the double-well package: protocol, conventions and a smoke run."""
import math
from pathlib import Path

import numpy as np
import torch
from scipy import stats

from leap import read_json
from experiments.double_well.baselines import fit_cubic, kramers_moyal, transitions
from experiments.double_well import figure
from experiments.double_well.evaluate import geometric, summarize
from experiments.double_well.inference import analytic_action, optimize_path, true_cost
from experiments.double_well.model import ConservativePotential, untrained_potential
from experiments.double_well.reference import TerminalWindowReference
from experiments.double_well.run import CHECKPOINT, main
from experiments.double_well.simulate import drift, simulate_selected

ROOT = Path(__file__).resolve().parents[1]
PAPER = read_json(ROOT/'configs/double_well/paper.json')
SMOKE = ROOT/'configs/double_well/smoke.json'
SYSTEM, INF = PAPER['system'], PAPER['inference']


def test_config_is_the_paper_protocol():
    assert PAPER['seeds'] == [42, 43, 44]
    assert SYSTEM == dict(diffusion=0.2, dt=0.01, horizon=3.0, start=-1.0, target=1.0, tolerance=0.1)
    assert PAPER['simulation'] == dict(n_paths=5000, round_size=20000)
    assert PAPER['model'] == dict(hidden_dim=512, hidden_layers=3)
    assert PAPER['training'] == dict(epochs=500, batch_size=64, n_references=8, lr=1e-3, weight_decay=1e-5,
                                     eta_min=1e-5, grad_clip=1.0)
    assert INF == dict(endpoints=[-1.0, 0.0], horizons=[1.0, 4.0, 8.0, 16.0], waits=[0.0, 0.25, 0.5, 0.75],
                       lbfgs=dict(max_iter=500, history_size=50, tolerance_grad=1e-7, tolerance_change=1e-10),
                       geometric_crossing=4.0)
    assert PAPER['evaluation'] == dict(low=-1.0, high=1.0, points=301)


def test_selected_paths_start_in_the_well_and_end_in_the_window():
    torch.manual_seed(0)
    paths = simulate_selected(20, **SYSTEM, round_size=4000, device='cpu')
    assert paths.shape == (20, 301, 1) and (paths[:, 0] == -1).all() and ((paths[:, -1]-1).abs() < .1).all()


def test_reference_is_the_window_conditioned_brownian_law():
    torch.manual_seed(0)
    ref = TerminalWindowReference(**SYSTEM, device='cpu')
    paths = ref.sample(20000)
    end = paths[:, -1, 0].double()
    assert (paths[:, 0, 0] == -1).all() and (end >= .9-1e-7).all() and (end <= 1.1+1e-7).all()
    law = stats.truncnorm((.9+1)/math.sqrt(1.2), (1.1+1)/math.sqrt(1.2), loc=-1, scale=math.sqrt(1.2))
    assert abs(end.mean().item()-law.mean()) < 4*law.std()/math.sqrt(end.numel())
    # the bridge at T/2 has mean (x_0 + x_T)/2 and variance 2D (T/2)^2/T
    mid = paths[:, 150, 0].double()-(paths[:, 0, 0]+paths[:, -1, 0]).double()/2
    assert abs(mid.mean().item()) < .02 and abs(mid.var().item()-.3) < .015
    # exact density: Gaussian increments over the window's probability (float32 edges), -inf off the window
    x = paths[:8].double()
    increments = stats.norm(0, math.sqrt(2*.2*.01)).logpdf(np.diff(x.numpy(), axis=1)).sum((1, 2))
    mass = np.diff(stats.norm(-1, math.sqrt(1.2)).cdf(np.float32([.9, 1.1]).astype(np.float64)))[0]
    assert np.allclose(ref.log_prob(x).numpy(), increments-math.log(mass), rtol=1e-12, atol=0)
    x[:4, -1] = 1.2
    assert (ref.log_prob(x)[:4] == -torch.inf).all() and ref.log_prob(x)[4:].isfinite().all()


def test_drift_is_the_potential_gradient_with_its_jacobian():
    torch.manual_seed(0)
    model = ConservativePotential(.2, .01, hidden_dim=16, hidden_layers=2).double().eval()
    x = torch.linspace(-1, 1, 11, dtype=torch.double)[:, None]
    assert torch.allclose(model.drift(x), -(model.U(x+1e-6)-model.U(x-1e-6))/2e-6, atol=1e-6)
    path = torch.randn(2, 9, 1, dtype=torch.double, requires_grad=True)
    assert torch.autograd.gradcheck(lambda p: model.action(p).sum(), (path,), fast_mode=True)
    assert torch.allclose(model.energy(path), model.action(path))


def test_untrained_control_is_seeded_and_leaves_the_generator_alone():
    kwargs = dict(diffusion=.2, dt=.01, hidden_dim=8, hidden_layers=2)
    torch.manual_seed(5)
    before = torch.rand(3)
    torch.manual_seed(5)
    a, b, c = untrained_potential(1, **kwargs), untrained_potential(1, **kwargs), untrained_potential(2, **kwargs)
    assert torch.equal(before, torch.rand(3))
    assert torch.equal(a.U[0].weight, b.U[0].weight) and not torch.equal(a.U[0].weight, c.U[0].weight)


def test_classical_estimators_recover_a_cubic_field():
    assert np.allclose(np.concatenate(transitions(torch.tensor([[[0.], [.1], [.3]]]), .1)), [0, .1, 1, 2])
    rng = np.random.default_rng(0)
    x = rng.uniform(-1, 1, 200_000)
    v = drift(x)+rng.normal(0, .1, x.size)
    assert np.allclose(fit_cubic(x, v), [0, 1, 0, -1], atol=5e-3)
    field, counts = kramers_moyal(x, v, -1., 1., 21)
    assert np.allclose(field[1:-1], drift(np.linspace(-1, 1, 21))[1:-1], atol=.01) and counts[0] < counts[1]


def test_geometric_and_analytic_paths_give_the_paper_costs():
    found, paths = geometric(dt=.01, endpoints=INF['endpoints'], horizons=INF['horizons'], crossing=4.0)
    assert [round(r['true_scaled_cost'], 4) for r in found] == [0.3940, 0.2637, 0.2637, 0.2637]
    assert (paths[16.0][0, :1200] == -1).all() and paths[16.0][0, -1, 0] == 0
    path, value = optimize_path(analytic_action(.01, .2), INF['endpoints'], 1.0, .01, INF['waits'], INF['lbfgs'], 'cpu')
    assert path[0, 0, 0] == -1 and path[0, -1, 0] == 0
    assert round(true_cost(path, .01).item(), 4) == 0.3939 and math.isclose(.2*value.item(), true_cost(path, .01).item(),
                                                                            rel_tol=1e-6)


def test_summarize_reports_mean_and_sample_sd():
    records = [{'seed': s, 'drift_mse': {'leap': m},
                'path_costs': [{'method': 'leap', 'horizon': 1.0, 'true_scaled_cost': c}]}
               for s, m, c in [(0, 1., 3.), (1, 3., 5.)]]
    summary = summarize(records)
    assert summary['drift_mse']['leap'] == {'mean': 2.0, 'sd': math.sqrt(2)}
    assert summary['true_scaled_cost']['leap']['1'] == {'mean': 4.0, 'sd': math.sqrt(2)}


def test_run_and_checkpoint_reuse(tmp_path):
    main(['--config', str(SMOKE), '--out', str(tmp_path/'run')])
    main(['--config', str(SMOKE), '--out', str(tmp_path/'reuse'), '--checkpoints', str(tmp_path/'run')])
    for seed in read_json(SMOKE)['seeds']:
        run, reuse = (read_json(tmp_path/d/f'seed{seed}'/'results.json') for d in ('run', 'reuse'))
        assert (tmp_path/'run'/f'seed{seed}'/CHECKPOINT).exists() and len(run['training_loss']) == 2
        assert run['retrained'] and not reuse['retrained']
        assert run['drift_mse'] == reuse['drift_mse'] and run['path_costs'] == reuse['path_costs']
        assert run['training_paths_sha256'] == reuse['training_paths_sha256']
        methods = {r['method'] for r in run['path_costs']}
        assert methods == {'cubic', 'zero', 'untrained', 'leap', 'analytic', 'geometric'}
        saved = set(np.load(tmp_path/'run'/f'seed{seed}'/'paths.npz'))
        assert saved == {f'{m}_T{h:g}' for m in methods for h in (0.5, 1.0)}
    summary = read_json(tmp_path/'run'/'summary.json')
    assert set(summary['drift_mse']) == {'kramers_moyal', 'cubic', 'zero', 'untrained', 'leap'}
    assert summary['seeds'] == read_json(SMOKE)['seeds']
    figure.plot(read_json(SMOKE), tmp_path/'run', tmp_path/'figure5.pdf')
    assert (tmp_path/'figure5.pdf').stat().st_size > 0

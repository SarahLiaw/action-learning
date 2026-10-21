"""Self-contained checks of experiments.orbital on a synthetic five-body system (no external data)."""
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from leap import action_calibration
from experiments.orbital import data as D, figure
from experiments.orbital.dynamics import G, G_SI, SolarMassMixture, exact_correction, verlet
from experiments.orbital.evaluate import physics_for
from experiments.orbital.evidence import implied_eta, solar_evidence
from experiments.orbital.geotdm import run as geotdm
from experiments.orbital.models import build_model
from experiments.orbital.run import run_train
from experiments.orbital.supervised_features import closed_form, fold_study, selection_study, step_features
from experiments.orbital.tables import collect

HORIZONS_TEXT = """Target body name: Sun (10)
$$SOE
2444239.500000000, A.D. 1980-Jan-01 00:00:00.0000,  7.909787597036500E-03, -6.834276059686257E-04, -2.305858989946859E-04,  3.623738906447844E-06,  7.669871086420855E-06, -1.393174936918079E-07,
2444270.500000000, A.D. 1980-Feb-01 00:00:00.0000,  8.017982129570051E-03, -4.429543007309179E-04, -2.348465605095118E-04,  3.353266721142415E-06,  7.847229994939121E-06, -1.355037398901046E-07,
$$EOE
"""


def synthetic_horizons(path: Path, epochs: int = 16) -> Path:
    """Near-circular orbits of the five heaviest bodies, one state every 30 days."""
    names = D.bodies(5)
    radii = {'sun': 0.0, 'jupiter': 5.2, 'saturn': 9.5, 'neptune': 30.1, 'uranus': 19.2}
    t = np.arange(epochs)*30.0
    states = np.zeros((epochs, 5, 6))
    for j, n in enumerate(names):
        r = radii[n]
        if r:
            w = math.sqrt(G/r**3)
            th = 0.7*j+w*t
            states[:, j, :3] = np.stack([r*np.cos(th), r*np.sin(th), 0.01*r*np.sin(th)], -1)
            states[:, j, 3:] = np.stack([-r*w*np.sin(th), r*w*np.cos(th), 0.01*r*w*np.cos(th)], -1)
    np.savez(path, states_au_au_per_day=states, body_names=np.asarray(names))
    return path


@pytest.fixture(scope='module')
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp('orbital')
    npz = synthetic_horizons(root/'horizons.npz')
    names = D.bodies(5)
    ctx = torch.as_tensor(D.load_states(npz, names)[:6], dtype=torch.float64)
    eta, distance = D.calibrate_step_distance(ctx, D.masses_kg(names))
    cfg = D.BundleConfig(body_count=5, num_steps=8, train_start=0, train_stop=6, validation_start=6,
                         validation_stop=8, test_start=8, test_stop=12, train_paths_per_context=2,
                         validation_paths_per_context=2, test_paths_per_context=4)
    bundle = D.build_bundle(npz, eta, cfg)
    (root/'bundles').mkdir()
    torch.save(bundle, root/'bundles'/'synthetic.pt')
    return SimpleNamespace(root=root, eta=eta, distance=distance, bundle=bundle, names=names)


def test_parse_vectors():
    jd, dates, states = D.parse_vectors(HORIZONS_TEXT)
    assert jd.tolist() == [2444239.5, 2444270.5] and dates[1] == 'A.D. 1980-Feb-01 00:00:00.0000'
    assert states.shape == (2, 6) and states[0, 0] == 7.909787597036500E-03 and states[1, 5] == -1.355037398901046E-07


def test_body_order():
    assert D.bodies(5) == ('sun', 'jupiter', 'saturn', 'neptune', 'uranus')
    assert D.bodies(42)[:31] == D.bodies(31) and len(set(D.bodies(42))) == 42
    assert D.MASS_KG['ceres'] == 62.6284*1e9/G_SI
    m = D.masses_kg(D.bodies(42)[31:])
    assert (np.diff(m) <= 0).all()


def test_calibrated_step_distance(world):
    assert abs(world.distance-0.45) < 1e-6 and 0 < world.eta < 0.2


def test_mixture_density_matches_gaussians(world):
    law = SolarMassMixture.data(D.masses_kg(world.names), world.eta, dt=1.0, position_std=1e-6, velocity_std=1e-6)
    x0 = torch.as_tensor(D.to_barycentric(D.load_states(world.root/'horizons.npz', world.names)[:3],
                                          D.masses_kg(world.names)), dtype=torch.float64)
    paths = law.rollout(x0, 5, torch.Generator().manual_seed(0))
    assert torch.equal(paths[:, 0], x0)
    std = torch.tensor([1e-6]*6, dtype=torch.float64)
    comps = torch.stack([torch.distributions.Normal(verlet(paths[:, :-1], m, 1.0), std).log_prob(paths[:, 1:]).sum((1, 2, 3))
                         for m in law.masses], -1)
    assert torch.allclose(law.log_prob(paths), torch.logsumexp(comps+math.log(0.5), -1), rtol=0, atol=1e-8)
    reference = SolarMassMixture.reference(D.masses_kg(world.names), dt=1.0, position_std=1e-6, velocity_std=1e-6)
    assert torch.allclose(reference.log_prob(paths), reference.component_log_probs(paths)[:, 0], rtol=0, atol=1e-8)


def test_bundle_oracle_is_exact_correction(world):
    b = world.bundle
    assert torch.equal(b['test_oracle_ratio'], b['test_reference_action']-b['test_true_action'])
    data, reference = D.laws(b['config'], world.eta)
    x0 = b['test_candidates'][:, 0].double()*b['scales']
    paths = data.rollout(x0[:4], 8, torch.Generator().manual_seed(1))
    g = exact_correction(data, reference, paths)
    assert torch.allclose(g, data.log_prob(paths)-reference.log_prob(paths), rtol=0, atol=1e-9)


def test_closed_form_approximates_exact_correction(world):
    a, b = step_features(world.bundle)
    pred = closed_form(a.sum(1), b.sum(1))
    assert action_calibration(pred, world.bundle['test_oracle_ratio'], world.bundle['test_context_ids']).pairwise_delta_nrmse < 0.1


def _physics(bodies: int = 31):
    return physics_for({'config': {'dt_days': 1.0, 'position_std_au': 1e-6, 'velocity_std_au_per_day': 1e-6},
                        'masses_kg': torch.ones(bodies, dtype=torch.float64), 'scales': torch.ones(6, dtype=torch.float64)})


@pytest.mark.parametrize('method,hidden,params', [
    ('leap_local', 96, 80_321), ('transition_nce', 96, 80_321), ('leap_global', 96, 303_714),
    ('leap_hybrid', 96, 313_123), ('leap_global_discover', 96, 340_866), ('temporal_conv', 96, 293_825),
    ('hybrid_conv', 96, 303_234), ('transformer', 96, 434_369), ('binary_nre_summary', 96, 444_258),
    ('gru', 96, 377_633), ('deepset', 96, 126_785), ('endpoints', 96, 98_849), ('gaussian_mle', 96, 89_548),
    ('endpoint_gaussian_mle', 96, 90_124), ('history_gaussian_mle', 96, 146_092), ('leap_global', 192, 1_196_226),
    ('leap_global', 384, 4_750_722), ('binary_nre_summary', 192, 1_753_794), ('leap_local', 192, 307_073)])
def test_parameter_counts(method, hidden, params):
    """Counts reported for the paper's models."""
    model = build_model(method, _physics(), hidden=hidden, body_embed=16, max_bodies=64, chunk=64)
    assert sum(p.numel() for p in model.parameters()) == params


@pytest.mark.parametrize('method', ['leap_hybrid', 'transition_nce', 'balanced_nre', 'categorical_nre_summary',
                                    'leap_global_discover', 'history_gaussian_mle'])
def test_train_and_evaluate(world, method, tmp_path):
    cfg = {'name': method, 'bundle': 'synthetic.pt', 'method': method, 'seeds': [3],
           'train': {'epochs': 2, 'batch_size': 4, 'K': 2, 'patience': 5},
           'model': {'hidden': 8, 'body_embed': 4, 'max_bodies': 8, 'chunk': 16}, 'eval_batch_size': 8}
    args = SimpleNamespace(bundles=world.root/'bundles', out=tmp_path, device='cpu')
    result = run_train(cfg, args, 3)
    assert math.isfinite(result['metrics']['nrmse']) and result['epochs_completed'] == 2
    run = tmp_path/method/'seed3'
    assert (run/'best_model.pt').is_file() and len(torch.load(run/'predictions.pt')) == 32
    assert json.loads((run/'result.json').read_text())['seed'] == 3


def test_training_is_deterministic(world, tmp_path):
    cfg = {'name': 'g', 'bundle': 'synthetic.pt', 'method': 'leap_global', 'seeds': [5],
           'train': {'epochs': 1, 'batch_size': 4, 'K': 2},
           'model': {'hidden': 8, 'body_embed': 4, 'max_bodies': 8, 'chunk': 16}, 'eval_batch_size': 8}
    a = run_train(cfg, SimpleNamespace(bundles=world.root/'bundles', out=tmp_path/'a', device='cpu'), 5)
    b = run_train(cfg, SimpleNamespace(bundles=world.root/'bundles', out=tmp_path/'b', device='cpu'), 5)
    assert a['metrics']['nrmse'] == b['metrics']['nrmse']


def test_supervised_features(world):
    r = fold_study(world.bundle, folds=2, seed=0, epochs=5)
    assert all(math.isfinite(r[k]) for k in ('additive_pooled', 'mlp_pooled', 'closed_form'))
    s = selection_study(world.bundle, epochs=5)
    assert set(s) >= {'accumulated_mlp', 'linear_A', 'untrained_mlp'} and len(s['held_out_contexts']) == 2


def test_geotdm_scores(world):
    b = world.bundle
    cfg = dict(epochs=1, batch_size=4, learning_rate=1e-4, hidden=8, n_layers=1, n_layers_ref=1, time_emb_dim=8, timesteps=50)
    mask = geotdm.condition_mask(b['train_paths'].shape[1], 'cpu')
    data, _ = D.laws(b['config'], world.eta)
    torch.manual_seed(0)
    model, _, history = geotdm.train_one(cfg, b, data, 0, 'cpu', b['masses_kg'], mask)
    x = b['test_candidates'][:4]
    simple = geotdm.simple_scores(model, x, b['masses_kg'], mask, 2, 1, 2)
    vb = geotdm.vb_scores(model, x, b['masses_kg'], mask, 2, 1, 2)
    assert len(history) == 1 and torch.isfinite(simple).all() and torch.isfinite(vb).all()
    assert torch.equal(simple, geotdm.simple_scores(model, x, b['masses_kg'], mask, 2, 1, 2))


def test_tables_collect(tmp_path):
    for seed, value in [(11, 0.2), (23, 0.3)]:
        run = tmp_path/'nre_leap_global'/f'seed{seed}'
        run.mkdir(parents=True)
        (run/'result.json').write_text(json.dumps({'seed': seed, 'metrics': {'nrmse': value}}))
    row = collect(tmp_path)['Table 2: trajectory information']['LEAP-Global']
    assert row['n'] == 2 and abs(row['mean']-0.25) < 1e-12 and abs(row['sd']-math.sqrt(0.005)) < 1e-12


def test_implied_eta_recovers_known_eta():
    rng = np.random.default_rng(5)
    ids = np.repeat(np.arange(12), 16)
    score, info = rng.normal(scale=2e4, size=ids.size), rng.uniform(1e8, 3e8, size=ids.size)
    eta = 1e-4
    exact = np.logaddexp(eta*score, -eta*score)-math.log(2)-0.5*eta**2*info
    pred = exact+rng.normal(size=12)[ids]
    assert implied_eta(pred, score, info, ids, eta=2e-4) == pytest.approx(eta, rel=1e-4)


def test_recovered_eta_from_saved_predictions(world, tmp_path):
    b = world.bundle
    run = tmp_path/'nre_leap_global'/'seed11'
    run.mkdir(parents=True)
    torch.save(b['test_oracle_ratio'], run/'predictions.pt')
    (run/'result.json').write_text(json.dumps({'seed': 11, 'metrics': {'nrmse': 0.0}, 'bundle': {'file': 'synthetic.pt'}}))
    score, info = solar_evidence(b)
    want = implied_eta(b['test_oracle_ratio'].numpy(), score, info, b['test_context_ids'].numpy(), world.eta)
    got = collect(tmp_path, world.root/'bundles')['Section 4.2: recovered eta']['LEAP-Global']
    assert got['n'] == 1 and got['mean'] == pytest.approx(want) and got['true_eta'] == pytest.approx(world.eta)


def test_figure(world, tmp_path):
    for config, pred in [('nre_leap_global', world.bundle['test_oracle_ratio']), ('nre_leap_local', torch.zeros(32))]:
        run = tmp_path/config/'seed11'
        run.mkdir(parents=True)
        torch.save(pred, run/'predictions.pt')
        (run/'result.json').write_text(json.dumps({'bundle': {'file': 'synthetic.pt'}}))
    figure.plot(tmp_path, world.root/'bundles', tmp_path/'figure6.pdf')
    assert (tmp_path/'figure6.pdf').stat().st_size > 0


def test_cli_end_to_end(tmp_path, monkeypatch):
    """calibrate -> bundle -> train -> transfer -> supervised -> tables through the command line."""
    from experiments.orbital.run import main
    (tmp_path/'horizons').mkdir()
    synthetic_horizons(tmp_path/'horizons'/'h.npz')
    split = dict(body_count=5, num_steps=6, train_start=0, train_stop=6, validation_start=6, validation_stop=8,
                 test_start=8, test_stop=12, train_paths_per_context=2, validation_paths_per_context=2,
                 test_paths_per_context=4)
    tiny = {'hidden': 8, 'body_embed': 4, 'max_bodies': 8, 'chunk': 16}
    configs = {
        'cal': {'kind': 'calibrate', 'horizons': 'h.npz', 'rule': 'step_distance', 'body_counts': [5],
                'contexts': [0, 6], 'target': 0.45},
        'tiny': {'bundle': 'syn.pt', 'method': 'leap_global', 'seeds': [1, 2], 'train': {'epochs': 1, 'batch_size': 4, 'K': 2},
                 'model': tiny, 'eval_batch_size': 8},
        'tr': {'kind': 'transfer', 'source': 'tiny', 'method': 'leap_global', 'model': tiny,
               'targets': {'syn': 'syn.pt'}, 'eval_batch_size': 8},
        'supervised_features': {'kind': 'supervised', 'bundles': ['syn.pt'], 'folds': 2, 'seed': 0, 'epochs': 3,
                'selection_bundle': 'syn.pt', 'penalty': 1.0},
    }
    for name, cfg in configs.items():
        (tmp_path/f'{name}.json').write_text(json.dumps({'name': name, **cfg}))

    def run(name):
        monkeypatch.setattr('sys.argv', ['run', '--config', str(tmp_path/f'{name}.json'), '--out', str(tmp_path/'out'),
                                         '--bundles', str(tmp_path/'bundles'), '--horizons', str(tmp_path/'horizons'),
                                         '--device', 'cpu'])
        main()
    run('cal')
    eta = json.loads((tmp_path/'out/calibration/cal.json').read_text())['results'][0]['eta']
    (tmp_path/'b.json').write_text(json.dumps({'name': 'b', 'kind': 'bundle', 'file': 'syn.pt', 'horizons': 'h.npz',
                                               'eta': eta, 'bundle': split}))
    for name in ('b', 'tiny', 'tr', 'supervised_features'):
        run(name)
    transfer = json.loads((tmp_path/'out/tr/result.json').read_text())['targets']['syn']
    tiny_runs = [json.loads((tmp_path/f'out/tiny/seed{s}/result.json').read_text())['metrics']['nrmse'] for s in (1, 2)]
    assert transfer['nrmse'] == tiny_runs                 # same bundle: transfer reproduces the native scores
    assert collect(tmp_path/'out')['Additional: supervised features']['N=6']['closed_form'] < 0.2

"""HORIZONS initial states, the solar-mass regimes, eta calibration and dataset bundles.

A bundle holds float32 paths in fixed scaled coordinates plus the exact float64 actions of the
test candidates. Its keys match the original bundles, so either can be loaded.
"""
from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import hashlib
import io
import math
from pathlib import Path
import urllib.parse
import urllib.request

import numpy as np
import torch

from leap import binary_auroc

from .dynamics import G_SI, SolarMassMixture, exact_correction

# name, JPL id, mass (kg), transcribed from solar_system_names.py of github.com/Pablo-Lemos/orbits
CATALOG = (
    ('sun', '10', 1.9885e30), ('mercury', '199', 0.33011e24), ('venus', '299', 4.8685e24),
    ('earth', '399', 5.9724e24), ('moon', '301', 0.0734767309e24), ('mars', '499', 0.64171e24),
    ('jupiter', '599', 1898.19e24), ('io', '501', 0.08931900e24), ('europa', '502', 0.048e24),
    ('ganymede', '503', 0.14819e24), ('callisto', '504', 0.10759e24), ('saturn', '699', 568.34e24),
    ('mimas', '601', 0.000037493e24), ('enceladus', '602', 0.000108022e24), ('tethys', '603', 0.000617449e24),
    ('dione', '604', 0.001095452e24), ('rhea', '605', 0.002306518e24), ('titan', '606', 0.1353452e24),
    ('hyperion', '607', 5.6199e18), ('iapetus', '608', 0.001805635e24), ('phoebe', '609', 8.292e18),
    ('uranus', '799', 86.8103e24), ('ariel', '701', 125100.0e16), ('umbriel', '702', 127500.0e16),
    ('titania', '703', 340000.0e16), ('oberon', '704', 307600.0e16), ('miranda', '705', 6400.0e16),
    ('neptune', '899', 102.41e24), ('triton', '801', 2139000.0e16), ('nereid', '802', 3.0873e19),
    ('proteus', '808', 4.4e19),
)
# name, HORIZONS command, measured GM (km^3/s^2) on 2026-08-31; mass = GM*1e9/G_SI
EXTENDED = (
    ('pluto', '999', 869.326), ('ceres', '1;', 62.6284), ('vesta', '4;', 17.28828),
    ('pallas', '2;', 13.63), ('hygiea', '10;', 7.0), ('interamnia', '704;', 5.0),
    ('psyche', '16;', 1.601), ('camilla', '107;', 0.7475), ('kalliope', '22;', 0.491),
    ('himalia', '506', 0.151552), ('lutetia', '21;', 0.1134),
)
# Sun, giant planets, terrestrial planets, moons by mass; then the extended bodies by mass
_ORDER = ('sun', 'jupiter', 'saturn', 'neptune', 'uranus', 'earth', 'venus', 'mars', 'mercury',
          'ganymede', 'titan', 'callisto', 'io', 'moon', 'europa', 'triton', 'titania', 'oberon',
          'rhea', 'iapetus', 'umbriel', 'ariel', 'dione', 'tethys', 'enceladus', 'miranda',
          'proteus', 'mimas', 'nereid', 'phoebe', 'hyperion')
MASS_KG = {n: m for n, _, m in CATALOG} | {n: gm*1e9/G_SI for n, _, gm in EXTENDED}
BODY_ORDER = _ORDER+tuple(sorted((n for n, _, _ in EXTENDED), key=lambda n: -MASS_KG[n]))
BODY_COUNTS = (5, 10, 20, 31, 42)


def bodies(count: int) -> tuple[str, ...]:
    """Nested mass-prioritized body subset."""
    if count not in BODY_COUNTS:
        raise ValueError(f'body count must be one of {BODY_COUNTS}')
    return BODY_ORDER[:count]


def masses_kg(names) -> np.ndarray:
    return np.asarray([MASS_KG[n] for n in names], dtype=np.float64)


# ---------------------------------------------------------------- HORIZONS

API = 'https://ssd.jpl.nasa.gov/api/horizons.api'


def horizons_query(command: str, start: str, stop: str, step: str) -> str:
    """Barycentric (500@0) ecliptic vectors in AU and AU/day, TDB epochs, no light-time correction."""
    return API+'?'+urllib.parse.urlencode({
        'format': 'text', 'COMMAND': f"'{command}'", 'OBJ_DATA': "'NO'", 'MAKE_EPHEM': "'YES'",
        'EPHEM_TYPE': "'VECTORS'", 'CENTER': "'500@0'", 'START_TIME': f"'{start}'",
        'STOP_TIME': f"'{stop}'", 'STEP_SIZE': f"'{step}'", 'OUT_UNITS': "'AU-D'",
        'REF_PLANE': "'ECLIPTIC'", 'VEC_TABLE': "'2'", 'VEC_CORR': "'NONE'", 'CSV_FORMAT': "'YES'"})


def parse_vectors(text: str):
    """(epochs_jd [T], dates [T], states [T, 6]) from the $$SOE..$$EOE block."""
    a = text.index('$$SOE')
    block = text[text.index('\n', a)+1:text.index('$$EOE', a)]
    jd, dates, states = [], [], []
    for row in csv.reader(io.StringIO(block), skipinitialspace=True):
        f = [x.strip() for x in row if x.strip()]
        if len(f) >= 8:
            jd.append(float(f[0]))
            dates.append(f[1])
            states.append([float(v) for v in f[2:8]])
    return np.asarray(jd, dtype=np.float64), dates, np.asarray(states, dtype=np.float64)


def fetch_vectors(command: str, cache: Path, *, start='1980-01-01', stop='2013-02-01', step='1 month',
                  cache_only: bool = False, timeout: float = 90.0) -> str:
    """Raw HORIZONS response, read from cache when present."""
    if cache.is_file():
        return cache.read_text(encoding='utf-8')
    if cache_only:
        raise FileNotFoundError(cache)
    req = urllib.request.Request(horizons_query(command, start, stop, step),
                                 headers={'User-Agent': 'LEAP-orbital/1.0'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        text = r.read().decode('utf-8')
    parse_vectors(text)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(text, encoding='utf-8')
    return text


def assemble_horizons(root: Path, *, extended: bool = False, cache_only: bool = False, **query) -> Path:
    """Monthly 1980-2013 states of the 31 (or 42) bodies -> root/horizons_{31,42}body_monthly_1980_2013.npz.

    Raw responses are cached in root/raw and fetched only when missing.
    """
    targets = [(n, i, f'{k:02d}_{n}_{i}') for k, (n, i, _) in enumerate(CATALOG, start=1)]
    if extended:
        targets += [(n, c, f'{k:02d}_{n}_{c.rstrip(";")}') for k, (n, c, _) in enumerate(EXTENDED, start=32)]
    epochs = dates = None
    states = []
    for name, command, stem in targets:
        jd, d, s = parse_vectors(fetch_vectors(command, root/'raw'/f'{stem}.txt', cache_only=cache_only, **query))
        if epochs is None:
            epochs, dates = jd, d
        elif not (np.array_equal(jd, epochs) and d == dates):
            raise RuntimeError(f'epoch grid mismatch for {name}')
        states.append(s)
    path = root/f'horizons_{len(targets)}body_monthly_1980_2013.npz'
    np.savez_compressed(path, states_au_au_per_day=np.stack(states, axis=1), epochs_jd_tdb=epochs,
                        calendar_dates=np.asarray(dates), body_names=np.asarray([t[0] for t in targets]),
                        horizons_ids=np.asarray([t[1] for t in targets]),
                        masses_kg=masses_kg([t[0] for t in targets]))
    return path


def load_states(npz: Path, names) -> np.ndarray:
    """HORIZONS states [epoch, body, 6] for the named bodies."""
    with np.load(npz, allow_pickle=False) as z:
        source = [str(v) for v in z['body_names']]
        return np.asarray(z['states_au_au_per_day'], dtype=np.float64)[:, [source.index(n) for n in names]]


def to_barycentric(states: np.ndarray, masses: np.ndarray) -> np.ndarray:
    """Remove the mass-weighted position and velocity centre."""
    w = masses.reshape((1,)*(states.ndim-2)+(states.shape[-2], 1))
    return states-((states*w).sum(axis=-2)/masses.sum())[..., None, :]


def rms_scales(states: np.ndarray, masses: np.ndarray) -> tuple[float, float]:
    """Rotation-invariant RMS position and velocity scales after barycentric centering."""
    c = to_barycentric(states, masses)
    return (max(float(np.sqrt(np.mean(np.square(c[..., :3])))), 1e-12),
            max(float(np.sqrt(np.mean(np.square(c[..., 3:])))), 1e-15))


# ---------------------------------------------------------------- eta calibration

def calibrate_step_distance(contexts: torch.Tensor, masses, *, dt=1.0, position_std=1e-6, velocity_std=1e-6,
                            target=0.45) -> tuple[float, float]:
    """eta whose median one-step Mahalanobis separation of mu_+ and mu_- equals target (canonical)."""
    std = torch.as_tensor([position_std]*3+[velocity_std]*3, dtype=contexts.dtype)

    def distance(eta):
        law = SolarMassMixture.data(masses, eta, dt=dt, position_std=position_std, velocity_std=velocity_std)
        d = ((law.mean(contexts, 1)-law.mean(contexts, 0))/std).square().sum(dim=(-2, -1)).sqrt()
        return float(d.median())
    eta = 1e-4
    for _ in range(3):
        eta *= target/distance(eta)
        if not 0.0 < eta < 0.2:
            raise RuntimeError(f'calibrated eta {eta:.6g} outside (0, 0.2)')
    return eta, distance(eta)


def oracle_auroc(contexts: torch.Tensor, masses, eta: float, *, steps: int, seed: int, dt=1.0,
                 position_std=1e-6, velocity_std=1e-6, paths_per_context=8, device='cpu') -> tuple[float, float]:
    """Exact data-vs-reference AUROC and within-context correction spread at this eta."""
    kw = dict(dt=dt, position_std=position_std, velocity_std=velocity_std)
    data, reference = SolarMassMixture.data(masses, eta, **kw), SolarMassMixture.reference(masses, **kw)
    gen = torch.Generator(device=device).manual_seed(seed)
    x0 = contexts.to(device).repeat_interleave(paths_per_context, dim=0)
    p = data.rollout(x0, steps, gen)
    q = reference.rollout(x0, steps, gen, z=0)
    g = torch.cat((exact_correction(data, reference, p), exact_correction(data, reference, q)))
    labels = torch.cat((torch.ones(len(p)), torch.zeros(len(q))))
    per_context = g[:len(p)].reshape(len(contexts), paths_per_context)
    return (float(binary_auroc(g.cpu(), labels)),
            float((per_context-per_context.mean(dim=1, keepdim=True)).std()))


def calibrate_auroc(contexts: torch.Tensor, masses, *, target=0.7308, tol=2e-3, **kw) -> dict:
    """Log-space bisection of eta so the exact oracle AUROC hits target (matched difficulty)."""
    low, high = 1e-8, 1e-1
    if not oracle_auroc(contexts, masses, low, **kw)[0] <= target <= oracle_auroc(contexts, masses, high, **kw)[0]:
        raise ValueError(f'target AUROC {target} not bracketed')
    for _ in range(34):
        eta = math.sqrt(low*high)
        auc, spread = oracle_auroc(contexts, masses, eta, **kw)
        if abs(auc-target) < tol:
            break
        low, high = (eta, high) if auc < target else (low, eta)
    return {'eta': eta, 'achieved_auroc': auc, 'correction_spread_nats': spread}


# ---------------------------------------------------------------- bundles

@dataclass(frozen=True)
class BundleConfig:
    body_count: int = 31
    num_steps: int = 32
    dt_days: float = 1.0
    position_std_au: float = 1e-6
    velocity_std_au_per_day: float = 1e-6
    train_paths_per_context: int = 4
    validation_paths_per_context: int = 4
    test_paths_per_context: int = 8
    seed: int = 20260726
    train_start: int = 48
    train_stop: int = 302
    validation_start: int = 302
    validation_stop: int = 350
    test_start: int = 350
    test_stop: int = 374


def laws(config: dict, eta: float) -> tuple[SolarMassMixture, SolarMassMixture]:
    """(data, reference) laws of a bundle configuration."""
    kw = dict(dt=config['dt_days'], position_std=config['position_std_au'],
              velocity_std=config['velocity_std_au_per_day'])
    m = masses_kg(bodies(config['body_count']))
    return SolarMassMixture.data(m, eta, **kw), SolarMassMixture.reference(m, **kw)


def build_bundle(npz: Path, eta: float, config: BundleConfig) -> dict:
    """Simulate chronologically disjoint splits; test candidates carry exact float64 actions."""
    names = bodies(config.body_count)
    m = masses_kg(names)
    states = load_states(npz, names)
    split = {k: np.arange(getattr(config, f'{k}_start'), getattr(config, f'{k}_stop'))
             for k in ('train', 'validation', 'test')}
    if len(np.unique(np.concatenate(list(split.values())))) != sum(len(v) for v in split.values()):
        raise ValueError('train, validation and test contexts overlap')
    ctx = {k: torch.as_tensor(to_barycentric(states[v], m), dtype=torch.float64) for k, v in split.items()}
    p_scale, v_scale = rms_scales(ctx['train'].numpy(), m)
    scales = torch.as_tensor([p_scale]*3+[v_scale]*3, dtype=torch.float64)
    data, reference = laws(asdict(config), eta)

    def simulate(law, x, copies, offset, **kw):
        return law.rollout(x.repeat_interleave(copies, dim=0), config.num_steps,
                           torch.Generator(device='cpu').manual_seed(config.seed+offset), **kw)
    train, train_z = simulate(data, ctx['train'], config.train_paths_per_context, 10_003, return_latent=True)
    val, val_z = simulate(data, ctx['validation'], config.validation_paths_per_context, 20_011, return_latent=True)
    test_p, test_z = simulate(data, ctx['test'], config.test_paths_per_context, 30_013, return_latent=True)
    test_q = simulate(reference, ctx['test'], config.test_paths_per_context, 40_009)
    candidates = torch.cat((test_p, test_q))
    with torch.no_grad():
        true_action, reference_action = -data.log_prob(candidates), -reference.log_prob(candidates)
    n = len(test_p)
    labels = torch.cat((torch.ones(n, dtype=torch.int64), torch.zeros(n, dtype=torch.int64)))
    per_context = torch.arange(len(ctx['test'])).repeat_interleave(config.test_paths_per_context)
    oracle = reference_action-true_action
    scale = lambda x: (x/scales).to(torch.float32)
    return {
        'schema_version': 1, 'config': asdict(config), 'epsilon': eta, 'scales': scales,
        'masses_kg': torch.as_tensor(m, dtype=torch.float64), 'body_names': list(names),
        'body_ids': torch.arange(config.body_count, dtype=torch.long),
        'train_paths': scale(train), 'validation_paths': scale(val), 'test_candidates': scale(candidates),
        'test_true_action': true_action, 'test_reference_action': reference_action,
        'test_oracle_ratio': oracle, 'test_labels': labels, 'test_context_ids': torch.cat((per_context, per_context)),
        'train_latent': train_z, 'validation_latent': val_z, 'test_data_latent': test_z,
        'audit': {'source_dataset': Path(npz).name,
                  'source_dataset_sha256': hashlib.sha256(Path(npz).read_bytes()).hexdigest(),
                  'split_indices': {k: v.tolist() for k, v in split.items()},
                  'oracle_data_reference_auroc': binary_auroc(oracle, labels)},
    }


def load_bundle(path) -> dict:
    return torch.load(path, map_location='cpu', weights_only=False)

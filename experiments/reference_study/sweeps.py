"""The paper's reference-study sweeps: the run config and its JSON sweep definitions."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

CONFIGS = Path(__file__).resolve().parents[2]/'configs'/'reference_study'


@dataclass(frozen=True)
class Config:
    """One run: data law, reference, estimators and optimizer."""
    seed: int
    n_train: int
    n_val: int
    n_test: int
    n_steps: int          # observed transitions N
    dt: float             # observed spacing
    D_data: float
    x0_mean: float
    x0_std: float
    substeps: int         # Euler steps per observed interval; 1 observes the chain itself
    obs_noise: float      # sigma_obs on every frame after x_0
    D_ref: float
    a_ref: float          # b_ref = a_ref * b
    K: int                # training references per data path
    eval_K: int           # evaluation references per test context
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    hidden: int
    depth: int
    patience: int
    min_delta: float
    grad_clip: float
    residual_l2: float

    @property
    def exact(self) -> bool:
        """The observed law is the Euler chain, so the exact action is known."""
        return self.substeps == 1 and self.obs_noise == 0.0


def read(path) -> tuple[dict, list[dict], list[int]]:
    """(base values, setting overrides, seeds) of a JSON config; a base file is one setting."""
    path = Path(path)
    spec = json.loads(path.read_text())
    if 'base' not in spec:
        return spec['config'], [{}], spec['seeds']
    base, _, seeds = read(path.parent/spec['base'])
    return base, spec['settings'], seeds


def setting_name(values: dict, base: dict) -> str:
    """'default', or the fields that differ from the base, e.g. 'D_ref=0.06'."""
    return ','.join(f'{k}={values[k]}' for k in base if values[k] != base[k]) or 'default'


def load(path) -> list[tuple[str, Config]]:
    """(setting, config) for every setting and seed of a JSON config."""
    base, settings, seeds = read(path)
    runs = []
    for overrides in settings:
        values = {**base, **overrides}
        runs += [(setting_name(values, base), Config(seed=seed, **values)) for seed in seeds]
    return runs


def members(paths) -> list[tuple[str, Config]]:
    """Unique runs of several JSON configs, sorted by (setting, seed); shared settings run once."""
    runs = {}
    for path in paths:
        for name, cfg in load(path):
            if runs.setdefault((name, cfg.seed), cfg) != cfg:
                raise ValueError(f'{name} seed {cfg.seed} has conflicting configs')
    return [(name, runs[name, seed]) for name, seed in sorted(runs)]

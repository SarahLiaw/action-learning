"""Properties of the shared core."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

import leap


def test_setwise_nce_is_softmax_cross_entropy():
    g = torch.Generator().manual_seed(0)
    pos, neg = torch.randn(6, generator=g), torch.randn(6, 4, generator=g)
    logits = torch.cat((pos[:, None], neg), 1)
    manual = -(logits.log_softmax(1)[:, 0]).mean()
    assert torch.allclose(leap.setwise_nce(pos, neg).loss, manual)


def test_setwise_nce_ignores_context_offsets():
    g = torch.Generator().manual_seed(1)
    pos, neg = torch.randn(6, generator=g), torch.randn(6, 4, generator=g)
    shift = torch.randn(6, generator=g)
    a = leap.setwise_nce(pos, neg).loss
    b = leap.setwise_nce(pos+shift, neg+shift[:, None]).loss
    assert torch.allclose(a, b, atol=1e-6)


def test_center_and_action():
    g = torch.Generator().manual_seed(4)
    paths, ref, log_p = torch.randn(5, 3, generator=g), torch.randn(5, 7, generator=g), torch.randn(5, 3, generator=g)
    centered = leap.center(paths, ref)
    assert torch.allclose(centered, paths-ref.mean(1, keepdim=True))
    assert torch.allclose(leap.center(ref, ref).mean(1), torch.zeros(5), atol=1e-6)
    assert torch.allclose(leap.action(log_p, centered), -log_p-centered)
    with pytest.raises(ValueError):
        leap.center(paths[:, 0], ref)


def test_action_calibration_limits():
    rng = np.random.default_rng(2)
    ctx = np.repeat(np.arange(8), 10)
    true = rng.normal(size=ctx.size)
    offsets = rng.normal(size=8)[ctx]
    assert leap.action_calibration(true+offsets, true, ctx).pairwise_delta_nrmse < 1e-12
    assert np.isclose(leap.action_calibration(offsets, true, ctx).pairwise_delta_nrmse, 1.0)


def test_auroc():
    assert leap.binary_auroc([3, 2, 1, 0], [1, 1, 0, 0]) == 1.0
    assert leap.binary_auroc([0, 1, 2, 3], [1, 1, 0, 0]) == 0.0
    assert leap.binary_auroc([1, 1, 1, 1], [1, 0, 1, 0]) == 0.5


def test_structured_correction_is_euler_log_ratio():
    rng = np.random.default_rng(3)
    paths = torch.tensor(np.cumsum(rng.normal(scale=.06, size=(4, 101, 1)), 1))
    drift = lambda x: x-x**3
    ratio = leap.em_log_prob(paths, drift, .01, .2)-leap.em_log_prob(paths, lambda x: 0*x, .01, .2)
    assert torch.allclose(leap.structured_correction(paths, drift, None, .01, .2), ratio)


class _LinearDrift(torch.nn.Module):
    def __init__(self, dt, diffusion):
        super().__init__()
        self.u, self.dt, self.diffusion = torch.nn.Linear(1, 1), dt, diffusion

    def forward(self, paths):
        return leap.structured_correction(paths, self.u, None, self.dt, self.diffusion)


def test_euler_reference_samples_and_scores_paths():
    ref = leap.EulerReference(0.2, 0.01, drift=lambda x: -x)
    x0 = torch.randn(3, 2, generator=torch.Generator().manual_seed(5))
    paths = ref.sample(x0, 10, k=4, generator=torch.Generator().manual_seed(6))
    assert paths.shape == (3, 4, 11, 2) and torch.equal(paths[:, :, 0], x0[:, None].expand(-1, 4, -1))
    flat = leap.em_log_prob(paths.reshape(12, 11, 2), lambda x: -x, 0.01, 0.2)
    assert torch.allclose(ref.log_prob(paths), flat.reshape(3, 4))


def test_fit_recovers_a_linear_drift():
    g = torch.Generator().manual_seed(7)
    torch.manual_seed(7)
    data = leap.EulerReference(0.2, 0.05, drift=lambda x: -x).sample(4*torch.rand(512, 1, generator=g)-2, 40, generator=g)
    reference = leap.EulerReference(0.2, 0.05)
    model = _LinearDrift(0.05, 0.2)
    leap.fit(model, data[:, 0], lambda batch, k: reference.sample(batch[:, 0], 40, k, generator=g), k=4, epochs=30,
             lr=1e-2, generator=g)
    assert abs(model.u.weight.item()+1) < 0.15 and abs(model.u.bias.item()) < 0.15


def test_optimize_path_keeps_fixed_entries():
    init = torch.zeros(1, 11, 1)
    init[:, -1] = 1.0
    fixed = torch.zeros_like(init, dtype=torch.bool)
    fixed[:, [0, -1]] = True
    path = leap.optimize_path(lambda p: (p.diff(dim=-2)**2).sum((-2, -1)), init, fixed)
    assert torch.equal(path[:, [0, -1]], init[:, [0, -1]])
    assert torch.allclose(path[0, :, 0], torch.linspace(0, 1, 11), atol=1e-4)


def test_custom_system_example_runs():
    spec = importlib.util.spec_from_file_location('custom_system', Path(__file__).parents[1]/'examples'/'custom_system.py')
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    assert all(np.isfinite(v) for v in example.main(n_paths=64, epochs=1).values())

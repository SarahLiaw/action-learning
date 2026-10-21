# Alanine dipeptide

This experiment reproduces Tables 13 to 15, Figure 7 and the geometry check, from Section 4.3 and Appendix F.

We simulate alanine dipeptide in vacuum at 500 K and 600 K and keep its transitions from C7eq to C7ax in the
backbone angles (φ, ψ). LEAP learns an action on these paths at each temperature, and we then keep the learned
networks fixed.

With the 600 K action, we complete paths from sparse observations, add a basin preference, impose an angular
constraint and combine these requests. We also transfer each learned path log ratio to the other temperature by
pairing it with that temperature's reference. Finally, we rebuild representative inferred paths in full atomic
coordinates to check their geometry.

This experiment needs OpenMM, so set up with `uv sync --extra md`.

## Run

Run these in order. Simulation runs on a GPU.

```bash
# 1. Simulate both temperatures
python -m experiments.alanine.run simulate --config configs/alanine/T500.json
python -m experiments.alanine.run simulate --config configs/alanine/T600.json
# 2. Extract reactive paths, splits and drift fields
python -m experiments.alanine.run prepare --config configs/alanine/T500.json
python -m experiments.alanine.run prepare --config configs/alanine/T600.json
# 3. Train LEAP, the controls and the conditional generators
python -m experiments.alanine.run train --config configs/alanine/T500.json
python -m experiments.alanine.run train --config configs/alanine/T600.json
# 4. Completion at both temperatures
python -m experiments.alanine.run infer --config configs/alanine/T500.json
python -m experiments.alanine.run evaluate --config configs/alanine/T500.json
python -m experiments.alanine.run infer --config configs/alanine/T600.json
python -m experiments.alanine.run evaluate --config configs/alanine/T600.json
# 5. Temperature transfer
python -m experiments.alanine.run evaluate --config configs/alanine/transfer.json
# 6. Preference reuse
python -m experiments.alanine.run infer --config configs/alanine/preference.json
python -m experiments.alanine.run evaluate --config configs/alanine/preference.json
# 7. Constraints and composed requests, the geometry check, then the composed table
python -m experiments.alanine.run infer --config configs/alanine/composed.json
python -m experiments.alanine.run audit --config configs/alanine/audit.json
python -m experiments.alanine.run evaluate --config configs/alanine/composed.json
# 8. Figure 7
python -m experiments.alanine.figure --results results/alanine --out results/figures/figure7.pdf
```

`--seed s` trains or infers one seed and `--replica r` simulates one replica, which is handy for SLURM arrays.

## Outputs

| Paper | File in `results/alanine/` |
|---|---|
| Table 13, completion at 600 K | `reuse/T600/summary.json` |
| Table 14, temperature transfer | `transfer/results.json` |
| Table 15, preference reuse | `preference/T600/summary.json` |
| Figure 7 and the values of its paths | `composed/` (paths, free energy and `table.json`) |
| Appendix F.4, geometry check | `audit/<set>/results.json` |

Simulation uses OpenMM's CUDA platform in mixed precision, as in the paper runs. Other platforms draw different
trajectories from the same distribution, so path counts (`data/T<t>/manifest.json`) and metrics differ slightly.

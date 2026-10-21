# Orbital system

This experiment reproduces Tables 2, 3 and 9 to 12, Figure 6 and the recovered mass perturbation, from Section 4.2
and Appendix E.

We simulate 31 bodies of the solar system (42 in the scaling study) from JPL HORIZONS initial conditions. Each path
draws a hidden regime z in {-1, +1} once, and z scales the solar mass by 1 + zη for the whole path.

Because the same z acts on every transition, marginalizing it makes the exact path log ratio depend on the whole
path through the log cosh of a path summary. No sum over transitions can match it, so models that see one
transition at a time cannot capture it, while LEAP-Global follows the exact log ratio.

## Data

Download the initial states and build the datasets on a CPU. Everything lands in `data/orbital/`.

```bash
python -m experiments.orbital.run --config configs/orbital/horizons_31.json
python -m experiments.orbital.run --config configs/orbital/horizons_42.json
python -m experiments.orbital.run --config configs/orbital/bundle_canonical.json
```

The data budget and scaling studies also need the `bundle_ppc*` and `bundle_matched_*` configs. Each bundle config
stores its η, and the `calibrate_*` configs show how it was chosen. `slurm/orbital_cpu.sh` runs any of these as a
CPU job.

## Run

Each training config lists its seeds. These commands train LEAP-Global and LEAP-Local on all of them, collect every
finished run into the paper's tables with the recovered η, and draw Figure 6.

```bash
python -m experiments.orbital.run --config configs/orbital/nre_leap_global.json
python -m experiments.orbital.run --config configs/orbital/nre_leap_local.json
python -m experiments.orbital.tables
python -m experiments.orbital.figure --results results/orbital --out results/figures/figure6.pdf
```

`--seed s` trains one seed. On SLURM, fill in the account and partition in the templates. Then
`mkdir -p logs && sbatch --array=0-2 slurm/orbital_train.sh configs/orbital/nre_leap_global.json` trains one seed
per array task, and `slurm/orbital_job.sh` runs a whole config in one GPU job.

| Paper | Configs |
|---|---|
| Table 2 | `nre_leap_local`, `nre_transition_nce`, `nre_leap_global` |
| Table 3 | `nre_binary`, `nre_binary_summary`, `nre_categorical_summary` |
| Table 9 | the configs of Tables 2 and 3, plus `nre_leap_hybrid`, `nre_gaussian_mle`, `nre_history_gaussian_mle` |
| Figure 6 and the recovered η | `nre_leap_global`, `nre_leap_local` |
| Table 10 | `k4_leap_global`, `k4_leap_global_275`, `k16_leap_global`, `k32_leap_global`, `data_*` |
| Table 11 | `scaling_*` |
| Table 12 | `diffusion_geotdm`, whose network is adapted from [GeoTDM](https://github.com/hanjq17/GeoTDM) (MIT license) |

The other configs run extra studies that are not in the paper.

## Outputs

`results/orbital/tables.json` holds every table (mean and sample SD over seeds) and the recovered η. Each run keeps
its weights, training history, test predictions and metrics in `results/orbital/<config>/seed<s>/`.

GPU training is not bitwise reproducible across hardware, so reruns on other GPUs give slightly different numbers.

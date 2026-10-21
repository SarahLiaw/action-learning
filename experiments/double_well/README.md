# Double well

This experiment reproduces Tables 1 and 6 and Figure 5, from Section 4.1 and Appendix C.

A particle starts at the bottom of the left well, and we keep only the paths that end near the bottom of the right
well. Kramers–Moyal and cubic velocity regression fit one step at a time, so they learn the Doob h-transformed
drift of these selected paths. LEAP compares whole paths and recovers the physical drift.

We then freeze each learned model, move the endpoint to the saddle at x = 0 and find the model's most likely path
at new horizons. We score each path with the true path cost.

## Run

Train and evaluate all three seeds on a GPU, then draw Figure 5.

```bash
python -m experiments.double_well.run --config configs/double_well/paper.json --out results/double_well
python -m experiments.double_well.figure --results results/double_well --out results/figures/figure5.pdf
```

Adding `--checkpoints DIR` to the first command evaluates the training paths and weights saved in `DIR/seed<s>/`
without retraining. To run one seed per SLURM array task, fill in the account and partition in
`slurm/double_well.sh` and run `mkdir -p logs && sbatch slurm/double_well.sh`.

## Outputs

`results/double_well/summary.json` holds Tables 1 and 6 (mean and sample SD over seeds). Each `seed<s>/` keeps the
training paths, the LEAP weights, the cubic fit and the inferred paths.

The paper trained on a GPU. A CPU run draws different paths from the same distribution, so its numbers differ from
the paper's.

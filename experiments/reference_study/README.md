# Reference choice and robustness

This experiment reproduces Tables 7 and 8 and the other results of Appendix D, which Section 4.1 also quotes.

We fit LEAP, Transition NCE and a Gaussian MLE to unconditioned double well paths. We then vary the reference, the
time grid, observation noise, hidden substeps and the number of data and reference paths.

Where the exact path density is known, we measure how well each method recovers the path log ratio. Paths with
observation noise or hidden substeps have no exact density, so there we report held out AUROC.

## Run

This fits all 78 runs on a CPU and then builds the tables.

```bash
python -m experiments.reference_study.run --config configs/reference_study/*.json
python -m experiments.reference_study.report --results results/reference_study --out results/reference_study/report
```

`--list` prints the runs and `--task i` runs only the i-th. On SLURM, fill in the account and partition in
`slurm/reference_study.sh` and run `mkdir -p logs && sbatch slurm/reference_study.sh`. It spreads the runs over 16
cores and then writes the report.

## Outputs

`results/reference_study/report/summary.json` holds every table entry (mean, sample SD and number of seeds) and the
numbers quoted in Appendix D. `tables.tex` has the same entries as LaTeX tables. Its reference table is Table 7,
panel A of its observation table is Table 8, and the rest backs the text of Appendix D.

Each run keeps its fitted models and a record of its metrics and training curves in
`results/reference_study/<setting>/seed<s>/`.

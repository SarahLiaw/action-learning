#!/bin/bash
# All 78 reference study runs (26 settings x 3 seeds), one single-threaded process per core, then the tables.
# CPU only. Submit from the repository root with the environment active:
#   mkdir -p logs && sbatch slurm/reference_study.sh
#SBATCH --job-name=reference-study
#SBATCH --account=<account>
#SBATCH --partition=<partition>
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
CONFIGS=(configs/reference_study/*.json)
N=$(python -m experiments.reference_study.run --config "${CONFIGS[@]}" --list | wc -l)
seq 0 $((N-1)) | xargs -P "${SLURM_CPUS_PER_TASK:-1}" -I{} \
    python -m experiments.reference_study.run --config "${CONFIGS[@]}" --task {}
python -m experiments.reference_study.report --results results/reference_study --out results/reference_study/report

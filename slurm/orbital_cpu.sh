#!/bin/bash
# One CPU-only orbital config: horizons_*, calibrate_canonical, bundle_* or supervised_features.
# Submit from the repository root with the environment active:
#   mkdir -p logs && sbatch slurm/orbital_cpu.sh configs/orbital/bundle_canonical.json
#SBATCH --job-name=orbital-cpu
#SBATCH --account=<account>
#SBATCH --partition=<partition>
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
python -m experiments.orbital.run --config "$1" --out results/orbital --device cpu

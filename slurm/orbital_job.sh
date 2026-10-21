#!/bin/bash
# One orbital config in a single GPU job: a transfer config, or every seed of a train config in sequence.
# Submit from the repository root with the environment active:
#   mkdir -p logs && sbatch slurm/orbital_job.sh configs/orbital/transfer_k4.json
#SBATCH --job-name=orbital
#SBATCH --account=<account>
#SBATCH --partition=<partition>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
python -m experiments.orbital.run --config "$1" --out results/orbital

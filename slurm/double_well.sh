#!/bin/bash
# The double well study, one seed per array task; the last task writes summary.json.
# Submit from the repository root with the environment active:
#   mkdir -p logs && sbatch slurm/double_well.sh
#SBATCH --job-name=double-well
#SBATCH --account=<account>
#SBATCH --partition=<partition>
#SBATCH --array=0-2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
set -euo pipefail
seeds=(42 43 44)
python -m experiments.double_well.run --config configs/double_well/paper.json --out results/double_well \
    --seed "${seeds[$SLURM_ARRAY_TASK_ID]}"

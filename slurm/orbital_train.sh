#!/bin/bash
# One seed of a train or diffusion_geotdm config per array task; the array range is 0 to (number of seeds - 1).
# Submit from the repository root with the environment active:
#   mkdir -p logs && sbatch --array=0-2 slurm/orbital_train.sh configs/orbital/nre_leap_global.json
#SBATCH --job-name=orbital-train
#SBATCH --account=<account>
#SBATCH --partition=<partition>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
set -euo pipefail
CONFIG=$1
SEED=$(python -c "import json, sys; print(json.load(open(sys.argv[1]))['seeds'][int(sys.argv[2])])" "$CONFIG" "$SLURM_ARRAY_TASK_ID")
python -m experiments.orbital.run --config "$CONFIG" --seed "$SEED" --out results/orbital

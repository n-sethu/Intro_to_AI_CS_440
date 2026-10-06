#!/bin/bash
#SBATCH --job-name=ai440-experiments
#SBATCH --array=0-13%4
#SBATCH --output=logs/ai440-experiments-%A_%a.out
#SBATCH --error=logs/ai440-experiments-%A_%a.err
#SBATCH --time=7-00:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"

# Override these with sbatch --export, for example:
# sbatch --export=ALL,TRIALS=5000,SEED=441 run_experiments.slurm
TRIALS="${TRIALS:-500}"
GRID_SIZE="${GRID_SIZE:-75}"
SEED="${SEED:-440}"
MAX_STEPS="${MAX_STEPS:-100000}"
RESULT_DIR="${RESULT_DIR:-${HOME:?}/ai440-results}"
mkdir -p "$RESULT_DIR"
Q_VALUES=(0.20 0.25 0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85)
Q="${Q_VALUES[${SLURM_ARRAY_TASK_ID:-0}]}"
OUTPUT="${OUTPUT:-$RESULT_DIR/bot4-sweep-q-${Q}-${SLURM_JOB_ID:-local}.csv}"

python3 experiments.py \
  --grid-size "$GRID_SIZE" \
  --trials "$TRIALS" \
  --q "$Q" \
  --seed "$SEED" \
  --max-steps "$MAX_STEPS" \
  --csv "$OUTPUT"

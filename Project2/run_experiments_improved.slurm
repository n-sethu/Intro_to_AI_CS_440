#!/bin/bash
#SBATCH --job-name=ai440-improved
#SBATCH --output=Project2/ai440-improved-%A_%a.out
#SBATCH --error=Project2/ai440-improved-%A_%a.err
#SBATCH --time=7-00:00:00
#SBATCH --mem=16G
#SBATCH --ntasks=1
#SBATCH --array=0-7%8

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "$0")/.." && pwd)}"
if [[ ! -f "$REPO_ROOT/Project2/experiments_improved.py" ]]; then
    REPO_ROOT="$(cd -- "$REPO_ROOT/.." && pwd)"
fi
cd "$REPO_ROOT"

# Override these with sbatch --export, for example:
# sbatch --export=ALL,TRIALS=1000,SEED=441 run_experiments_improved.slurm
GRID_SIZE="${GRID_SIZE:-75}"
TRIALS="${TRIALS:-100}"
SEED="${SEED:-440}"
RESULT_DIR="${RESULT_DIR:-$REPO_ROOT/Project2/improved-results}"

# Each task owns disjoint alpha values and its own CSVs.
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
NUM_TASKS="${SLURM_ARRAY_TASK_COUNT:-8}"
RESULT_DIR="$RESULT_DIR/job-${SLURM_ARRAY_JOB_ID:-manual}/shard-$TASK_ID"
mkdir -p "$RESULT_DIR"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

python3 -u Project2/experiments_improved.py \
  --grid-size "$GRID_SIZE" \
  --trials "$TRIALS" \
  --seed "$SEED" \
  --output-dir "$RESULT_DIR" \
  "$@" \
  --shard-index "$TASK_ID" \
  --num-shards "$NUM_TASKS"

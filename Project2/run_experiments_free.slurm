#!/bin/bash
#SBATCH --job-name=ai440-free
#SBATCH --output=Project2/ai440-free-%j.out
#SBATCH --error=Project2/ai440-free-%j.err
#SBATCH --time=7-00:00:00
#SBATCH --mem=16G

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "$0")/.." && pwd)}"
if [[ ! -f "$REPO_ROOT/Project2/experiments_free.py" ]]; then
    REPO_ROOT="$(cd -- "$REPO_ROOT/.." && pwd)"
fi
cd "$REPO_ROOT"

# Override these with sbatch --export, for example:
# sbatch --export=ALL,TRIALS=1000,SEED=441 run_experiments_free.slurm
GRID_SIZE="${GRID_SIZE:-75}"
TRIALS="${TRIALS:-500}"
DETECTOR="${DETECTOR:-both}"
SEED="${SEED:-440}"
RESULT_DIR="${RESULT_DIR:-$REPO_ROOT/Project2/free-results}"

mkdir -p "$RESULT_DIR"

python3 Project2/experiments_free.py \
  --grid-size "$GRID_SIZE" \
  --trials "$TRIALS" \
  --seed "$SEED" \
  --detector "$DETECTOR" \
  --output-dir "$RESULT_DIR" \
  "$@"

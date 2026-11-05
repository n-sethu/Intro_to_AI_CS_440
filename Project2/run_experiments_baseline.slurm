#!/bin/bash
#SBATCH --job-name=ai440-baseline
#SBATCH --output=Project2/ai440-baseline-%j.out
#SBATCH --error=Project2/ai440-baseline-%j.err
#SBATCH --time=7-00:00:00
#SBATCH --mem=16G

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "$0")/.." && pwd)}"
if [[ ! -f "$REPO_ROOT/Project2/experiments_baseline.py" ]]; then
    REPO_ROOT="$(cd -- "$REPO_ROOT/.." && pwd)"
fi
cd "$REPO_ROOT"

# Override these with sbatch --export, for example:
# sbatch --export=ALL,TRIALS=1000,SEED=441 run_experiments_baseline.slurm
GRID_SIZE="${GRID_SIZE:-75}"
TRIALS="${TRIALS:-500}"
SEED="${SEED:-440}"
DETECTOR="${DETECTOR:-both}"
RESULT_DIR="${RESULT_DIR:-$REPO_ROOT/Project2/baseline-results}"

mkdir -p "$RESULT_DIR"

python3 Project2/experiments_baseline.py \
  --grid-size "$GRID_SIZE" \
  --trials "$TRIALS" \
  --seed "$SEED" \
  --detector "$DETECTOR" \
  --output-dir "$RESULT_DIR" \
  "$@"

"""Improved Type B sweep using the baseline scenarios and CSV metrics."""

import argparse
from functools import partial
from pathlib import Path
import sys
from time import perf_counter

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Project2.bots import ImprovedTypeBBot
from Project2.experiments_baseline import (
    DEFAULT_D, DEFAULT_TRIALS, _run_trials, build_scenarios,
    summarize_trials, write_summary_csv, write_trials_csv,
)

DEFAULT_ALPHA_VALUES = (
    tuple(i / 100 for i in range(1, 16))
    + tuple(i / 100 for i in range(20, 61, 5))
)
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "improved-results"


def run_sweep(D=DEFAULT_D, n_trials=DEFAULT_TRIALS, seed=440,
              alpha_values=DEFAULT_ALPHA_VALUES, output_dir=DEFAULT_OUTPUT_DIR,
              move_weight=1.0):
    started = perf_counter()
    print(f"Improved Type B: D={D}, {n_trials} trials per alpha, seed={seed}", flush=True)
    print("Generating scenarios...", flush=True)
    print(f"Planning cost: {move_weight} * moves + 0.5 * scans", flush=True)
    scenarios = build_scenarios(D, n_trials, seed)
    print(f"Scenarios ready in {perf_counter() - started:.1f}s", flush=True)
    rows = []
    results = []
    output_dir = Path(output_dir)
    for alpha in dict.fromkeys(alpha_values):
        rows.extend(_run_trials(
            scenarios, "B", [alpha], n_trials, seed, bot_class=partial(ImprovedTypeBBot, move_weight=move_weight),
            progress_interval=10,
        ))
        results = summarize_trials(rows)
        write_trials_csv(rows, output_dir / "improved_trials.csv")
        write_summary_csv(results, output_dir / "improved_summary.csv")
        latest = results[-1]
        print(f"  successes={latest['successes']}/{latest['trials']}; "
              f"mean cost={latest['discovery_cost_mean']}; "
              f"mean free-movement cost={latest['free_movement_cost_mean']}", flush=True)
    print(f"Saved results to {output_dir.resolve()}; "
          f"total elapsed={perf_counter() - started:.1f}s", flush=True)
    return rows, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-size", type=int, default=DEFAULT_D)
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--seed", type=int, default=440)
    parser.add_argument("--alpha", type=float, nargs="+", default=DEFAULT_ALPHA_VALUES)
    parser.add_argument("--move-weight", type=float, default=1.0,
                        help="Planning cost per move; scans cost 0.5 (default: 1). "
                             "CSV costs still use the baseline scoring formula.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--shard-index", type=int, default=0,
                        help="Zero-based index of this alpha shard.")
    parser.add_argument("--num-shards", type=int, default=1,
                        help="Split alpha values across this many workers.")
    args = parser.parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error("require num-shards >= 1 and 0 <= shard-index < num-shards")
    args.alpha = list(dict.fromkeys(args.alpha))[args.shard_index::args.num_shards]
    if not args.alpha:
        print(f"Shard {args.shard_index}: no alpha values assigned", flush=True)
        sys.exit(0)
    print(f"Shard {args.shard_index}/{args.num_shards}: alpha={args.alpha}", flush=True)
    run_sweep(D=args.grid_size, n_trials=args.trials, seed=args.seed,
              alpha_values=args.alpha, output_dir=args.output_dir,
              move_weight=args.move_weight)

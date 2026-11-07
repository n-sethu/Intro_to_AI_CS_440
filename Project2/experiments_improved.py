"""Improved sweeps (Type A and Type B) on the same seeded scenarios as the baseline.

Type A -> ImprovedTypeABot (information gain per unit cost)
Type B -> ImprovedTypeBBot (rollout-based planning)
"""

import argparse
from functools import partial
from pathlib import Path
import sys
from time import perf_counter

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Project2.bots import ImprovedTypeABot, ImprovedTypeBBot
from Project2.experiments_baseline import (
    DEFAULT_D, DEFAULT_TRIALS, _run_trials, build_scenarios, default_k_values,
    summarize_trials, write_summary_csv, write_trials_csv,
)

DEFAULT_ALPHA_VALUES = (
    tuple(i / 100 for i in range(1, 16))
    + tuple(i / 100 for i in range(20, 61, 5))
)
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "improved-results"


def run_sweep(D=DEFAULT_D, n_trials=DEFAULT_TRIALS, seed=440, k_values=None,
              alpha_values=DEFAULT_ALPHA_VALUES, detector="both",
              output_dir=DEFAULT_OUTPUT_DIR, move_weight=1.0):
    started = perf_counter()
    ks = default_k_values(D) if k_values is None else tuple(dict.fromkeys(k_values))
    alphas = tuple(dict.fromkeys(alpha_values))
    print(f"Improved sweep: D={D}, {n_trials} trials, seed={seed}, detector={detector}", flush=True)
    print(f"Planning cost: {move_weight} * moves + 0.5 * scans", flush=True)
    scenarios = build_scenarios(D, n_trials, seed)
    print(f"Scenarios ready in {perf_counter() - started:.1f}s", flush=True)

    jobs = []
    if detector in ("A", "both"):
        jobs += [("A", k, partial(ImprovedTypeABot, move_weight=move_weight)) for k in ks]
    if detector in ("B", "both"):
        jobs += [("B", a, partial(ImprovedTypeBBot, move_weight=move_weight)) for a in alphas]

    output_dir = Path(output_dir)
    rows, results = [], []
    for kind, value, cls in jobs:
        rows.extend(_run_trials(
            scenarios, kind, [value], n_trials, seed, bot_class=cls,
            progress_interval=10 if kind == "B" else None,
        ))
        results = summarize_trials(rows)
        write_trials_csv(rows, output_dir / "improved_trials.csv")
        write_summary_csv(results, output_dir / "improved_summary.csv")
        latest = results[-1]
        print(f"  {latest['bot']}: successes={latest['successes']}/{latest['trials']}; "
              f"mean cost={latest['discovery_cost_mean']}; "
              f"mean free-movement cost={latest['free_movement_cost_mean']}; "
              f"elapsed={perf_counter() - started:.0f}s", flush=True)
    print(f"Saved results to {output_dir.resolve()}; "
          f"total elapsed={perf_counter() - started:.1f}s", flush=True)
    return rows, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-size", type=int, default=DEFAULT_D)
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--seed", type=int, default=440)
    parser.add_argument("--detector", choices=("A", "B", "both"), default="both")
    parser.add_argument("--k", type=int, nargs="+", help="Override Type A k values")
    parser.add_argument("--alpha", type=float, nargs="+", default=DEFAULT_ALPHA_VALUES)
    parser.add_argument("--move-weight", type=float, default=1.0,
                        help="Planning cost per move; scans cost 0.5 (default: 1). "
                             "CSV costs still use the baseline scoring formula.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    run_sweep(D=args.grid_size, n_trials=args.trials, seed=args.seed, k_values=args.k,
              alpha_values=args.alpha, detector=args.detector,
              output_dir=args.output_dir, move_weight=args.move_weight)
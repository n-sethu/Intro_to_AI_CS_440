# movement is free
# scans cost 0.5 each
# Runs the free-movement bots on the same seeded scenarios as the baseline and improved sweeps so results are paired

import argparse
from pathlib import Path
import sys
from time import perf_counter

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Project2.bots import (
    FreeTypeABot, FreeTypeBBot, ImprovedTypeAFreeBot,
)
from Project2.experiments_baseline import (
    DEFAULT_ALPHA_VALUES, DEFAULT_D, DEFAULT_TRIALS, _run_trials, build_scenarios,
    default_k_values, summarize_trials, write_summary_csv, write_trials_csv,
)

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "free-results"
# FreeTypeABot: entropy-based split.  ImprovedTypeAFreeBot: expected-remaining split.
A_BOTS = {"FreeTypeABot": FreeTypeABot, "ImprovedTypeAFreeBot": ImprovedTypeAFreeBot}
B_BOTS = {"FreeTypeBBot": FreeTypeBBot}


def run_sweep(D=DEFAULT_D, n_trials=DEFAULT_TRIALS, seed=440, k_values=None,
              alpha_values=None, detector="both", output_dir=DEFAULT_OUTPUT_DIR):
    started = perf_counter()
    ks = default_k_values(D) if k_values is None else tuple(dict.fromkeys(k_values))
    alphas = DEFAULT_ALPHA_VALUES if alpha_values is None else tuple(dict.fromkeys(alpha_values))
    scenarios = build_scenarios(D, n_trials, seed)
    jobs = []
    if detector in ("A", "both"):
        jobs += [("A", k, cls) for k in ks for cls in A_BOTS.values()]
    if detector in ("B", "both"):
        jobs += [("B", a, cls) for a in alphas for cls in B_BOTS.values()]
    rows = []
    for kind, value, cls in jobs:
        rows.extend(_run_trials(scenarios, kind, [value], n_trials, seed, bot_class=cls))
        results = summarize_trials(rows)
        write_trials_csv(rows, Path(output_dir) / "free_trials.csv")
        write_summary_csv(results, Path(output_dir) / "free_summary.csv")
        latest = next(r for r in reversed(results)
                      if r["detector"] == kind and float(r["k" if kind == "A" else "alpha"]) == value)
        print(f"  {latest['bot']}: successes={latest['successes']}/{latest['trials']}; "
              f"mean free cost={latest['free_movement_cost_mean']}; "
              f"elapsed={perf_counter() - started:.0f}s", flush=True)
    print(f"Saved results to {Path(output_dir).resolve()}", flush=True)
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-size", type=int, default=DEFAULT_D)
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--seed", type=int, default=440)
    parser.add_argument("--detector", choices=("A", "B", "both"), default="both")
    parser.add_argument("--k", type=int, nargs="+")
    parser.add_argument("--alpha", type=float, nargs="+")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    a = parser.parse_args()
    run_sweep(a.grid_size, a.trials, a.seed, a.k, a.alpha, a.detector, a.output_dir)
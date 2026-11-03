"""Reproducible baseline sweeps; run this file with --help for options."""

import argparse
import csv
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
import random
import sys
from time import perf_counter

import numpy as np

# Allow both direct execution and importing as Project2.experiments_baseline.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Project1.ship import ShipGrid
from Project2.bots import BaselineTypeABot, BaselineTypeBBot


DEFAULT_D = 75
DEFAULT_TRIALS = 500
DEFAULT_ALPHA_VALUES = (0.01,) + tuple(step / 100 for step in range(5, 61, 5))
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "baseline-results"
TRIAL_FIELDS = [
    "scenario_id", "seed", "layout_seed", "bot_seed", "grid_size", "open_cells",
    "start_row", "start_col", "lost_row", "lost_col", "initial_path_distance",
    "initial_euclidean_distance", "bot", "detector", "k", "alpha", "found",
    "status", "error", "move_actions", "detector_actions", "discovery_cost",
    "free_movement_cost", "moves_over_shortest_path", "elapsed_seconds",
]
METRICS = (
    "discovery_cost", "free_movement_cost", "move_actions", "detector_actions",
    "moves_over_shortest_path", "elapsed_seconds",
)
STAT_NAMES = ("mean", "std", "sem", "ci95_low", "ci95_high", "median", "min", "max")
SUMMARY_FIELDS = [
    "bot", "detector", "grid_size", "k", "alpha", "trials", "successes",
    "failures", "success_rate",
] + [f"{metric}_{stat}" for metric in METRICS for stat in STAT_NAMES]


@dataclass(frozen=True)
class Scenario:
    scenario_id: int
    seed: int
    layout_seed: int
    bot_seed: int
    ship: object
    start: tuple
    lost_bot: tuple
    initial_path_distance: int


def default_k_values(D=DEFAULT_D, k_step=5):
    # k is integral: floor D/2 for odd D, and start the useful range at 2.
    useful = [k for k in range(2, D, k_step) if k < D / 4]
    return tuple(sorted({0, *useful, D // 2, D - 1}))


def _path_distance(ship, start, goal):
    queue = deque([(start, 0)])
    seen = {start}
    while queue:
        cell, distance = queue.popleft()
        if cell == goal:
            return distance
        for neighbor in ship.get_open_neighbors(cell):
            if neighbor not in seen:
                seen.add(neighbor)
                queue.append((neighbor, distance + 1))
    raise ValueError("LostBot is unreachable from the starting cell.")


def _run_trials(scenarios, detector, values, n_trials, seed):
    rows = []
    for value in values:
        print(f"Type {detector}: {'k' if detector == 'A' else 'alpha'}={value}", flush=True)
        for index, scenario in enumerate(scenarios[:n_trials], start=1):
            # Same bot seed per scenario across parameter settings; never use
            # global random state for detector draws or random path tie breaks.
            bot_class = BaselineTypeABot if detector == "A" else BaselineTypeBBot
            bot = bot_class(scenario.ship, scenario.start, scenario.lost_bot,
                            value, rng=random.Random(scenario.bot_seed))
            started = perf_counter()
            status, error = "round_limit", ""
            try:
                # Every unsuccessful scan rules out at least its current cell.
                for _ in range(len(bot.open_cells) + 1):
                    if bot.step():
                        status = "found"
                        break
            except RuntimeError as exc:
                status, error = "error", str(exc)
            elapsed = perf_counter() - started
            row = build_trial_row(
                bot, scenario.scenario_id, seed,
                k=value if detector == "A" else None,
                alpha=value if detector == "B" else None,
            )
            row.update(
                layout_seed=scenario.layout_seed, bot_seed=scenario.bot_seed,
                grid_size=scenario.ship.D, open_cells=len(bot.open_cells),
                start_row=scenario.start[0], start_col=scenario.start[1],
                lost_row=scenario.lost_bot[0], lost_col=scenario.lost_bot[1],
                initial_path_distance=scenario.initial_path_distance,
                initial_euclidean_distance=float(np.linalg.norm(
                    np.subtract(scenario.start, scenario.lost_bot))),
                moves_over_shortest_path=bot.move_actions - scenario.initial_path_distance,
                status=status, error=error, elapsed_seconds=elapsed,
            )
            rows.append(row)
            if index % max(1, n_trials // 10) == 0 or index == n_trials:
                print(f"  {index}/{n_trials} trials complete", flush=True)
    return rows


# Reuse ShipGrid(D) from Project 1 when building seeded scenarios.
def build_scenarios(D=DEFAULT_D, n_trials=DEFAULT_TRIALS, seed=440):
    rng = random.Random(seed)
    scenarios = []
    for scenario_id in range(n_trials):
        layout_seed = rng.randrange(2**32)
        state = random.getstate()
        try:
            random.seed(layout_seed)
            ship = ShipGrid(D)
        finally:
            random.setstate(state)
        start, lost_bot = rng.sample(ship.get_open_cells(), 2)
        scenarios.append(Scenario(
            scenario_id, seed, layout_seed, rng.randrange(2**32), ship,
            start, lost_bot, _path_distance(ship, start, lost_bot),
        ))
        if (scenario_id + 1) % max(1, n_trials // 10) == 0:
            print(f"Generated {scenario_id + 1}/{n_trials} ships (D={D})", flush=True)
    return scenarios


# $k = 0$ -> only the current cell no probability ~ maybe 100 trials
# The window is $1 x 1$. The detector pings if and only if the LocatorBot shares the exact cell with the LostBot. 
# While perfectly precise, it offers zero non-local information.
def run_type_a_zero_radius(scenarios, n_trials=500, seed=440):
    return _run_trials(scenarios, "A", [0], n_trials, seed)


# $1<k<D/4$ (High Sensitivity / Useful Region): ~ 100 trials for k in step 5 each one
def run_type_a_useful_region(scenarios, D, k_step=3, n_trials=500, seed=440):
    values = [k for k in default_k_values(D, k_step) if 1 < k < D / 4]
    return _run_trials(scenarios, "A", values, n_trials, seed)


# $k = D/2 $ (Half-Grid Coverage): 100 trials
# The footprint covers roughly a quarter to half of the ship.
def run_type_a_half_grid(scenarios, D, n_trials=500, seed=440):
    return _run_trials(scenarios, "A", [D // 2], n_trials, seed)


# $k>=D - 1$ : 100 trials
# The window covers the entire ship regardless of where the LocatorBot stands.
# The detector returns 1 unconditionally for all $a, b$ .
# The observation provides zero mutual information:
def run_type_a_full_coverage(scenarios, D, k_values=None, n_trials=500, seed=440):
    values = [D - 1] if k_values is None else list(k_values)
    return _run_trials(scenarios, "A", values, n_trials, seed)


# up to you to decide which is most important/most trials. We prolly need some metrics for comparison
def summarize_trials(trial_rows):
    # Group by detector/parameter; cost statistics include successful runs only.
    # CI columns use the normal approximation mean +/- 1.96 * standard error.


    groups = defaultdict(list)
    for row in trial_rows:
        key = tuple(row[field] for field in ("bot", "detector", "grid_size", "k", "alpha"))
        groups[key].append(row)
    results = []
    for key, rows in groups.items():
        result = dict(zip(("bot", "detector", "grid_size", "k", "alpha"), key))
        successful = [row for row in rows if row["found"]]
        result.update(trials=len(rows), successes=len(successful),
                      failures=len(rows) - len(successful),
                      success_rate=len(successful) / len(rows))
        for metric in METRICS:
            values = np.array([row[metric] for row in successful], dtype=float)
            stats = dict.fromkeys(STAT_NAMES, "")
            if len(values):
                mean = float(values.mean())
                stats.update(mean=mean, median=float(np.median(values)),
                             min=float(values.min()), max=float(values.max()))
                if len(values) > 1:
                    std = float(values.std(ddof=1))
                    sem = std / np.sqrt(len(values))
                    stats.update(std=std, sem=float(sem),
                                 ci95_low=float(mean - 1.96 * sem),
                                 ci95_high=float(mean + 1.96 * sem))
            result.update({f"{metric}_{stat}": value for stat, value in stats.items()})
        results.append(result)
    return results


# range of alpha for D=75
# The ping probability decays exponentially with Euclidean distance:

# $$\alpha \in \{0.01,0.05,0.10, ...., 0.60}$$ step size 0.05
def run_type_b_sweep(scenarios, alpha_values=DEFAULT_ALPHA_VALUES, n_trials=500, seed=440):
    return _run_trials(scenarios, "B", alpha_values, n_trials, seed)


# when tracking things to keep in csv: discovery cost as a function of alpha and k so cost, alpha and k for each rng seed
def build_trial_row(bot, scenario_id, seed, k=None, alpha=None):
    costs = calculate_discovery_costs(bot.move_actions, bot.detector_actions)
    return dict(
        scenario_id=scenario_id, seed=seed, bot=type(bot).__name__,
        detector="A" if k is not None else "B",
        k="" if k is None else k, alpha="" if alpha is None else alpha,
        found=int(bot.found), move_actions=bot.move_actions,
        detector_actions=bot.detector_actions, **costs,
    )


# also edit the discovery cost for question 6: see how things change if movement is free
# just a small change when calculating cost prolly dont have to rerun but keep track of both
# @Zahra


def calculate_discovery_costs(move_actions, detector_actions):
    # Return both standard cost and free-movement cost from the same trial.
    return {
        "discovery_cost": move_actions + 0.5 * detector_actions,
        "free_movement_cost": 0.5 * detector_actions,
    }


# CSV summary: aggregate discovery costs for each detector parameter.
def write_summary_csv(results, csv_path):
    _write_csv(results, csv_path, SUMMARY_FIELDS)


# Detailed CSV: per-seed k/alpha, move/scan counts, and both discovery costs.
def write_trials_csv(trial_rows, trials_csv_path):
    _write_csv(trial_rows, trials_csv_path, TRIAL_FIELDS)


def _write_csv(rows, path, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def run_sweep(D=DEFAULT_D, n_trials=DEFAULT_TRIALS, seed=440,
              k_values=None, alpha_values=None, detector="both",
              output_dir=DEFAULT_OUTPUT_DIR):
    """Run paired scenarios and checkpoint both CSVs after every parameter.

    Free-movement costs rescore these baseline trajectories; they do not measure
    a strategy redesigned for free movement. Existing output CSVs are replaced.
    """
    ks = default_k_values(D) if k_values is None else tuple(dict.fromkeys(k_values))
    alphas = DEFAULT_ALPHA_VALUES if alpha_values is None else tuple(dict.fromkeys(alpha_values))
    scenarios = build_scenarios(D, n_trials, seed)
    rows = []
    settings = []
    if detector in ("A", "both"):
        settings.extend(("A", k) for k in ks)
    if detector in ("B", "both"):
        settings.extend(("B", alpha) for alpha in alphas)
    output_dir = Path(output_dir)
    for kind, value in settings:
        rows.extend(_run_trials(scenarios, kind, [value], n_trials, seed))
        results = summarize_trials(rows)
        write_trials_csv(rows, output_dir / "baseline_trials.csv")
        write_summary_csv(results, output_dir / "baseline_summary.csv")
        latest = results[-1]
        print(f"  successes={latest['successes']}/{latest['trials']}; "
              f"mean cost={latest['discovery_cost_mean']}; "
              f"mean free-movement cost={latest['free_movement_cost_mean']}", flush=True)
    print(f"Saved results to {output_dir.resolve()}", flush=True)
    return rows, results


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-size", type=int, default=DEFAULT_D)
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS,
                        help=f"Trials per parameter, using the same scenarios (default: {DEFAULT_TRIALS})")
    parser.add_argument("--seed", type=int, default=440)
    parser.add_argument("--detector", choices=("A", "B", "both"), default="both")
    parser.add_argument("--k", type=int, nargs="+", help="Override Type A k values")
    parser.add_argument("--alpha", type=float, nargs="+", help="Override Type B alpha values")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_sweep(D=args.grid_size, n_trials=args.trials, seed=args.seed,
              k_values=args.k, alpha_values=args.alpha,
              detector=args.detector, output_dir=args.output_dir)

# Still need to make the free movement experiments after finding the best k,alpha values
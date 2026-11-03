from pathlib import Path
import sys

# Allow both direct execution and importing as Project2.experiments_baseline.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Project1.ship import ShipGrid


# Reuse ShipGrid(D) from Project 1 when building seeded scenarios.
def build_scenarios(D, n_trials=100, seed=440):
    pass


# $k = 0$ -> only the current cell no probability ~ maybe 100 trials
# The window is $1 x 1$. The detector pings if and only if the LocatorBot shares the exact cell with the LostBot. While perfectly precise, it offers zero non-local information.
def run_type_a_zero_radius(scenarios, n_trials=100, seed=440):
    pass


# $1<k<D/4$ (High Sensitivity / Useful Region): ~ 100 trials for k in step 5 each one
def run_type_a_useful_region(scenarios, D, k_step=5, n_trials=100, seed=440):
    pass


# $k = D/2 $ (Half-Grid Coverage): 100 trials
# The footprint covers roughly a quarter to half of the ship.
def run_type_a_half_grid(scenarios, D, n_trials=100, seed=440):
    pass


# $k>=D - 1$ : 100 trials 
# The window covers the entire ship regardless of where the LocatorBot stands. 
# The detector returns 1 unconditionally for all $a, b$ . 
# The observation provides zero mutual information:
def run_type_a_full_coverage(scenarios, D, k_values=None, n_trials=100, seed=440):
    pass


# up to you to decide which is most important/most trials. We prolly need some metrics for comparison
def summarize_trials(trial_rows):
    pass


# range of alpha for D+75 
def run_type_b_sweep(scenarios, alpha_values, n_trials=100, seed=440):
    pass


# when tracking things to keep in csv: discovery cost as a function of alpha and k so cost, alpha and k for each rng seed
def build_trial_row(bot, scenario_id, seed, k=None, alpha=None):
    pass


# also edit the discovery cost for question 6: see how things change is movement is free
# just a small change when calculating cost prolly dont have to rerun but keep track of both
# @Zahra 


def calculate_discovery_costs(move_actions, detector_actions):
    # Return both standard cost and free-movement cost from the same trial.
    pass


# CSV summary: aggregate discovery costs for each detector parameter.
def write_summary_csv(results, csv_path):
    pass


# Detailed CSV: per-seed k/alpha, move/scan counts, and both discovery costs.
def write_trials_csv(trial_rows, trials_csv_path):
    pass

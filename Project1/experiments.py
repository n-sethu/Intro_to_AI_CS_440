
# run reproducible experiments on bots 1-4 to compare their behaviors
# csv for summary of trials
# trials-csv for detailed comparisons 

import argparse
import csv
import os
import random
from bisect import bisect_right
from collections import deque
from dataclasses import dataclass
from time import perf_counter

from bots import Bot1, Bot2, Bot3, Bot4
from fire import FireSystem
from pathfinding import Pathfinder
from simulation import run_trial
from ship import ShipGrid


DEFAULT_BOTS = (Bot1, Bot2, Bot3, Bot4)
DEFAULT_Q_VALUES = tuple(step / 100 for step in range(20, 86, 5))
REASONS = ("button", "walked_into_fire", "fire_reached_bot", "timeout")
INF = float("inf")

@dataclass
class ExperimentResult:
	bot: str
	q: float
	trials: int
	successes: int
	elapsed_seconds: float
	reasons: dict[str, int]

	@property
	def success_rate(self):
		return self.successes / self.trials if self.trials else 0.0

	@property
	def average_seconds(self):
		return self.elapsed_seconds / self.trials if self.trials else 0.0

@dataclass
class Scenario:
	scenario_id: int
	ship: object
	start: tuple
	button: tuple
	fire_start: tuple
	fire_seed: int
	layout_seed: int
	features: dict      # q-independent geometry, computed once
	bot1_path: list     # bot 1's fixed plan, none if there is no path
	fire_dist: dict     # static BFS distance from the fire start 
	button_dist: dict   # static BFS distance from the button

@dataclass
class Timeline:
	# fire progression that is the same for each bot in each scenario
	ignition: dict          # step at which cell ignites, fire start = 0
	button_time: float      # step at which the button ignites
	sorted_times: list      # sorted ignition times, for burning-cell counts
	complete_until: int     # ignition info is exact for steps

def bfs_distances(ship, source):
	dist = {source: 0}
	queue = deque([source])
	while queue:
		cell = queue.popleft()
		for nb in ship.get_open_neighbors(cell):
			if nb not in dist:
				dist[nb] = dist[cell] + 1
				queue.append(nb)
	return dist
 
def manhattan(a, b):
	return abs(a[0] - b[0]) + abs(a[1] - b[1])
 
def cell_str(cell):
	return f"{cell[0]}-{cell[1]}"

def safe_route_length(ship, start, button, ignition):
	# length of shortest path that doesn't stand in a burning cell
	# at time t it is safe as long as cell ignites after time t
	if start == button:
		return 0
	seen = {start}
	frontier = [start]
	t = 0
	while frontier:
		t += 1
		nxt = []
		for cell in frontier:
			for nb in ship.get_open_neighbors(cell):
				if nb in seen:
					continue
				limit = t - 1 if nb == button else t
				if ignition.get(nb, INF) <= limit:
					continue
				if nb == button:
					return t
				seen.add(nb)
				nxt.append(nb)
		frontier = nxt
	return None

def path_survives(path, ignition):
	# true if a path doesn't catch on fire
	if path is None or len(path) < 2:
		return False
	last = len(path) - 1
	for t in range(1, len(path)):
		limit = t - 1 if t == last else t
		if ignition.get(path[t], INF) <= limit:
			return False
	return True

def scenario_features(ship, start, button, fire_start):
	fire_dist = bfs_distances(ship, fire_start)
	start_dist = bfs_distances(ship, start)
	button_dist = bfs_distances(ship, button)
	bot1_path = Pathfinder.next_move_astar(ship, start, button, {fire_start})
 
	open_cells = ship.get_open_cells()
	degrees = [len(ship.get_open_neighbors(c)) for c in open_cells]
	start_button = start_dist[button]
	fire_button = fire_dist[button]
	fire_start_dist = fire_dist[start]
	# fire can only move 1 cell 
	# a route whose cells are all farther from the fire than from the bot cannot be caught
	guaranteed = safe_route_length(ship, start, button, fire_dist)

	features = {
		"open_cells": len(open_cells),
		"open_fraction": f"{len(open_cells) / ship.D ** 2:.6f}",
		"dead_ends": sum(1 for d in degrees if d == 1),
		"junctions": sum(1 for d in degrees if d >= 3),
		"start": cell_str(start),
		"button": cell_str(button),
		"fire_start": cell_str(fire_start),
		"start_button_dist": start_button,
		"start_button_manhattan": manhattan(start, button),
		"start_fire_dist": fire_start_dist,
		"start_fire_manhattan": manhattan(start, fire_start),
		"button_fire_dist": fire_button,
		"button_fire_manhattan": manhattan(button, fire_start),
		"fire_closer_to_button": int(fire_button < start_button),
		"fire_start_on_shortest_path": int(
			bot1_path is None or len(bot1_path) - 1 != start_button
		),
		"bot1_path_len": (len(bot1_path) - 1) if bot1_path else "",
		"path_exists_avoiding_fire_start": int(bot1_path is not None),
		"guaranteed_win_route_len": "" if guaranteed is None else guaranteed,
		"guaranteed_win": int(guaranteed is not None),
	}
	return features, bot1_path, fire_dist, button_dist

def build_scenarios(D, n_trials, seed):
	# create a scenario for the bots
	rng = random.Random(seed)
	scenarios = []

	for scenario_id in range(n_trials):
		layout_seed = rng.randrange(2**32)
		random_state = random.getstate()
		random.seed(layout_seed)
		try:
			ship = ShipGrid(D)
		finally:
			random.setstate(random_state)
 
		start, button, fire_start = rng.sample(ship.get_open_cells(), 3)
		fire_seed = rng.randrange(2**32)
		features, bot1_path, fire_dist, button_dist = scenario_features(
			ship, start, button, fire_start
		)
		scenarios.append(
			Scenario(
				scenario_id, ship, start, button, fire_start, fire_seed,
				layout_seed, features, bot1_path, fire_dist, button_dist,
			)
		)

	return scenarios

def compute_timeline(scenario, q, max_steps):
	# run the fire just to track when each cell ignites
	state = random.getstate()
	random.seed(scenario.fire_seed)
	try:
		ship = scenario.ship
		fire = FireSystem(ship, q, scenario.fire_start)
		ignition = {scenario.fire_start: 0}
		n_open = len(ship.get_open_cells())
		t = 0
		while scenario.button not in ignition and len(ignition) < n_open and t < max_steps:
			t += 1
			for cell in fire.step():
				ignition[cell] = t
	finally:
		random.setstate(state)
	button_time = ignition.get(scenario.button, INF)
	return Timeline(ignition, button_time, sorted(ignition.values()), t)

def burning_count(timeline, step):
	# cells that are burning after a step
	if step > timeline.complete_until:
		return None
	return bisect_right(timeline.sorted_times, step)

def first_divergence(trace_a, trace_b):
	# track when two bots diverge paths
	for t, (a, b) in enumerate(zip(trace_a, trace_b)):
		if a != b:
			return t
	return -1 if len(trace_a) == len(trace_b) else min(len(trace_a), len(trace_b))

def trajectory_metrics(trace, timeline, scenario):
	steps = len(trace) - 1
	moves = sum(1 for a, b in zip(trace, trace[1:]) if a != b)
	# smallest along the way
	# 0 or below - fire got there
	# small positive - a close call
	# cells that never ignited - button_time + 1 
	cap = timeline.button_time + 1 if timeline.button_time != INF else timeline.complete_until + 1
	min_slack = min(timeline.ignition.get(cell, cap) - t for t, cell in enumerate(trace))
	final_to_button = scenario.button_dist.get(trace[-1], "")
	start_to_button = scenario.features["start_button_dist"]
	progress = ""
	if final_to_button != "" and start_to_button:
		progress = f"{1 - final_to_button / start_to_button:.4f}"
	return {
		"steps": steps,
		"moves": moves,
		"waits": steps - moves,
		"min_slack": min_slack,
		"final_dist_to_button": final_to_button,
		"progress_to_button": progress,
	}

BOT_EXTRA_FIELDS = {
	"Bot4": (
		"plan_count", "relaxed_count", "fallback_count", "no_path_count",
		"max_active_threshold", "final_strategy",
	),
}
 
def bot_extras(bot_name, bot):
	if bot_name != "Bot4":
		return {}
	if bot.active_threshold is not None:
		strategy = "risk_path"
	elif bot.path is not None:
		strategy = "shortest_path_fallback"
	else:
		strategy = "no_path"
	return {
		"plan_count": bot.plan_count,
		"relaxed_count": bot.relaxed_count,
		"fallback_count": bot.fallback_count,
		"no_path_count": bot.no_path_count,
		"max_active_threshold": "" if bot.max_active_threshold is None else bot.max_active_threshold,
		"final_strategy": strategy,
	}

SCENARIO_FIELDS = [
	"scenario_id", "q", "grid_size", "seed", "layout_seed", "fire_seed",
	"open_cells", "open_fraction", "dead_ends", "junctions",
	"start", "button", "fire_start",
	"start_button_dist", "start_button_manhattan",
	"start_fire_dist", "start_fire_manhattan",
	"button_fire_dist", "button_fire_manhattan",
	"fire_closer_to_button", "fire_start_on_shortest_path",
	"path_exists_avoiding_fire_start", "bot1_path_len",
	"guaranteed_win", "guaranteed_win_route_len",
	"fire_button_time", "oracle_win", "oracle_route_len", "oracle_margin",
	"bot1_predicted_win",
]
PER_BOT_FIELDS = [
	"result", "success", "steps", "moves", "waits", "min_slack",
	"final_dist_to_button", "progress_to_button", "fire_cells_at_end",
	"trial_seconds", "diverges_from_Bot2", "diverges_from_Bot3",
]
CROSS_FIELDS = [
	"n_bots_succeeded", "succeeded_bots", "failed_bots",
	"all_bots_succeeded", "all_bots_failed",
]

def trial_fieldnames(bot_names):
	fields = list(SCENARIO_FIELDS)
	for name in bot_names:
		fields += [f"{name}_{f}" for f in PER_BOT_FIELDS]
		fields += [f"{name}_{f}" for f in BOT_EXTRA_FIELDS.get(name, ())]
	return fields + CROSS_FIELDS

def build_trial_row(scenario, q, timeline, bot_names, records, D, seed):
	f = scenario.features
	row = {k: f[k] for k in SCENARIO_FIELDS if k in f}
	oracle_len = safe_route_length(scenario.ship, scenario.start, scenario.button, timeline.ignition)
	row.update(
		scenario_id=scenario.scenario_id,
		q=f"{q:.2f}",
		grid_size=D,
		seed=seed,
		layout_seed=scenario.layout_seed,
		fire_seed=scenario.fire_seed,
		fire_button_time="" if timeline.button_time == INF else timeline.button_time,
		oracle_win=int(oracle_len is not None),
		oracle_route_len="" if oracle_len is None else oracle_len,
		oracle_margin="" if oracle_len is None or timeline.button_time == INF
		else timeline.button_time - oracle_len,
		bot1_predicted_win=int(path_survives(scenario.bot1_path, timeline.ignition)),
	)
 
	succeeded, failed = [], []
	for name in bot_names:
		rec = records[name]
		for key, value in rec["fields"].items():
			row[f"{name}_{key}"] = value
		(succeeded if rec["fields"]["success"] else failed).append(name)
		for ref in ("Bot2", "Bot3"):
			if name != ref and ref in records:
				row[f"{name}_diverges_from_{ref}"] = first_divergence(
					rec["trace"], records[ref]["trace"]
				)
	row["n_bots_succeeded"] = len(succeeded)
	row["succeeded_bots"] = "|".join(succeeded)
	row["failed_bots"] = "|".join(failed)
	row["all_bots_succeeded"] = int(not failed)
	row["all_bots_failed"] = int(not succeeded)
	return row

def run_sweep(
	D=75,
	n_trials=1000,
	q_values=DEFAULT_Q_VALUES,
	seed=440,
	max_steps=100000,
	bot_classes=DEFAULT_BOTS,
	csv_path=None,
	trials_csv_path=None,
):
	# run every bot on the same scenarios
	# return the times results
	scenarios = build_scenarios(D, n_trials, seed)
	bot_names = [b.__name__ for b in bot_classes]
	results = []
	progress_interval = max(1, n_trials // 10)
	print(
		f"Starting sweep: {len(q_values)} q values, "
		f"{len(bot_classes)} bots, {n_trials} trials each",
		flush=True,
	)

	trials_file = None
	trials_writer = None
	if trials_csv_path:
		trials_file = open(trials_csv_path, "w", newline="")
		trials_writer = csv.DictWriter(trials_file, fieldnames=trial_fieldnames(bot_names))
		trials_writer.writeheader()

	try:
		for q in q_values:
			timelines = [compute_timeline(sc, q, max_steps) for sc in scenarios]
			records = [{} for _ in scenarios]
			bot1_mismatches = 0
 
			for bot_class in bot_classes:
				name = bot_class.__name__
				reasons = {reason: 0 for reason in REASONS}
				started = perf_counter()
 
				for trial_number, (sc, tl) in enumerate(zip(scenarios, timelines), start=1):
					trace = []
					trial_started = perf_counter()
					success, reason, bot = run_trial(
						sc.ship,
						bot_class,
						sc.start,
						sc.button,
						sc.fire_start,
						q,
						seed=sc.fire_seed,
						max_steps=max_steps,
						trace=trace,
					)
					trial_seconds = perf_counter() - trial_started
					reasons[reason] = reasons.get(reason, 0) + 1
					if success != (reason == "button"):
						raise RuntimeError(
							f"{name} returned inconsistent trial result"
						)
					# bot 1 follows a fixed path, timeline predicts its result exactly
					if name == "Bot1" and success != path_survives(sc.bot1_path, tl.ignition):
						bot1_mismatches += 1
 
					fields = {
						"result": reason,
						"success": int(success),
						**trajectory_metrics(trace, tl, sc),
						"trial_seconds": f"{trial_seconds:.6f}",
					}
					# bot checked before fire step for botton or walked into fire
					# bot checked after otherwise
					steps = fields["steps"]
					checked_after_fire = reason in ("fire_reached_bot", "timeout")
					count = burning_count(tl, steps if checked_after_fire else steps - 1)
					fields["fire_cells_at_end"] = "" if count is None else count
					fields.update(bot_extras(name, bot))
					records[trial_number - 1][name] = {"fields": fields, "trace": trace}
 
					if trial_number % progress_interval == 0 or trial_number == n_trials:
						print(
							f"Progress q={q:.2f} bot={name}: "
							f"{trial_number}/{n_trials} trials "
							f"({perf_counter() - started:.1f}s)",
							flush=True,
						)
 
				elapsed_seconds = perf_counter() - started
				result = ExperimentResult(
					bot=name,
					q=q,
					trials=n_trials,
					successes=reasons["button"],
					elapsed_seconds=elapsed_seconds,
					reasons=reasons,
				)
				results.append(result)
				print_result(result)
 
			if bot1_mismatches:
				print(
					f"WARNING q={q:.2f}: Bot1 result disagreed with the fire-timeline "
					f"prediction in {bot1_mismatches} trials (check determinism)",
					flush=True,
				)
 
			if trials_writer:
				for sc, tl, rec in zip(scenarios, timelines, records):
					trials_writer.writerow(
						build_trial_row(sc, q, tl, bot_names, rec, D, seed)
					)
				trials_file.flush()
	finally:
		if trials_file:
			trials_file.close()
 
	if csv_path:
		write_results_csv(results, csv_path)
 
	return results

def print_result(result):
	print(
		f"q={result.q:.2f} bot={result.bot:<5} "
		f"success={result.successes}/{result.trials} "
		f"rate={result.success_rate:.3f} "
		f"time={result.elapsed_seconds:.3f}s "
		f"avg={result.average_seconds:.3f}s",
		flush=True,
	)


def write_results_csv(results, csv_path):
	# one row per bot/q pair
	fieldnames = [
		"q",
		"bot",
		"trials",
		"successes",
		"success_rate",
		"elapsed_seconds",
		"average_seconds",
		"button",
		"walked_into_fire",
		"fire_reached_bot",
		"timeout",
	]
	with open(csv_path, "w", newline="") as output_file:
		writer = csv.DictWriter(output_file, fieldnames=fieldnames)
		writer.writeheader()
		for result in results:
			writer.writerow(
				{
					"q": f"{result.q:.2f}",
					"bot": result.bot,
					"trials": result.trials,
					"successes": result.successes,
					"success_rate": f"{result.success_rate:.6f}",
					"elapsed_seconds": f"{result.elapsed_seconds:.6f}",
					"average_seconds": f"{result.average_seconds:.6f}",
					**result.reasons,
				}
			)

def default_trials_path(csv_path):
	root, ext = os.path.splitext(csv_path)
	return f"{root}_trials{ext or '.csv'}"

def parse_args():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--grid-size", type=int, default=75)
	parser.add_argument("--trials", type=int, default=10)
	parser.add_argument("--seed", type=int, default=440)
	parser.add_argument("--max-steps", type=int, default=100000)
	parser.add_argument(
		"--q",
		type=float,
		action="append",
		help="Run only the specified fire-spread probability; may be repeated",
	)
	parser.add_argument("--csv", default="bot4_sweep.csv")
	parser.add_argument(
		"--trials-csv",
		default=None,
		help="Per-trial detail CSV (default: <--csv name>_trials.csv)",
	)
	parser.add_argument(
		"--no-trials-csv", action="store_true", help="Skip the per-trial detail CSV"
	)
	return parser.parse_args()


if __name__ == "__main__":
	args = parse_args()
	if args.grid_size < 3:
		raise SystemExit("--grid-size must be at least 3")
	if args.trials < 1:
		raise SystemExit("--trials must be at least 1")
	trials_csv = None
	if not args.no_trials_csv:
		trials_csv = args.trials_csv or default_trials_path(args.csv)
	run_sweep(
		D=args.grid_size,
		n_trials=args.trials,
		q_values=args.q if args.q else DEFAULT_Q_VALUES,
		seed=args.seed,
		max_steps=args.max_steps,
		csv_path=args.csv,
		trials_csv_path=trials_csv,
	)

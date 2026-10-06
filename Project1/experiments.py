"""Run reproducible Bot 1-4 experiments across a fire-spread sweep."""

import argparse
import csv
import random
from dataclasses import dataclass
from time import perf_counter

from bots import Bot1, Bot2, Bot3, Bot4
from simulation import run_trial
from ship import ShipGrid


DEFAULT_BOTS = (Bot1, Bot2, Bot3, Bot4)
DEFAULT_Q_VALUES = tuple(step / 100 for step in range(20, 86, 5))


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


def build_scenarios(D, n_trials, seed):
	"""Create fixed ship/start/fire scenarios so bots receive fair comparisons."""
	rng = random.Random(seed)
	scenarios = []

	for _ in range(n_trials):
		layout_seed = rng.randrange(2**32)
		random_state = random.getstate()
		random.seed(layout_seed)
		try:
			ship = ShipGrid(D)
		finally:
			random.setstate(random_state)

		start, button, fire_start = rng.sample(ship.get_open_cells(), 3)
		fire_seed = rng.randrange(2**32)
		scenarios.append((ship, start, button, fire_start, fire_seed))

	return scenarios


def run_sweep(
	D=75,
	n_trials=1000,
	q_values=DEFAULT_Q_VALUES,
	seed=440,
	max_steps=100000,
	bot_classes=DEFAULT_BOTS,
	csv_path=None,
):
	"""Run every bot on the same scenarios and return timed result records."""
	scenarios = build_scenarios(D, n_trials, seed)
	results = []
	progress_interval = max(1, n_trials // 10)
	print(
		f"Starting sweep: {len(q_values)} q values, "
		f"{len(bot_classes)} bots, {n_trials} trials each",
		flush=True,
	)

	for q in q_values:
		for bot_class in bot_classes:
			reasons = {
				"button": 0,
				"walked_into_fire": 0,
				"fire_reached_bot": 0,
				"timeout": 0,
			}
			started = perf_counter()

			for trial_number, (ship, start, button, fire_start, fire_seed) in enumerate(
				scenarios, start=1
			):
				success, reason, _bot = run_trial(
					ship,
					bot_class,
					start,
					button,
					fire_start,
					q,
					seed=fire_seed,
					max_steps=max_steps,
				)
				reasons[reason] = reasons.get(reason, 0) + 1
				if success != (reason == "button"):
					raise RuntimeError(
						f"{bot_class.__name__} returned inconsistent trial result"
					)
				if trial_number % progress_interval == 0 or trial_number == n_trials:
					print(
						f"Progress q={q:.2f} bot={bot_class.__name__}: "
						f"{trial_number}/{n_trials} trials "
						f"({perf_counter() - started:.1f}s)",
						flush=True,
					)

			elapsed_seconds = perf_counter() - started
			result = ExperimentResult(
				bot=bot_class.__name__,
				q=q,
				trials=n_trials,
				successes=reasons["button"],
				elapsed_seconds=elapsed_seconds,
				reasons=reasons,
			)
			results.append(result)
			print_result(result)

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
	"""Write one row per bot/q pair for plotting or statistical analysis."""
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
	return parser.parse_args()


if __name__ == "__main__":
	args = parse_args()
	if args.grid_size < 3:
		raise SystemExit("--grid-size must be at least 3")
	if args.trials < 1:
		raise SystemExit("--trials must be at least 1")
	run_sweep(
		D=args.grid_size,
		n_trials=args.trials,
		q_values=args.q if args.q else DEFAULT_Q_VALUES,
		seed=args.seed,
		max_steps=args.max_steps,
		csv_path=args.csv,
	)

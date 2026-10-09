
# Run: python analysis.py [--output-dir PATH] [--show]


import argparse
import csv
import math
from pathlib import Path

BOT_NAMES = ("Bot1", "Bot2", "Bot3", "Bot4")
COLORS = ("#707070", "#0072B2", "#E69F00", "#009E73")


def wilson(successes, n):
    # Pointwise 95% Wilson interval for a binomial success frequency
    p, z = successes / n, 1.959963984540054
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return center - radius, center + radius


def load_sweeps(directory):
    data = {}
    for path in sorted(directory.glob("bot4-sweep-q-*_trials.csv")):
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise ValueError(f"Empty sweep: {path}")
        q = float(rows[0]["q"])
        if q in data:
            raise ValueError(f"Duplicate q={q}: refusing to count repeated sweeps")
        keys = set()
        for row in rows:
            key = (row["layout_seed"], row["fire_seed"], row["scenario_id"])
            if float(row["q"]) != q or key in keys:
                raise ValueError(f"Mixed q or duplicate scenario in {path}")
            keys.add(key)
            for bot in BOT_NAMES:
                if row[f"{bot}_success"] not in ("0", "1"):
                    raise ValueError(f"Invalid success indicator in {path}")
        summary_path = path.with_name(path.name.replace("_trials.csv", ".csv"))
        with summary_path.open(newline="") as handle:
            summary = {row["bot"]: row for row in csv.DictReader(handle)}
        for bot in BOT_NAMES:
            successes = sum(int(row[f"{bot}_success"]) for row in rows)
            if int(summary[bot]["trials"]) != len(rows) or int(summary[bot]["successes"]) != successes:
                raise ValueError(f"Summary and trial counts disagree: {summary_path}, {bot}")
        data[q] = rows
    if not data:
        raise ValueError(f"No bot4 sweep trial CSVs found in {directory}")
    return dict(sorted(data.items()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "graphs")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = load_sweeps(Path(__file__).resolve().parent)
    qs = list(data)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(args.output_dir / f"{name}.png", dpi=200, bbox_inches="tight")
        if not args.show:
            plt.close(fig)

    def decorate(ax, ylabel):
        ax.set(xlabel="Fire spread probability q", ylabel=ylabel)
        ax.grid(alpha=0.2)

    rates = {}
    fig, ax = plt.subplots(figsize=(10, 6))
    for bot, color in zip(BOT_NAMES, COLORS):
        counts = [sum(int(r[f"{bot}_success"]) for r in data[q]) for q in qs]
        rates[bot] = [k / len(data[q]) for k, q in zip(counts, qs)]
        intervals = [wilson(k, len(data[q])) for k, q in zip(counts, qs)]
        ax.plot(qs, rates[bot], "o-", color=color, label=bot)
        ax.fill_between(qs, [lo for lo, hi in intervals], [hi for lo, hi in intervals], color=color, alpha=0.10)
    ax.set(title="Fire extinguishing success across all tested q", ylim=(0, 1.02))
    decorate(ax, "Successful trials / total trials")
    ax.legend(title="Shading: pointwise 95% Wilson intervals")
    save(fig, "success_rate_vs_q")

    fig, ax = plt.subplots(figsize=(10, 5))
    for bot, color in zip(BOT_NAMES[1:], COLORS[1:]):
        means, errors = [], []
        for rows in data.values():
            differences = [int(r[f"{bot}_success"]) - int(r["Bot1_success"]) for r in rows]
            n = len(rows)
            mean = sum(differences) / n
            se = math.sqrt(sum((d - mean) ** 2 for d in differences) / (n * (n - 1))) if n > 1 else 0
            means.append(100 * mean)
            errors.append(100 * 1.96 * se)
        ax.errorbar(qs, means, yerr=errors, fmt="o-", capsize=3, color=color, label=f"{bot} − Bot1")
    ax.axhline(0, color="black", linewidth=1)
    ax.set_title("Where adaptive planning helps: paired gains over Bot1")
    decorate(ax, "Success-rate gain (percentage points)")
    ax.legend(title="Approximate pointwise 95% paired intervals")
    save(fig, "paired_success_gain")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, closer in zip(axes, (True, False)):
        for bot, color in zip(BOT_NAMES, COLORS):
            probabilities = []
            for rows in data.values():
                subset = [r for r in rows if (int(r["start_button_dist"]) < int(r["button_fire_dist"])) == closer]
                probabilities.append(sum(int(r[f"{bot}_success"]) for r in subset) / len(subset) if subset else float("nan"))
            ax.plot(qs, probabilities, "o-", color=color, label=bot)
        ax.set_title("Bot starts closer to button" if closer else "Fire starts closer or equally close")
        decorate(ax, "Success frequency within distance group")
        ax.set_ylim(0, 1.02)
    axes[1].legend()
    fig.suptitle("Initial shortest-path distances explain the high-q regime")
    save(fig, "success_by_initial_distance")

    print(f"Saved three graphs to {args.output_dir}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()

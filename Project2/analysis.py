"""Combine Project2 trial snapshots and generate writeup figures (no simulations).

python analysis.py --combine-only    # refresh combined_trials.csv, stdlib only
python analysis.py                   # refresh CSV, then analyze (pandas/matplotlib)
python analysis.py --no-combine      # analyze the existing snapshot

Never mixes summary rows into trial data. Run again after the last shard finishes.
"""
import argparse
import csv
import io
import os
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parent
SCENARIO = ['seed', 'scenario_id', 'layout_seed', 'bot_seed', 'grid_size',
            'start_row', 'start_col', 'lost_row', 'lost_col']
IDENTITY = SCENARIO + ['bot', 'detector', 'k', 'alpha']


def combine(root, destination):
    """Read each saved trial file once; reject conflicting duplicate outcomes."""
    rows, fields = {}, None
    sources = sorted(p for p in root.rglob('*_trials.csv')
                     if p.resolve() != destination.resolve()
                     and 'analysis-results' not in p.parts
                     and not p.name.startswith('combined'))
    if not sources:
        raise ValueError('No source trial CSVs found')
    duplicate_count = 0
    for source in sources:
        # The experiment writer replaces its CSV after each parameter batch.
        before = source.stat()
        content = source.read_text()
        after = source.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError(f'{source} changed during reading; retry after the save')
        reader = csv.DictReader(io.StringIO(content))
        if fields is None:
            fields = reader.fieldnames
        if reader.fieldnames != fields:
            raise ValueError(f'Unexpected schema: {source}')
        for row in reader:
            if None in row or any(v is None for v in row.values()):
                raise ValueError(f'Incomplete CSV row in {source}; retry after the save')
            key = tuple(row[f] for f in IDENTITY)
            relative = str(source.relative_to(root))
            if key in rows:
                previous = rows[key]
                # Runtime varies between repeat exports/runs; observations must agree.
                if any(previous[f] != row[f] for f in fields if f != 'elapsed_seconds'):
                    raise ValueError(f'Conflicting trial {key} in {relative}; select one run explicitly')
                previous['source_files'] += ';' + relative
                duplicate_count += 1
            else:
                rows[key] = dict(row, source_files=relative)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', newline='', dir=destination.parent,
                                     delete=False) as handle:
        temporary = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=fields + ['source_files'])
        writer.writeheader()
        writer.writerows(rows.values())
    os.replace(temporary, destination)
    print(f'Saved {len(rows)} unique trials from {len(sources)} files to {destination}; '
          f'{duplicate_count} duplicate rows merged.')


def analyze(source, output):
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(source)
    data['parameter'] = data['k'].where(data.detector.eq('A'), data.alpha)
    data['success'] = data.found.eq(1) & data.status.eq('found')
    keys = ['grid_size', 'bot', 'detector', 'parameter']
    metrics = ['discovery_cost', 'free_movement_cost', 'move_actions', 'detector_actions']
    coverage = data.groupby(keys).agg(trials=('success', 'size'), successes=('success', 'sum'))
    coverage['success_rate'] = coverage.successes / coverage.trials
    coverage.to_csv(output / 'coverage.csv')
    summary = coverage.copy()
    for metric in metrics:
        stats = data[data.success].groupby(keys)[metric].agg(['mean', 'std', 'count'])
        summary[metric + '_mean'] = stats['mean']
        summary[metric + '_ci95'] = 1.96 * stats['std'] / stats['count'] ** .5
    summary.to_csv(output / 'summary.csv')
    notes = [
        'Snapshot analysis; rerun after shard 7 completes. Only CSV-persisted trials are included.',
        'Costs: discovery = moves + 0.5 * detector calls; free movement = 0.5 * detector calls.',
        'Error bars are approximate 95% normal confidence intervals across successful scenarios.',
        'Means exclude failures; inspect coverage.csv before comparing costs. Failed runs are not zero-cost successes.',
        'Unequal sample sizes are shown in coverage.csv; paired differences use identical scenario identities.',
        'Free-movement plots rescore trajectories. Only explicitly free-tuned bot classes demonstrate a changed strategy.',
        'ImprovedTypeBBot CSVs do not record move_weight. Keep different planning objectives in separate datasets; the existing job uses the default weight 1.',
        'Best parameters are observed sweep minima, not independently validated optima. Boundary minima warrant a wider sweep.',
        'Q1/Q2 require formulas and strategy explanations from bots.py; plots alone do not establish correctness.',
    ]
    for detector, required in [('A', 'ImprovedTypeABot'), ('B', 'ImprovedTypeBBot')]:
        if required not in set(data.bot):
            notes.append(f'MISSING Q3/Q4 comparison data: {required}.')
    if 'ImprovedTypeAFreeBot' not in set(data.bot):
        notes.append('MISSING Q6: no ImprovedTypeAFreeBot trials. No explicit free-tuned Type B label is available in the current schema.')

    best_rows = []
    for size, ship in data.groupby('grid_size'):
        for metric in metrics[:2]:
            for detector in ['A', 'B']:
                table = summary.reset_index()
                table = table[(table.grid_size == size) & (table.detector == detector)]
                if table.empty:
                    continue
                fig, ax = plt.subplots(figsize=(9, 5))
                for bot, group in table.groupby('bot'):
                    group = group.sort_values('parameter')
                    ax.errorbar(group.parameter, group[metric + '_mean'],
                                yerr=group[metric + '_ci95'], marker='o', capsize=3, label=bot)
                ax.set(xlabel='k' if detector == 'A' else 'α',
                       ylabel='Mean discovery cost' if metric == 'discovery_cost' else 'Mean detector-only cost',
                       title=f'Type {detector}, {size}×{size} ship (successful trials; 95% CI)')
                ax.legend(); ax.grid(alpha=.25)
                fig.tight_layout(); fig.savefig(output / f'type_{detector}_{metric}_D{size}.png', dpi=180)
                plt.close(fig)

            # Common scenarios across every bot/parameter, so tuning uses equal cohorts.
            groups = list(ship.groupby(['bot', 'detector', 'parameter']))
            sets = [set(map(tuple, g[SCENARIO].to_numpy())) for _, g in groups]
            common = set.intersection(*sets) if sets else set()
            matched = ship[ship[SCENARIO].apply(tuple, axis=1).isin(common)]
            scores = matched.groupby(['bot', 'detector', 'parameter']).agg(
                mean_cost=(metric, 'mean'), n=('success', 'size'), successes=('success', 'sum'))
            scores = scores[scores.n.eq(scores.successes)]
            for (bot, detector), group in scores.groupby(level=[0, 1]):
                index = group.mean_cost.idxmin()
                best_rows.append(dict(grid_size=size, metric=metric, bot=bot, detector=detector,
                                      parameter=index[2], mean_cost=group.loc[index, 'mean_cost'],
                                      n=group.loc[index, 'n']))

        # Matched parameter/scenario comparison: negative differences favor improved.
        for detector in ['A', 'B']:
            baseline = ship[ship.bot.eq(f'BaselineType{detector}Bot') & ship.success]
            for bot, improved in ship[~ship.bot.str.startswith('Baseline') & ship.detector.eq(detector) & ship.success].groupby('bot'):
                pairs = improved.merge(baseline, on=SCENARIO + ['parameter'], suffixes=('_improved', '_baseline'), validate='one_to_one')
                if pairs.empty:
                    notes.append(f'No matched parameter/scenario pairs for {bot}, D={size}.')
                    continue
                fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                for ax, metric in zip(axes, metrics[:2]):
                    pairs['difference'] = pairs[metric + '_improved'] - pairs[metric + '_baseline']
                    stats = pairs.groupby('parameter').difference.agg(['mean', 'std', 'count'])
                    stats.to_csv(output / f'paired_{bot}_{metric}_D{size}.csv')
                    ax.errorbar(stats.index, stats['mean'], yerr=1.96 * stats['std'] / stats['count'] ** .5, marker='o', capsize=3)
                    ax.axhline(0, color='black', linewidth=1)
                    ax.set(xlabel='k' if detector == 'A' else 'α', ylabel='Improved − baseline', title=metric)
                fig.suptitle(f'{bot}: paired successful trials, D={size}, 95% CI')
                fig.tight_layout(); fig.savefig(output / f'paired_{bot}_D{size}.png', dpi=180)
                plt.close(fig)
    pd.DataFrame(best_rows).to_csv(output / 'best_parameters_common_scenarios.csv', index=False)
    if best_rows:
        best = pd.DataFrame(best_rows)
        for (size, metric), group in best.groupby(['grid_size', 'metric']):
            fig, ax = plt.subplots(figsize=(10, 5))
            labels = [f'{r.bot}\nparameter={r.parameter:g}, n={r.n}' for r in group.itertuples()]
            ax.bar(labels, group.mean_cost)
            ax.set(ylabel='Mean cost', title=f'Observed best parameters on common scenarios: {metric}, D={size}')
            ax.tick_params(axis='x', labelrotation=15)
            fig.tight_layout(); fig.savefig(output / f'tuned_{metric}_D{size}.png', dpi=180)
            plt.close(fig)
    notes.extend([
        'Q3: use Type A discovery-cost curves; discuss the transition/near-minimum k range and missing strategy data.',
        'Q4: use Type B curves and paired differences; justify alpha using its distance-dependent likelihood and sampled range.',
        'Q5: compare observed minima in best_parameters_common_scenarios.csv and tuned plots, with the common-cohort n.',
        'Q6: use detector-only curves, but collect changed-strategy trials before claiming optimal free-movement performance.',
    ])
    (output / 'analysis_notes.txt').write_text('\n'.join(notes) + '\n')
    print(f'Analysis saved to {output}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--combined', type=Path, default=None)
    parser.add_argument('--output-dir', type=Path, default=None)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--combine-only', action='store_true')
    mode.add_argument('--no-combine', action='store_true')
    args = parser.parse_args()
    combined = args.combined or args.root / 'combined_trials.csv'
    if not args.no_combine:
        combine(args.root, combined)
    if not args.combine_only:
        analyze(combined, args.output_dir or args.root / 'analysis-results')


if __name__ == '__main__':
    main()

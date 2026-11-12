#!/usr/bin/env python3
"""Regenerate the updated comparison from the CSV snapshots in data/."""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR'] = str(ROOT / '.matplotlib')
if importlib.util.find_spec('matplotlib') is None:
    existing = Path('/Users/skundaliya/Documents/Originial Siren/siren-main/venv/bin/python')
    if existing.is_file() and Path(sys.executable).absolute() != existing.absolute():
        os.execv(str(existing), [str(existing), str(Path(__file__).resolve()), *sys.argv[1:]])
    raise SystemExit('Install Matplotlib first: python3 -m pip install -r "' + str(ROOT / 'requirements.txt') + '"')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator

MODELS = ['coin', 'gaussian-gaussian', 'gtree', 'kalman', 'wheels', 'outlier', 'slam', 'gaussian-gaussian-hybrid']
ENGINES = ['Miking CorePPL', 'Siren', 'SirenC']
COLORS = ['#009E73', '#0072B2', '#D55E00']
MARKERS = ['o', 's', '^']
FLOOR = 1e-12


def read(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def write(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def aggregate():
    points, inventory = [], []
    for engine, name in [('Siren', 'siren_raw.csv'), ('SirenC', 'sirenc_raw.csv')]:
        rows = read(ROOT / 'data' / name)
        seen = set()
        for row in rows:
            key = (row['benchmark'], int(row['particles']), int(row['repeat']))
            if key in seen:
                raise ValueError(f'Duplicate {engine} run: {key}')
            seen.add(key)
            if row['status'] == 'ok':
                mse = float(row['standardized_mse'])
                rmse = float(row['mean_standardized_rmse'])
                if not math.isfinite(mse) or mse < 0 or not math.isclose(mse, rmse**2, rel_tol=1e-10, abs_tol=1e-25):
                    raise ValueError(f'Inconsistent error metrics: {engine} {key}')
        groups = sorted({(row['benchmark'], int(row['particles'])) for row in rows})
        for model, count in groups:
            group = [r for r in rows if r['benchmark'] == model and int(r['particles']) == count]
            valid = [r for r in group if r['status'] == 'ok']
            if not valid:
                continue
            points.append(dict(engine=engine, model=model, particles=count,
                successful_runs=len(valid), failed_runs=len(group)-len(valid),
                median_time_ms=statistics.median(float(r['time_ms']) for r in valid),
                standardized_rmse=math.sqrt(statistics.fmean(float(r['standardized_mse']) for r in valid)),
                mean_rmse_rms=math.sqrt(statistics.fmean(float(r['mean_rmse'])**2 for r in valid)),
                readout='symbolic', requested_readout='symbolic',
                method='smc-ssi' if engine == 'Siren' else 'compiled-symbolic',
                known_issue='', summary_time_ms='', source_file='data/' + name,
                run_origin=valid[0]['run_origin']))
    seen = set()
    for lineno, row in enumerate(read(ROOT / 'data' / 'coreppl_summary.csv'), 2):
        reason = ''
        if row['engine'] != 'coreppl':
            reason = 'Use the updated user-supplied Siren raw.csv instead of repository Siren runs.'
        elif row['method'] != 'is-lw' or row['mode'] != 'static':
            reason = 'Outside static is-lw comparison.'
        elif not (row.get('requested_readout') == 'posterior' or
                  (not row.get('requested_readout') and row.get('readout') == 'posterior')):
            reason = 'Explicit sample-readout control.'
        elif not int(row['successful_runs']):
            reason = 'No successful measurements.'
        inventory.append(dict(source_row=lineno, **row, selected=not bool(reason), reason=reason))
        if reason:
            continue
        key = (row['model'], int(row['particles']))
        if key in seen:
            raise ValueError(f'Duplicate CorePPL summary: {key}')
        seen.add(key)
        points.append(dict(engine='Miking CorePPL', model=key[0], particles=key[1],
            successful_runs=int(row['successful_runs']), failed_runs=int(row['runs'])-int(row['successful_runs']),
            median_time_ms=1000*float(row['inference_seconds_median']),
            standardized_rmse=float(row['mean_standardized_rmse_rms']),
            mean_rmse_rms=float(row['mean_rmse_rms']),
            readout=row['readout'], requested_readout=row.get('requested_readout', ''), method=row['method'],
            known_issue=row.get('known_issue', ''), summary_time_ms=1000*float(row['summary_seconds_median']),
            source_file='data/coreppl_summary.csv', source_results=row.get('source_results', ''),
            run_origin='historical CorePPL; not rerun'))
    for point in points:
        if not (point['median_time_ms'] > 0 and math.isfinite(point['median_time_ms']) and
                point['standardized_rmse'] >= 0 and math.isfinite(point['standardized_rmse'])):
            raise ValueError(f'Invalid plotted value: {point}')
    return points, inventory


def partition(points):
    coverage, shared = [], {}
    for model in MODELS:
        sets = {e: {p['particles'] for p in points if p['engine'] == e and p['model'] == model} for e in ENGINES}
        shared[model] = set.intersection(*sets.values())
        if not shared[model]:
            raise ValueError(f'No three-system shared particle count for {model}')
        coverage.append(dict(model=model, **{e.lower().replace(' ', '_') + '_particles':
            ' '.join(map(str, sorted(sets[e]))) for e in ENGINES},
            matched_particles=' '.join(map(str, sorted(shared[model])))))
    matched = [p for p in points if p['particles'] in shared[p['model']]]
    for model in MODELS:
        for engine in ENGINES:
            actual = {p['particles'] for p in matched if p['model'] == model and p['engine'] == engine}
            if actual != shared[model]:
                raise ValueError(f'Unequal matched counts for {model}/{engine}')
    return matched, coverage


def draw(points, output, matched=True):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    fig.subplots_adjust(left=.07, right=.985, bottom=.085, top=.88, wspace=.25, hspace=.39)
    fig.text(.07, .966, 'Miking CorePPL, Siren and SirenC', fontsize=21, weight='bold')
    subtitle = ('Matching particle counts and identical logarithmic axes in every panel' if matched else
                'All measured particle counts, with axes fitted separately to each example')
    fig.text(.07, .937, subtitle, fontsize=12, color='#444444')
    fig.text(.07, .913, 'Connected markers follow increasing particle count.', fontsize=10)
    for model, ax in zip(MODELS, axes.flat):
        for index, engine in enumerate(ENGINES):
            group = sorted([p for p in points if p['model'] == model and p['engine'] == engine], key=lambda p: p['particles'])
            x = [p['median_time_ms'] for p in group]
            y = [max(FLOOR, p['standardized_rmse']) for p in group]
            label = 'Miking static DS' if engine == 'Miking CorePPL' else ('Siren SSI' if engine == 'Siren' else 'SirenC symbolic')
            ax.plot(x, y, marker=MARKERS[index], color=COLORS[index], label=label, markersize=5, linewidth=1.2, alpha=.93)
            indices = [0] if len(group) == 1 else [0, len(group)-1]
            for i in indices:
                # Keep endpoint labels distinct when several errors are at the display floor.
                offset = (0, [7, 19, 31][index]) if len(group) == 1 else ((-5 if i else 5), [7, 19, 31][index])
                ha = 'right' if i and len(group) > 1 else 'left'
                if model == 'wheels' and not matched and index > 0:
                    offset = (7 if i else -7, [7, 19, 31][index])
                    ha = 'left' if i else 'right'
                ax.annotate(str(group[i]['particles']), (x[i], y[i]), xytext=offset, textcoords='offset points',
                    ha=ha, fontsize=7, color=COLORS[index])
        ax.set(xscale='log', yscale='log', title=model + (' *' if model == 'gtree' else ''),
            xlabel='Median inference time (ms)', ylabel='Standardized mean RMSE')
        if matched:
            ax.set_xlim(1e-4, 1e6)
            ax.set_ylim(3e-13, 2e3)
            ax.set_xticks([1e-3, 1e-1, 1e1, 1e3, 1e5])
            ax.set_yticks([1e-12, 1e-9, 1e-6, 1e-3, 1, 1e3])
            ax.xaxis.set_minor_locator(NullLocator())
            ax.yaxis.set_minor_locator(NullLocator())
        else:
            xs = [p['median_time_ms'] for p in points if p['model'] == model]
            ys = [max(FLOOR, p['standardized_rmse']) for p in points if p['model'] == model]
            ax.set_xlim(min(xs)/1.8, max(xs)*1.8)
            ax.set_ylim(min(ys)/2, max(ys)*8 if max(ys) > min(ys)*1.01 else max(ys)*100)
        ax.grid(True, alpha=.2)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=8, loc='upper right' if model in ['outlier', 'slam'] else 'center right')
        if model == 'wheels':
            text = 'Shared particles: 1 only' if matched else 'Siren/SirenC: 1; Miking: 1–10,000'
            ax.text(.025, .2, text, va='top', transform=ax.transAxes, fontsize=8)
    ax = axes.flat[-1]
    ax.axis('off')
    ax.text(0, .98,
        'Endpoint labels denote particle counts.\nErrors below 10⁻¹² are displayed at 10⁻¹².\n\n'
        'Siren and SirenC use symbolic component means.\nMiking uses retained posterior moments where\n'
        'available; wheels, outlier, SLAM and hybrid\nretain point-particle readout.\n\n'
        '* Miking static gtree has recorded posterior bias.\nHybrid mean error alone cannot verify its\nmixture weights.\n\n'
        'All plotted points summarize five successful runs.\nSee README.md for sources and timing boundaries.',
        va='top', fontsize=10, linespacing=1.55)
    fig.text(.07, .032,
        'Siren/SirenC timing includes particle creation, execution and weighted means. Miking inference excludes final mixture aggregation.',
        fontsize=9, color='#444444')
    fig.text(.07, .015,
        'Siren/SirenC rerun sequentially; CorePPL was not rerun. Cross-system timing boundaries and machine conditions differ.',
        fontsize=9, color='#444444')
    stem = 'comparison_matched_particles' if matched else 'comparison_miking_style_all_counts'
    for ext in ['png', 'svg']:
        fig.savefig(output / (stem + '.' + ext), dpi=200, facecolor='white')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'graphs', help='Output directory (default: graphs beside this script)')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    points, inventory = aggregate()
    matched, coverage = partition(points)
    draw(matched, args.output, True)
    draw(points, args.output, False)
    write(args.output / 'matched_plot_data.csv', matched)
    write(args.output / 'all_plot_data.csv', points)
    write(args.output / 'particle_coverage.csv', coverage)
    write(args.output / 'repository_row_inventory.csv', inventory)
    manifest = [dict(input_file=str(p.relative_to(ROOT)), rows=len(read(p)), sha256=hashlib.sha256(p.read_bytes()).hexdigest())
        for p in sorted((ROOT / 'data').glob('*.csv'))]
    (args.output / 'input_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Generated both figures (PNG and SVG) in {args.output.resolve()}')
    print(f'{len(points)} available points; {len(matched)} matched points; {len(MODELS)} models.')


if __name__ == '__main__':
    main()

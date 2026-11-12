"""Benchmark generated SirenC programs without executing their top-level run calls.

By default, run five repetitions at 1, 10, 100, 1000, and 10000 particles
for models other than wheels. Wheels uses 1, 2, and 3 particles without a
timeout. Each repetition has one unrecorded warm-up.
Write raw.csv, summary.csv, config.json, and runtime/error plots (PNG/PDF)
to output/sirenc, or to the directory supplied with --output.
"""
import argparse
import ast
import csv
import json
import math
import hashlib
from functools import lru_cache
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import numpy as np
from benchmark_protocol import VERSION, export_comparison, metrics, run_seed
from benchmark_reference import reference as exact_reference

ROOT = Path(__file__).resolve().parent
EXAMPLES = ROOT / 'examples' / 'libsiren-examples'
MODELS = ('gaussian-gaussian', 'gtree', 'kalman', 'wheels', 'slam', 'outlier', 'coin', 'gaussian-gaussian-hybrid')
DEFAULT_PARTICLES = (1, 10, 100, 1000, 10000)
WHEELS_PARTICLES = (1, 2, 3)
FIELDS = ('protocol', 'readout', 'benchmark', 'particles', 'repeat', 'seed', 'status', 'time_ms', 'standardized_mse', 'mean_rmse', 'mean_standardized_rmse', 'mean_max_abs_error', 'estimate', 'error')


@lru_cache(maxsize=None)
def reference(name):
    truth = exact_reference(name)
    return np.array(truth['mean']), np.array(truth['variance'])


def load_program(name):
    path = EXAMPLES / f'{name}-compiled.py'
    tree = ast.parse(path.read_text(), filename=str(path))
    # Only definitions/imports: generated scripts contain unconditional run(..., 1000).
    tree.body = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef))]
    namespace = {'__file__': str(path), '__name__': 'sirenc_benchmark_program'}
    exec(compile(tree, str(path), 'exec'), namespace)
    return namespace


def flatten_mean(value, state):
    if isinstance(value, (tuple, list)):
        return [number for item in value for number in flatten_mean(item, state)]
    return [float(state.mean(value))]


def measure(name, count, seed):
    namespace = load_program(name)
    from siren.inference.ssi import SSIState
    from siren.libsiren import Particle

    program = namespace['program']
    truth, variance = reference(name)
    if not (np.all(np.isfinite(truth)) and np.all(np.isfinite(variance))
            and np.all(variance > 0)):
        raise ValueError('Invalid exact reference')

    def run_once(run_seed):
        # Include particle creation, execution, and weighted mean evaluation.
        start = time.perf_counter_ns()
        particle_seeds = run_seed.spawn(count)
        particles = [
            Particle(program, SSIState(seed=child),
                     0., False)
            for child in particle_seeds
        ]
        for particle in particles:
            particle.cont = program(particle)
            if not particle.finished:
                raise RuntimeError('Program did not finish')
        values = np.array([flatten_mean(p.cont, p.state) for p in particles])
        scores = np.array([p.score for p in particles])
        # Negative infinity is a valid zero-weight particle.
        if (np.any(np.isnan(scores)) or np.any(np.isposinf(scores))
                or not np.any(np.isfinite(scores))):
            raise ValueError('Invalid or all-zero particle weights')
        weights = np.exp(scores - scores.max())
        estimate = np.average(values, axis=0, weights=weights)
        elapsed = (time.perf_counter_ns() - start) / 1e6
        return estimate, elapsed

    warmup_seed = np.random.SeedSequence(seed + 1)
    measured_seed = np.random.SeedSequence(seed)
    run_once(warmup_seed)
    estimate, elapsed = run_once(measured_seed)
    if estimate.shape != truth.shape or not np.all(np.isfinite(estimate)):
        raise ValueError(f'Invalid estimate shape/value: {estimate}')
    if not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError(f'Invalid inference time: {elapsed}')
    return dict(time_ms=elapsed, **metrics(name, estimate.tolist()),
                estimate=estimate.tolist())


def raw_mean_rmse(row):
    # Older CSVs lack mean_rmse but retain enough data to reconstruct it.
    if row.get('mean_rmse') not in (None, ''):
        return float(row['mean_rmse'])
    estimate = np.asarray(json.loads(row['estimate']), dtype=float)
    truth, _ = reference(row['benchmark'])
    if estimate.shape != truth.shape or not np.all(np.isfinite(estimate)):
        raise ValueError('Invalid estimate in legacy CSV')
    return float(np.sqrt(np.mean((estimate - truth)**2)))


def plot(csv_path, output):
    # Failed runs can store tracebacks larger than the default CSV field limit.
    csv.field_size_limit(sys.maxsize)
    os.environ.setdefault('MPLCONFIGDIR', str(output / '.matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with csv_path.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    export_comparison(rows, output, 'sirenc')
    with (output / 'wheels_raw.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys() if rows else FIELDS)
        writer.writeheader()
        writer.writerows(r for r in rows if r['benchmark'] == 'wheels')
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    def panel(name, ax):
        points = []
        for count in sorted({int(r['particles']) for r in rows if r['benchmark'] == name}):
            valid = [r for r in rows if r['benchmark'] == name and int(r['particles']) == count and r['status'] == 'ok']
            if valid:
                points.append((count, statistics.median(float(r['time_ms']) for r in valid),
                               max(1e-12, math.sqrt(statistics.mean(float(r['standardized_mse']) for r in valid)))))
        if points:
            ax.plot([p[1] for p in points], [p[2] for p in points], 'o-', color='tab:green', label='SirenC (symbolic means)')
            for p in [points[0]] + ([points[-1]] if len(points) > 1 else []):
                ax.annotate(str(p[0]), (p[1], p[2]), xytext=(5, 8), textcoords='offset points')
            ax.legend(loc='center right')
        else:
            ax.text(.5, .5, 'No successful runs\nSee raw.csv for failures', ha='center', transform=ax.transAxes)
        ax.set(xscale='log', yscale='log', title=name, xlabel='Median inference time (ms)', ylabel='Standardized mean RMSE')
        ax.set_ylim(3e-13, max(10., max((p[2] for p in points), default=1.) * 5))
        ax.margins(x=.12)
        ax.grid(alpha=.2)
    for name, ax in zip(MODELS, axes.flat):
        panel(name, ax)
        individual, single = plt.subplots(figsize=(7, 5))
        panel(name, single)
        individual.tight_layout()
        individual.savefig(output / f'{name}.png', dpi=180)
        plt.close(individual)
    axes.flat[-1].axis('off')
    axes.flat[-1].text(0, .95, 'Endpoint labels denote particle counts.\nErrors below 1e-12 are displayed at 1e-12.\n\nRMSE uses exact posterior marginal standard deviations.\nTiming includes particle creation, execution, and weighted\nsymbolic mean evaluation; excludes imports, program loading,\nwarm-up, and error calculation.\n\nGenerated programs and inference interfaces are unchanged.\nReference values are used only to score estimates.\nFailures are recorded; successful execution alone does\nnot establish posterior accuracy.', va='top', linespacing=1.6)
    with (output / 'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['benchmark', 'particles', 'successful_runs', 'failed_runs', 'median_time_ms', 'standardized_rmse', 'mean_rmse_rms'])
        writer.writeheader()
        for name in MODELS:
            for count in sorted({int(r['particles']) for r in rows if r['benchmark'] == name}):
                group = [r for r in rows if r['benchmark'] == name and int(r['particles']) == count]
                valid = [r for r in group if r['status'] == 'ok']
                writer.writerow(dict(benchmark=name, particles=count, successful_runs=len(valid), failed_runs=len(group)-len(valid),
                                     median_time_ms=statistics.median(float(r['time_ms']) for r in valid) if valid else '',
                                     standardized_rmse=math.sqrt(statistics.mean(float(r['standardized_mse']) for r in valid)) if valid else '',
                                     mean_rmse_rms=math.sqrt(statistics.mean(raw_mean_rmse(r)**2 for r in valid)) if valid else ''))
    with (output / 'summary.csv').open(newline='') as stream:
        reader = csv.DictReader(stream)
        summary = list(reader)
        with (output / 'wheels_summary.csv').open('w', newline='') as target:
            writer = csv.DictWriter(target, fieldnames=reader.fieldnames)
            writer.writeheader()
            writer.writerows(r for r in summary if r['benchmark'] == 'wheels')
    fig.tight_layout()
    for extension in ('png', 'pdf'):
        fig.savefig(output / f'sirenc_runtime_error.{extension}', dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmarks', nargs='+', choices=MODELS, default=list(MODELS))
    parser.add_argument('--particles', nargs='+', type=int, default=DEFAULT_PARTICLES,
                        help='Particle counts for examples other than wheels (default: 1 10 100 1000 10000)')
    parser.add_argument('--wheels-particles', nargs='+', type=int, choices=range(1, 11), default=WHEELS_PARTICLES,
                        help='Wheels particle counts (default: 1 2 3)')
    parser.add_argument('--all-particles', action='store_true', help='Every integer from 1 through 10000 for examples other than wheels')
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--timeout', type=float, default=120, help='Seconds per repetition including warm-up, except wheels (no timeout)')
    parser.add_argument('--output', type=Path, default=ROOT / 'output' / 'sirenc')
    parser.add_argument('--plot-only', action='store_true')
    parser.add_argument('--worker', nargs=3, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        name, count, seed = args.worker
        print(json.dumps(measure(name, int(count), int(seed))))
        return
    counts = list(range(1, 10001)) if args.all_particles else sorted(set(args.particles))
    particles_by_benchmark = {
        name: sorted(set(args.wheels_particles)) if name == 'wheels' else counts
        for name in args.benchmarks
    }
    if args.repeats < 1 or args.timeout <= 0 or any(p < 1 or p > 10000 for p in counts):
        parser.error('Require positive repeats/timeout and particle counts in 1..10000')
    args.output.mkdir(parents=True, exist_ok=True)
    csv_path = args.output / 'raw.csv'
    if not args.plot_only:
        with csv_path.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            for name in args.benchmarks:
                for count in particles_by_benchmark[name]:
                    for repeat in range(args.repeats):
                        seed = run_seed(args.seed, repeat)
                        row = dict(protocol=VERSION, readout='symbolic', benchmark=name, particles=count, repeat=repeat, seed=seed, status='ok')
                        try:
                            result = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker', name, str(count), str(seed)], capture_output=True, text=True, timeout=None if name == 'wheels' else args.timeout, check=True)
                            row.update(json.loads(result.stdout))
                        except subprocess.TimeoutExpired:
                            row.update(status='timeout', error=f'Exceeded {args.timeout}s')
                        except subprocess.CalledProcessError as exc:
                            row.update(status='error', error=exc.stderr.strip())
                        writer.writerow(row)
                        stream.flush()
                        print(f'{name}: particles={count} repeat={repeat+1} {row["status"]}', flush=True)
        (args.output / 'config.json').write_text(json.dumps(dict(protocol=VERSION, benchmarks=args.benchmarks, particles_by_benchmark=particles_by_benchmark, repeats=args.repeats, seed=args.seed, timeout=args.timeout,
            wheels_timeout=None, warmups_per_repetition=1, seed_policy='SeedSequence child streams per particle',
            timing_policy='particle creation + execution + weighted mean',
            estimator='weighted symbolic component means'), indent=2))
        sources = [Path(__file__), ROOT / 'siren/libsiren.py', ROOT / 'siren/inference/interface.py',
                   ROOT / 'benchmark_reference.py', ROOT / 'benchmark_protocol.py']
        sources += [EXAMPLES / f'{name}-compiled.py' for name in args.benchmarks]
        (args.output / 'source_hashes.json').write_text(json.dumps(
            {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
             for path in sources}, indent=2))
    plot(csv_path, args.output)
    print(f'Results: {args.output}')


if __name__ == '__main__':
    main()

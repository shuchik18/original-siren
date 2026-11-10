"""Benchmark source libsiren examples in symbolic and eager sampling modes."""
import argparse
import ast
import csv
import math
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent
DEFAULTS = ['example1', 'example5', 'example9', 'example10', 'example12',
            'gaussian-gaussian', 'kalman']
FIELDS = ['example', 'mode', 'particles', 'iteration', 'seed', 'runtime_s',
          'estimate', 'reference', 'absolute_error', 'squared_error', 'ess',
          'status', 'message']


def reference(name):
    fixed = {'example1': 0., 'example5': .5, 'example9': 300 / 101,
             'example10': 300 / 105, 'gaussian-gaussian': 30 / 11}
    if name in fixed:
        return fixed[name]
    if name in ('example12', 'kalman'):
        mean, variance = 0., 100.
        for observation in range(1, 3 if name == 'example12' else 10):
            variance += 1
            gain = variance / (variance + 1)
            mean += gain * (observation - mean)
            variance *= 1 - gain
        return mean
    raise ValueError(f'No analytic reference for {name}; use --reference VALUE')


def load_program(name, mode):
    # Retain definitions/imports, never execute top-level run/print calls.
    tree = ast.parse((ROOT / 'examples' / 'libsiren-examples' / f'{name}.py').read_text())
    tree.body = [node for node in tree.body
                 if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef))]
    namespace = {'__name__': f'benchmark_{name}'}
    exec(compile(tree, name, 'exec'), namespace)
    from siren import libsiren

    def draw(particle, distribution):
        return particle.state.value(libsiren.assume(particle, distribution))

    # Explicit sample calls always draw; assume stays symbolic only in symbolic mode.
    namespace['sample'] = draw
    if mode == 'sample':
        namespace['assume'] = draw
    return namespace['program']


def measure(program, count, seed):
    import numpy as np
    from siren.libsiren import Particle
    from siren.inference.ssi import SSIState
    rng = np.random.default_rng(seed)
    values, scores = [], []
    start = time.perf_counter()
    for _ in range(count):
        particle = Particle(program, SSIState(seed=int(rng.integers(0, 2**63))), 0., False)
        result = program(particle)
        if not particle.finished:
            raise ValueError('Program did not finish')
        values.append(float(particle.state.mean(result)))
        scores.append(float(particle.score))
    scores = np.asarray(scores)
    if np.isnan(scores).any() or np.isposinf(scores).any() or not np.isfinite(scores).any():
        raise ValueError('Invalid or zero total particle weight')
    weights = np.exp(scores - np.max(scores))
    weights /= weights.sum()
    estimate = float(np.dot(weights, values))
    ess = float(1 / np.dot(weights, weights))
    if not math.isfinite(estimate):
        raise ValueError('Non-finite posterior mean')
    return time.perf_counter() - start, estimate, ess


def plot(csv_path, output):
    os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'output' / '.matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with csv_path.open(newline='') as stream:
        rows = [r for r in csv.DictReader(stream) if r['status'] == 'ok']
    if not rows:
        raise ValueError('CSV contains no successful measurements')
    names = list(dict.fromkeys(r['example'] for r in rows))
    fig, axes = plt.subplots(len(names), 1, figsize=(7, 4 * len(names)), squeeze=False)
    for name, ax in zip(names, axes[:, 0]):
        for mode, marker in [('symbolic', 'o'), ('sample', 'x')]:
            group = [r for r in rows if r['example'] == name and r['mode'] == mode]
            ax.scatter([float(r['runtime_s']) for r in group],
                       [float(r['absolute_error']) for r in group],
                       label=mode, marker=marker, alpha=.75)
        ax.set(title=name, xlabel='Runtime (seconds)', ylabel='Absolute posterior mean error')
        ax.set_xscale('log')
        ax.grid(alpha=.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def positive(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError('Must be positive')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-b', '--benchmarks', default=','.join(DEFAULTS), help='Comma-separated source example names')
    parser.add_argument('-p', '--particles', nargs='+', type=positive, default=[10, 100, 1000])
    parser.add_argument('-n', '--iterations', type=positive, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--reference', type=float, help='Reference posterior mean for a single custom example')
    parser.add_argument('-o', '--output', type=Path, default=ROOT / 'output' / 'examples')
    parser.add_argument('--plot-only', type=Path, help='Regenerate the scatter plot from an existing CSV')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.plot_only:
        plot(args.plot_only, args.output / 'runtime_error.png')
        return
    names = [n.strip().removesuffix('.py') for n in args.benchmarks.split(',') if n.strip()]
    if not names or (args.reference is not None and (len(names) != 1 or not math.isfinite(args.reference))):
        parser.error('--reference requires exactly one example and a finite value')
    try:
        references = {name: reference(name) if args.reference is None else args.reference for name in names}
    except ValueError as exc:
        parser.error(str(exc))
    csv_path = args.output / 'runtime_error.csv'
    failures = 0
    with csv_path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        for name in names:
            for mode in ('symbolic', 'sample'):
                for count in args.particles:
                    for iteration in range(1, args.iterations + 1):
                        seed = args.seed + iteration - 1
                        row = dict(example=name, mode=mode, particles=count, iteration=iteration,
                                   seed=seed, reference=references[name], status='ok', message='')
                        try:
                            program = load_program(name, mode)
                            runtime, estimate, ess = measure(program, count, seed)
                            error = abs(estimate - references[name])
                            row.update(runtime_s=runtime, estimate=estimate, absolute_error=error,
                                       squared_error=error**2, ess=ess)
                        except Exception as exc:
                            row.update(status='error', message=f'{type(exc).__name__}: {exc}')
                            failures += 1
                        writer.writerow(row)
                        stream.flush()
                print(f'{name}: {mode} finished', flush=True)
    plot(csv_path, args.output / 'runtime_error.png')
    print(f'Saved {csv_path} and {args.output / "runtime_error.png"}; {failures} failed runs')
    return bool(failures)


if __name__ == '__main__':
    raise SystemExit(main())

"""Paired sequential timing; plot only after every worker has exited."""
import csv
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
ORIGINAL = Path('/Users/skundaliya/Documents/Originial Siren/siren-main')
MODELS = ('gaussian-gaussian', 'gtree', 'kalman', 'coin', 'gaussian-gaussian-hybrid', 'wheels')
ENV = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
           MKL_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
           MPLBACKEND='Agg', MPLCONFIGDIR=str(OUT / '.matplotlib'), SIREN_REPO=str(ORIGINAL))


def hashes():
    paths = list((ROOT / 'siren').rglob('*.py')) + list((ORIGINAL / 'siren').rglob('*.py'))
    paths += [ROOT / f'benchmark_{engine}.py' for engine in ('siren', 'sirenc')]
    paths += [ROOT / 'benchmark_reference.py', ROOT / 'benchmark_protocol.py']
    paths += [ROOT / 'examples/libsiren-examples' / f'{name}-compiled.py' for name in MODELS]
    paths += [ORIGINAL / 'examples' / f'{name}.si' for name in MODELS]
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def main():
    before = hashes()
    config = dict(models=MODELS, particles=[10, 100, 1000], repeats=5,
                  wheels_particles=[1], wheels_repeats=3, base_seed=1234,
                  order='Siren first on even repetitions; SirenC first on odd repetitions',
                  parallel_workers=1, warmups_per_worker=1,
                  timing='particle creation + inference + weighted posterior mean; startup/imports/parsing excluded',
                  thread_limits={k: ENV[k] for k in ENV if k.endswith('NUM_THREADS') or k == 'VECLIB_MAXIMUM_THREADS'},
                  initial_load_average=os.getloadavg(), started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                  environment_note='No competing Python benchmark workers observed before launch; normal apps and OS services remain active.')
    (OUT / 'config.json').write_text(json.dumps(config, indent=2))
    (OUT / 'source_hashes.json').write_text(json.dumps(before, indent=2))
    fields = ['benchmark', 'particles', 'repeat', 'seed', 'engine', 'order', 'status', 'time_ms',
              'mean_rmse', 'mean_standardized_rmse', 'estimate', 'worker_wall_seconds', 'load_before', 'load_after', 'error']
    rows = []
    with (OUT / 'raw.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for name in MODELS:
            for count in ([1] if name == 'wheels' else [10, 100, 1000]):
                for repeat in range(3 if name == 'wheels' else 5):
                    engines = ['siren', 'sirenc'] if repeat % 2 == 0 else ['sirenc', 'siren']
                    for order, engine in enumerate(engines):
                        seed = 1234 + repeat * 10000
                        row = dict(benchmark=name, particles=count, repeat=repeat, seed=seed,
                                   engine=engine, order=order, status='ok', load_before=json.dumps(os.getloadavg()))
                        start = time.monotonic()
                        print(f'{name} n={count} repetition={repeat+1} {engine} starting', flush=True)
                        try:
                            result = subprocess.run([sys.executable, str(ROOT / f'benchmark_{engine}.py'), '--worker', name, str(count), str(seed)],
                                                    cwd=ROOT, env=ENV, capture_output=True, text=True,
                                                    check=True, timeout=360 if name == 'wheels' else 90)
                            data = json.loads(result.stdout)
                            row.update({key: data[key] for key in ('time_ms', 'mean_rmse', 'mean_standardized_rmse')})
                            row['estimate'] = json.dumps(data['estimate'])
                        except subprocess.TimeoutExpired:
                            row.update(status='timeout', error='Worker exceeded time limit, including warmup')
                        except (subprocess.CalledProcessError, ValueError, KeyError) as exc:
                            row.update(status='error', error=getattr(exc, 'stderr', str(exc)))
                        row.update(worker_wall_seconds=time.monotonic()-start, load_after=json.dumps(os.getloadavg()))
                        rows.append(row)
                        writer.writerow(row)
                        stream.flush()
                        print(f'  {row["status"]}: {row.get("time_ms", "")} ms', flush=True)
    after = hashes()
    changed = [p for p, digest in before.items() if after.get(p) != digest]
    (OUT / 'source_validation.json').write_text(json.dumps(dict(changed_sources=changed), indent=2))
    if changed:
        raise RuntimeError('Sources changed during measurement; results must not be compared')
    summaries = []
    comparisons = []
    for name in MODELS:
        for count in ([1] if name == 'wheels' else [10, 100, 1000]):
            group = [r for r in rows if r['benchmark'] == name and r['particles'] == count]
            pair = {}
            for engine in ('siren', 'sirenc'):
                runs = [r for r in group if r['engine'] == engine and r['status'] == 'ok']
                vals = [r['time_ms'] for r in runs]
                summary = dict(benchmark=name, particles=count, engine=engine, successful_runs=len(vals),
                               failed_runs=sum(r['status'] != 'ok' for r in group if r['engine'] == engine),
                               mean_ms=statistics.fmean(vals) if vals else '', median_ms=statistics.median(vals) if vals else '',
                               std_ms=statistics.stdev(vals) if len(vals)>1 else 0,
                               max_standardized_rmse=max((r['mean_standardized_rmse'] for r in runs), default=''))
                summaries.append(summary)
                pair[engine] = summary
            differences = []
            for repeat in range(3 if name == 'wheels' else 5):
                runs = [r for r in group if r['repeat'] == repeat and r['status'] == 'ok']
                if len(runs) == 2:
                    differences.append(max(abs(a-b) for a,b in zip(json.loads(runs[0]['estimate']), json.loads(runs[1]['estimate']))))
            comparisons.append(dict(benchmark=name, particles=count,
                                    mean_speedup=pair['siren']['mean_ms']/pair['sirenc']['mean_ms'] if all(pair[e]['mean_ms'] for e in pair) else '',
                                    median_speedup=pair['siren']['median_ms']/pair['sirenc']['median_ms'] if all(pair[e]['median_ms'] for e in pair) else '',
                                    max_paired_estimate_difference=max(differences, default='')))
    for filename, data in [('summary.csv', summaries), ('speedup.csv', comparisons)]:
        with (OUT / filename).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]))
            writer.writeheader(); writer.writerows(data)
    os.environ.update({k: ENV[k] for k in ('MPLBACKEND', 'MPLCONFIGDIR')})
    import matplotlib.pyplot as plt
    for metric, label in [('mean_ms', 'Mean'), ('median_ms', 'Median')]:
        fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), layout='constrained')
        for ax, name in zip(axes.flat, MODELS):
            for engine, color in [('siren', '#616b77'), ('sirenc', '#007f73')]:
                points = [r for r in summaries if r['benchmark'] == name and r['engine'] == engine and r[metric] != '']
                ax.errorbar([r['particles'] for r in points], [r[metric] for r in points],
                            yerr=[r['std_ms'] for r in points] if metric == 'mean_ms' else None,
                            marker='o', capsize=3, label='SirenC' if engine == 'sirenc' else 'Siren', color=color)
            ax.set(title=name, xlabel='Particles', ylabel=f'{label} inference time (ms)', yscale='log')
            if name != 'wheels': ax.set_xscale('log')
            else: ax.set_xticks([1])
            ax.grid(alpha=.2); ax.legend()
        fig.suptitle(f'{label} execution time — sequential paired runs\n5 repetitions; Wheels: 3 repetitions at 1 particle' + ('; error bars: ±1 sample SD' if metric == 'mean_ms' else ''), fontsize=13)
        fig.savefig(OUT / f'{metric}.png', dpi=180)
        fig.savefig(OUT / f'{metric}.pdf')
        plt.close(fig)
    print(json.dumps(comparisons, indent=2), flush=True)


if __name__ == '__main__':
    main()

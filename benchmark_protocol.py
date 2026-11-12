"""Common contract for SirenC, Original Siren, and CorePPL benchmark results."""
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
from benchmark_reference import reference

VERSION = 'siren-comparison-v1'
FIELDS = ('protocol', 'engine', 'benchmark', 'particles', 'repeat', 'seed', 'status',
          'time_ms', 'mean_rmse', 'mean_standardized_rmse', 'mean_max_abs_error',
          'estimate', 'readout', 'known_issue', 'reference_sha256', 'error')


def particle_counts(name, counts=(1, 10, 100, 1000, 10000)):
    return [1, 2, 3] if name == 'wheels' else list(counts)


def run_seed(base, repeat, warmup=False):
    return base + repeat * 10000 + (1 if warmup else 0)


def reference_hash(name):
    truth = reference(name)
    return hashlib.sha256(json.dumps({k: truth[k] for k in ('mean', 'variance')},
                                     sort_keys=True).encode()).hexdigest()


def metrics(name, estimate):
    truth = reference(name)
    if len(estimate) != len(truth['mean']) or not all(math.isfinite(x) for x in estimate):
        raise ValueError('Invalid estimate dimension or value')
    differences = [x-y for x, y in zip(estimate, truth['mean'])]
    mse = statistics.fmean(x*x/v for x, v in zip(differences, truth['variance']))
    return dict(standardized_mse=mse,
                mean_rmse=math.sqrt(statistics.fmean(x*x for x in differences)),
                mean_standardized_rmse=math.sqrt(mse),
                mean_max_abs_error=max(map(abs, differences)))


def export_comparison(rows, output, engine):
    """Write the same comparison files without discarding failure records."""
    normalized = []
    for row in rows:
        name = row.get('benchmark', row.get('model'))
        item = {key: row.get(key, '') for key in FIELDS}
        item.update(engine=row.get('engine', engine), benchmark=name,
                    reference_sha256=reference_hash(name))
        if row.get('status') == 'ok':
            estimate = row.get('estimate', row.get('mean'))
            if isinstance(estimate, str):
                estimate = json.loads(estimate)
            item.update({k: v for k, v in metrics(name, estimate).items() if k in FIELDS})
            item['estimate'] = json.dumps(estimate)
            item['time_ms'] = row.get('time_ms', 1000 * float(row.get('inference_seconds', 0)))
        normalized.append(item)
    output = Path(output)
    with (output / 'comparison.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(normalized)
    fields = ('protocol', 'engine', 'benchmark', 'particles', 'readout', 'known_issue',
              'successful_runs', 'failed_runs', 'median_time_ms', 'mean_rmse_rms',
              'mean_standardized_rmse_rms', 'reference_sha256')
    groups = {}
    for row in normalized:
        key = tuple(row[k] for k in ('protocol', 'engine', 'benchmark', 'particles', 'readout', 'known_issue'))
        groups.setdefault(key, []).append(row)
    with (output / 'comparison_summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for key, group in groups.items():
            valid = [r for r in group if r['status'] == 'ok']
            summary = dict(zip(fields[:6], key))
            summary.update(successful_runs=len(valid), failed_runs=sum(r['status'] in ('error', 'timeout', 'runtime_error', 'compile_error', 'invalid_output') for r in group),
                           median_time_ms=statistics.median(float(r['time_ms']) for r in valid) if valid else '',
                           reference_sha256=group[0]['reference_sha256'])
            for metric in ('mean_rmse', 'mean_standardized_rmse'):
                summary[metric + '_rms'] = math.sqrt(statistics.fmean(float(r[metric])**2 for r in valid)) if valid else ''
            writer.writerow(summary)

"""Measure missing one-particle points sequentially, then refresh bundled inputs."""
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

OUT = Path(__file__).resolve().parent
ROOT = Path('/Users/skundaliya/Documents/libsiren/original-siren')
BASE = ROOT / 'output/sequential-comparison-2026-11-12'
MODELS = ('coin', 'gaussian-gaussian', 'gtree', 'kalman', 'gaussian-gaussian-hybrid')
ENV = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
           MKL_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
           MPLBACKEND='Agg', MPLCONFIGDIR=str(OUT / '.matplotlib'),
           SIREN_REPO='/Users/skundaliya/Documents/Originial Siren/siren-main')


def read(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def write(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    baseline = json.loads((BASE / 'source_hashes.json').read_text())
    before = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in baseline}
    assert before == baseline, 'Sources changed since the larger-particle measurements'
    rows = read(OUT / 'data/sequential_raw.csv')
    supplemental = OUT / 'data/particle_one_raw.csv'
    additions = read(supplemental) if supplemental.exists() else []
    existing = {(r['benchmark'], int(r['particles']), int(r['repeat']), r['engine']) for r in rows + additions}
    started = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    for model in MODELS:
        for repeat in range(5):
            engines = ('siren', 'sirenc') if repeat % 2 == 0 else ('sirenc', 'siren')
            for order, engine in enumerate(engines):
                if (model, 1, repeat, engine) in existing:
                    continue
                seed = 1234 + 10000 * repeat
                row = dict(benchmark=model, particles=1, repeat=repeat, seed=seed, engine=engine,
                           order=order, status='ok', load_before=json.dumps(os.getloadavg()))
                start = time.monotonic()
                result = subprocess.run([sys.executable, str(ROOT / f'benchmark_{engine}.py'),
                                         '--worker', model, '1', str(seed)], cwd=ROOT, env=ENV,
                                        capture_output=True, text=True, check=True, timeout=90)
                data = json.loads(result.stdout)
                row.update({key: data[key] for key in ('time_ms', 'mean_rmse', 'mean_standardized_rmse')})
                row.update(estimate=json.dumps(data['estimate']), worker_wall_seconds=time.monotonic()-start,
                           load_after=json.dumps(os.getloadavg()), error='')
                additions.append(row)
                write(supplemental, additions)
                print(f'{model} n=1 repeat={repeat+1} {engine}: {row["time_ms"]:.6f} ms', flush=True)
    after = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in baseline}
    assert after == before, 'Sources changed during measurement'
    assert len(additions) == 50
    rows = [r for r in rows if (r['benchmark'], int(r['particles']), int(r['repeat']), r['engine'])
            not in {(a['benchmark'], int(a['particles']), int(a['repeat']), a['engine']) for a in additions}] + additions
    write(OUT / 'data/sequential_raw.csv', rows)
    for engine in ('siren', 'sirenc'):
        historical = [dict(r, run_origin='historical; not rerun') for r in
                      read(OUT / 'data' / f'historical_{engine}_raw.csv') if r['benchmark'] in ('slam', 'outlier')]
        fresh = [dict(r, standardized_mse=float(r['mean_standardized_rmse'])**2,
                      run_origin='sequential paired run 2026-11-12') for r in rows if r['engine'] == engine]
        write(OUT / 'data' / f'{engine}_raw.csv', historical + fresh)
    (OUT / 'particle_one_validation.json').write_text(json.dumps(dict(
        started_utc=started, supplemental_runs=len(additions), particles=1, repeats=5,
        parallel_workers=1, warmups_per_worker=1, same_sources_as_original_sequential_run=True,
        sources_unchanged_during_measurement=True, source_hashes=before), indent=2)+'\n')
    print('Added all 50 measured one-particle runs; all source hashes match.', flush=True)


if __name__ == '__main__':
    main()

"""Rebuild the three-way plots from this completed Siren/SirenC session."""
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics
import subprocess
import sys

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
PREVIOUS = ROOT / 'output/three-way-fresh-timing'
PACKAGE = OUT / 'comparison-package'


def read(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def write(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    validation = json.loads((OUT / 'source_validation.json').read_text())
    assert not validation['changed_sources'], 'Sources changed during measurement'
    rows = read(OUT / 'raw.csv')
    assert len(rows) == 290 and all(r['status'] == 'ok' for r in rows)
    groups = {}
    for row in rows:
        groups.setdefault((row['benchmark'], row['engine'], int(row['particles'])), []).append(row)
    assert len(groups) == 58 and all(len(group) == 5 for group in groups.values())
    (PACKAGE / 'data').mkdir(parents=True, exist_ok=True)
    (PACKAGE / 'graphs').mkdir(exist_ok=True)
    for engine in ('siren', 'sirenc'):
        data = [dict(r, standardized_mse=float(r['mean_standardized_rmse'])**2,
                     run_origin='complete sequential rerun 2026-11-12') for r in rows if r['engine'] == engine]
        write(PACKAGE / 'data' / f'{engine}_raw.csv', data)
    shutil.copy2(OUT / 'raw.csv', PACKAGE / 'data/sequential_raw.csv')
    for filename in ('coreppl_summary.csv', 'coreppl_posterior_summary.csv'):
        shutil.copy2(PREVIOUS / 'data' / filename, PACKAGE / 'data' / filename)
    code = (PREVIOUS / 'plot_comparison.py').read_text()
    code = code.replace('Siren/SirenC: 5 runs per point; Wheels: 3.', 'All plotted points summarize five successful runs.')
    (PACKAGE / 'plot_comparison.py').write_text(code)
    shutil.copy2(PREVIOUS / 'requirements.txt', PACKAGE / 'requirements.txt')
    (PACKAGE / 'source_manifest.json').write_text(json.dumps(dict(
        siren_and_sirenc_source=str(OUT / 'raw.csv'), successful_runs=290,
        repetitions=5, particles=[1, 10, 100, 1000], wheels_particles=[1],
        all_eight_models_fresh=True, coreppl_rerun=False,
        coreppl_policy='Existing measurements retained at explicit user request',
        coreppl_sha256=hashlib.sha256((PACKAGE / 'data/coreppl_summary.csv').read_bytes()).hexdigest(),
        source_validation=str(OUT / 'source_validation.json')), indent=2)+'\n')
    (PACKAGE / 'README.md').write_text('''# Consistent Siren / SirenC rerun and three-way comparison

Every Siren and SirenC point is from one complete sequential benchmark session
on November 12, 2026. All eight models were rerun; no earlier Outlier, SLAM,
or other Siren/SirenC measurements are mixed into these graphs.

Seven models use 1, 10, 100, and 1,000 particles; Wheels uses one particle.
Every engine/model/particle group contains five successful measured repetitions.
Each worker has an independent untimed warm-up. Engine order alternates between
repetitions. Both engines use paired seed labels and independent particle RNG
streams, a shared Python environment, and one thread per numerical library.
One worker runs at a time; all plotting occurs after measurement. Normal apps
and OS services remain active, so a completely idle machine is not claimed.

CorePPL's existing measurements were retained at the user's explicit request.
They were measured separately; its compiler toolchain is unavailable locally.
The matched figure uses identical particle counts across all three systems.
The all-counts figure also shows CorePPL's separately measured 10,000-particle
points and larger Wheels counts. No corresponding Siren/SirenC points were
invented. All plotted points summarize five runs.

The horizontal axis is median inference time in milliseconds (left is faster).
The vertical axis is RMS standardized posterior-mean error across repetitions.
Errors below 1e-12 are floored for display only. Timing and error are both
recomputed from the same raw runs for every Siren/SirenC point.

Siren/SirenC timing includes particle creation, execution, and weighted mean
evaluation, and excludes startup, imports, parsing, error scoring, and plotting.
CorePPL's recorded inference timer excludes final mixture aggregation. These
timer boundaries and separately recorded machine conditions preclude a fully
controlled three-system speedup claim.

CorePPL static gtree retains its documented posterior bias. CorePPL readout uses
retained moments where available, and samples for Wheels, Outlier, SLAM, and
the hybrid. Siren/SirenC use symbolic component means. The hybrid's zero mean
cannot verify mixture weights. Only returned posterior means are compared here;
matching means does not establish equal posterior distributions or evidence.

Run `python plot_comparison.py` to regenerate both PNG/SVG figures from the
bundled CSVs. `data/sequential_raw.csv` preserves the complete 290-run session.
`validation.json` independently checks every timing/error aggregate and particle
intersection. Original implementation sources were unchanged during the run.
''')
    subprocess.run([sys.executable, str(PACKAGE / 'plot_comparison.py')], check=True)
    points = read(PACKAGE / 'graphs/all_plot_data.csv')
    matched = read(PACKAGE / 'graphs/matched_plot_data.csv')
    assert len(points) == 98 and len(matched) == 87
    for (model, engine, count), group in groups.items():
        point = next(p for p in points if p['model'] == model and p['engine'].lower() == engine and int(p['particles']) == count)
        assert float(point['median_time_ms']) == statistics.median(float(r['time_ms']) for r in group)
        expected = math.sqrt(statistics.fmean(float(r['mean_standardized_rmse'])**2 for r in group))
        assert math.isclose(float(point['standardized_rmse']), expected, rel_tol=1e-12, abs_tol=1e-25)
        assert int(point['successful_runs']) == 5
    for model in {r['benchmark'] for r in rows}:
        expected = {1} if model == 'wheels' else {1, 10, 100, 1000}
        for engine in ('Siren', 'SirenC', 'Miking CorePPL'):
            assert {int(p['particles']) for p in matched if p['model'] == model and p['engine'] == engine} == expected
    result = dict(all_eight_siren_sirenc_models_rerun=True, fresh_runs=290,
                  groups_checked=58, repetitions_per_group=5,
                  all_timing_and_error_aggregates_match=True,
                  all_counts_points=98, matched_points=87, coreppl_rerun=False,
                  coreppl_unchanged=(PACKAGE / 'data/coreppl_summary.csv').read_bytes() == (PREVIOUS / 'data/coreppl_summary.csv').read_bytes(),
                  sources_unchanged=True)
    (PACKAGE / 'validation.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

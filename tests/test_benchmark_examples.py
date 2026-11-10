"""Regression checks for paired discovery and mode-specific exports."""
import csv
from pathlib import Path
import pytest

import benchmark_examples as benchmark


def test_slam_reference_matches_exact_robot_path_probabilities():
    # The three paths (0,0), (1,0), (1,1) have prior probabilities
    # .1, .81, .09; the probability of two false observations is .2804.
    assert benchmark.reference('slam') == pytest.approx(
        [98 / 701, 103 / 701, 1197 / 1402, 369 / 2804])


def test_discovery_includes_every_python_file(tmp_path, monkeypatch):
    names = ['example9.py', 'example-9-compiled.py', 'example-9-desired.py']
    for name in names + ['notes.txt']:
        (tmp_path / name).touch()
    monkeypatch.setattr(benchmark, 'EXAMPLE_DIR', tmp_path)
    assert {p.name for p in benchmark.discover()} == set(names)


def test_selecting_either_filename_includes_both_implementations():
    paths = [Path('example9.py'), Path('example-9-compiled.py'), Path('example10.py')]
    for selection in ('example9', 'example9.py', 'example-9-compiled.py'):
        selected = benchmark.select_paths(paths, selection)
        assert selected == paths[:2]
        assert [benchmark.implementation_for(p) for p in selected] == ['unoptimized siren', 'sirenC']


def test_split_preserves_both_implementations_and_failures(tmp_path):
    source = tmp_path / 'runtime_error.csv'
    with source.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=benchmark.FIELDS)
        writer.writeheader()
        for mode in ('symbolic', 'sample'):
            for filename in ('example9.py', 'example-9-compiled.py'):
                writer.writerow(dict(example=benchmark.example_key(Path(filename).stem),
                                     filename=filename, mode=mode,
                                     implementation=benchmark.implementation_for(Path(filename)),
                                     status='error' if filename.endswith('-compiled.py') else 'ok'))
    benchmark.split_csv(source, tmp_path)
    for mode in ('symbolic', 'sample'):
        with (tmp_path / f'{mode}_runtime_error.csv').open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == 2
        assert {r['mode'] for r in rows} == {mode}
        assert {r['implementation'] for r in rows} == {'unoptimized siren', 'sirenC'}
        assert {r['status'] for r in rows} == {'ok', 'error'}

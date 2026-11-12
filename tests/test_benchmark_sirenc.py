"""Check the benchmark's RNG contract without changing compiled programs."""
import numpy as np
import pytest

import benchmark_sirenc as benchmark
from siren.libsiren import const


def test_measure_uses_independent_matching_child_streams(monkeypatch):
    draws = []
    def program(particle):
        value = float(particle.state.rng.normal())
        draws.append(value)
        particle.finished = True
        return const(value)
    monkeypatch.setattr(benchmark, 'load_program', lambda name: {'program': program})
    result = benchmark.measure('gaussian-gaussian', 10, 1234)
    expected_warmup = [np.random.default_rng(child).normal()
                       for child in np.random.SeedSequence(1235).spawn(10)]
    expected_measured = [np.random.default_rng(child).normal()
                         for child in np.random.SeedSequence(1234).spawn(10)]
    assert draws == pytest.approx(expected_warmup + expected_measured)
    assert len(set(draws[10:])) == 10
    assert result['estimate'] == pytest.approx([np.mean(expected_measured)])


def test_measure_reproduces_estimates_for_the_same_seed(monkeypatch):
    def program(particle):
        particle.finished = True
        return const(float(particle.state.rng.normal()))
    monkeypatch.setattr(benchmark, 'load_program', lambda name: {'program': program})
    first = benchmark.measure('gaussian-gaussian', 10, 1234)
    second = benchmark.measure('gaussian-gaussian', 10, 1234)
    assert first['estimate'] == second['estimate']
    assert first['standardized_mse'] == second['standardized_mse']

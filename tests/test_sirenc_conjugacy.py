"""Check generated conjugate updates against exact posterior probabilities."""
import math
import ast
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from benchmark_sirenc import flatten_mean, load_program
from benchmark_reference import reference
from siren.inference.ssi import SSIState
from siren.libsiren import Particle, const


def test_compiled_coin_exact_posterior_and_evidence():
    program = load_program('coin')['program']
    particle = Particle(program, SSIState(seed=1234))
    result = program(particle)
    posterior = particle.state.eval_distr(particle.state.distr(result))
    assert posterior.a.v == 8
    assert posterior.b.v == 2
    assert particle.state.mean(result) == pytest.approx(.8)
    assert particle.score == pytest.approx(math.log(1 / 72))
    assert particle.finished


@pytest.mark.parametrize('cell_indices, expected_map, evidence', [
    ([0, 0], [1 / 82, .5], .41),
    ([0, 1], [.1, .1], .25),
])
def test_compiled_slam_updates_without_dependency_cycle(cell_indices, expected_map, evidence):
    namespace = load_program('slam')
    # Fix motion; reorder cells to exercise each parent in the generated update.
    namespace['sample'] = lambda particle, distribution: const(True)
    particle = Particle(None, SSIState(seed=1234))
    cells = [namespace['bernoulli_priors'](particle, const(i)) for i in range(2)]
    accumulator = (const(1.), cells, [])
    for index in cell_indices:
        accumulator = (const(1.), [cells[index], cells[1 - index]], accumulator[2])
        step = namespace.get('step', namespace.get('new_make_obs1'))
        accumulator = step(
            particle, (const(False), const(1.)), accumulator)
    assert [particle.state.mean(cell) for cell in cells] == pytest.approx(expected_map)
    assert particle.score == pytest.approx(math.log(evidence))



@pytest.mark.parametrize('model,steps,probability', [('slam', 2, .1), ('slam-uncompiled', 2, .1), ('outlier', 9, .9)])
def test_compiled_mixture_exact_posterior_and_evidence(model, steps, probability):
    values, weights = [], []
    for bits in itertools.product((False, True), repeat=steps):
        if model == 'slam-uncompiled':
            import benchmark_sirenc as benchmark
            tree = ast.parse((benchmark.EXAMPLES / 'slam.py').read_text())
            tree.body = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef))]
            namespace = {}
            exec(compile(tree, '<uncompiled-slam>', 'exec'), namespace)
        else:
            namespace = load_program(model)
        draws = iter(bits)
        namespace['sample'] = lambda particle, distribution: const(next(draws))
        particle = Particle(None, SSIState(seed=1234))
        result = namespace['program'](particle)
        with pytest.raises(StopIteration):
            next(draws)
        values.append(flatten_mean(result, particle.state))
        weights.append(math.prod(probability if bit else 1-probability for bit in bits) * math.exp(particle.score))
    truth = reference('slam' if model == 'slam-uncompiled' else model)
    assert np.average(values, axis=0, weights=weights) == pytest.approx(truth['mean'], abs=1e-12)
    assert math.log(sum(weights)) == pytest.approx(truth['log_z'], abs=1e-12)


def test_sample_realizes_boolean_values():
    from siren.libsiren import bernoulli, sample
    particle = Particle(None, SSIState(seed=1234))
    assert sample(particle, bernoulli(const(0))).v is False
    assert sample(particle, bernoulli(const(1))).v is True


@pytest.mark.parametrize('model', ['slam', 'outlier', 'kalman'])
def test_generated_model_contains_only_libsiren_import(model):
    import benchmark_sirenc as benchmark
    path = benchmark.EXAMPLES / f'{model}-compiled.py'
    tree = ast.parse(path.read_text())
    imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
    assert len(imports) == 1
    assert isinstance(imports[0], ast.ImportFrom) and imports[0].module == 'siren.libsiren'
    assert not any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                   and node.func.id == 'isinstance' for node in ast.walk(tree))
    assert not any(isinstance(node, ast.FunctionDef) and node.name.startswith('_sirenc_')
                   for node in ast.walk(tree))
    assert not any(isinstance(node, ast.Name) and node.id.startswith('__siren_cse_')
                   for node in ast.walk(tree))


@pytest.mark.parametrize('model,folder', [('slam', 'slam-example'), ('outlier', 'outlier'), ('kalman', 'kalman')])
def test_compiler_reproduces_generated_model(model, folder):
    repo = Path(os.environ.get('SIRENC_REPO', '/Users/skundaliya/Documents/SirenC/SirenC'))
    if not (repo / 'llcsiren/src/compiler.py').is_file():
        pytest.skip('SirenC checkout unavailable')
    script = """
from pathlib import Path
import contextlib, io, sys
from siren.parser import parse_program
from siren.evaluate import evaluate
from llcsiren.src.compiler import compile_program, compile_source
source = Path('tests/test_llcsiren') / sys.argv[1] / 'source.si'
with contextlib.redirect_stdout(io.StringIO()):
    residual = evaluate(parse_program(source.read_text()), str(source.parent))
    assert isinstance(residual, str), 'symbolic evaluation must produce its normal residual trace'
    code = compile_source(residual)
print(code, end='')
"""
    result = subprocess.run([sys.executable, '-B', '-c', script, folder], cwd=repo,
                            capture_output=True, text=True, check=True, timeout=60)
    import benchmark_sirenc as benchmark
    assert result.stdout == (benchmark.EXAMPLES / f'{model}-compiled.py').read_text()


def test_sample_with_symbolic_parent_preserves_conditioning():
    from siren.libsiren import assume, beta, bernoulli, sample
    particle = Particle(None, SSIState(seed=1234))
    probability = assume(particle, beta(const(1), const(1)))
    value = sample(particle, bernoulli(probability))
    expected = 2/3 if value.v else 1/3
    assert particle.state.mean(probability) == pytest.approx(expected)


@pytest.mark.parametrize('model,folder', [
    ('slam', 'slam-example'), ('outlier', 'outlier'),
    ('coin', 'beta-example'), ('hybrid', 'gaussian-gaussian-hybrid'),
    ('kalman', 'kalman'), ('kalman-nonterminal', 'kalman'),
    pytest.param('kalman-affine', 'kalman', marks=pytest.mark.xfail(
        strict=True, reason='Existing affine-fold mismatch also reproduced without the new simplification')),
])
def test_normal_compilation_matches_siren_with_changed_inputs(model, folder):
    import benchmark_siren
    repo = Path(os.environ.get('SIRENC_REPO', '/Users/skundaliya/Documents/SirenC/SirenC'))
    source_path = repo / 'tests/test_llcsiren' / folder / 'source.si'
    if not source_path.is_file() or not benchmark_siren.ROOT.is_dir():
        pytest.skip('Compiler or original Siren checkout unavailable')
    source = source_path.read_text()
    if model == 'slam':
        source = (source.replace('bernoulli(0.5)', 'bernoulli(0.3)')
                  .replace('wheel_noise = 0.1', 'wheel_noise = 0.2')
                  .replace('sensor_noise = 0.1', 'sensor_noise = 0.25')
                  .replace('[(0, 1),(0, -1)]', '[(1, 1),(0, -1),(1, 1)]')
                  .replace('step', 'changed_step'))
    elif model == 'outlier':
        source = (source.replace('gaussian(x_pre, 1)', 'gaussian(x_pre, 2)')
                  .replace('gaussian(x,1)', 'gaussian(x,3)')
                  .replace('gaussian(0,100)', 'gaussian(0,50)')
                  .replace('bernoulli(0.9)', 'bernoulli(0.8)')
                  .replace('List.range(1, 10)', 'List.range(2, 6)')
                  .replace('make_obs', 'changed_step'))
    elif model == 'hybrid':
        source = source.replace('let input = 0', 'let input = 2')
    elif model == 'kalman':
        source = (source.replace('gaussian(x_pre, 1)', 'gaussian(x_pre, 2)')
                  .replace('gaussian(x,1)', 'gaussian(x,3)')
                  .replace('gaussian(0,100)', 'gaussian(5,50)')
                  .replace('List.range(1, 10)', 'List.range(-2, 6)')
                  .replace('make_obs', 'renamed_filter'))
    elif model == 'kalman-nonterminal':
        source = source.rstrip()[:-2] + '(x_init, xs)'
    elif model == 'kalman-affine':
        source = (source.replace('gaussian(x_pre, 1)', 'gaussian(2*x_pre+3, 2)')
                  .replace('gaussian(x,1)', 'gaussian(2*x+1,3)')
                  .replace('List.range(1, 10)', 'List.range(2, 6)'))
    compile_script = """
import contextlib, io, sys
from siren.parser import parse_program
from siren.evaluate import evaluate
from llcsiren.src.compiler import compile_source
with contextlib.redirect_stdout(io.StringIO()):
    residual = evaluate(parse_program(sys.argv[1]), '.')
    assert isinstance(residual, str)
    code = compile_source(residual)
print(code, end='')
"""
    code = subprocess.check_output([sys.executable, '-B', '-c', compile_script, source],
                                   cwd=repo, text=True, timeout=60)
    tree = ast.parse(code)
    tree.body = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef))]
    namespace = {}
    exec(compile(tree, '<compiled-variant>', 'exec'), namespace)
    values, scores = [], []
    for seed in np.random.SeedSequence(1234).spawn(100):
        particle = Particle(None, SSIState(seed=seed))
        values.append(flatten_mean(namespace['program'](particle), particle.state))
        scores.append(particle.score)
    weights = np.exp(np.asarray(scores) - max(scores))
    estimate = np.average(values, axis=0, weights=weights)
    reference_script = """
import json, sys
sys.path.insert(0, sys.argv[1])
import numpy as np
from siren.parser import parse_program
from siren.evaluate import SMC
from siren.inference.ssi import SSIState
seeds = iter(np.random.SeedSequence(1234).spawn(100))
class SeededState(SSIState):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rng = np.random.default_rng(next(seeds))
def flatten(value):
    if isinstance(value, (tuple, list)):
        return [x for part in value for x in flatten(part)]
    return [float(value)]
value, particles = SMC().infer(parse_program(sys.argv[2]), SeededState, '.', seed=1234, n_particles=100)
print(json.dumps(dict(mean=flatten(value.v), scores=[p.score for p in particles])))
"""
    expected = json.loads(subprocess.check_output(
        [sys.executable, '-B', '-c', reference_script, str(benchmark_siren.ROOT), source],
        text=True, timeout=60))
    assert estimate == pytest.approx(expected['mean'], abs=1e-12)
    assert scores == pytest.approx(expected['scores'], abs=1e-12)


def test_compiled_kalman_exact_moments_and_evidence():
    namespace = load_program('kalman')
    particle = Particle(None, SSIState(seed=1234))
    result = namespace['program'](particle)
    truth = reference('kalman')
    assert particle.state.mean(result) == pytest.approx(truth['mean'][0], abs=1e-12)
    posterior = particle.state.eval_distr(particle.state.distr(result))
    assert posterior.var.v == pytest.approx(truth['variance'][0], abs=1e-12)
    assert particle.score == pytest.approx(truth['log_z'], abs=1e-12)

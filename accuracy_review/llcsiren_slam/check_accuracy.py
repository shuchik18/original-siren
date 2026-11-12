"""Isolated diagnostic: no runtime, generated-example or interface file edits.

Run from the libsiren repository root:
  ../.venv11/bin/python accuracy_review/llcsiren_slam/check_accuracy.py

The optional AST adapter models correct Python condition lowering and sampled
Boolean semantics only. It does not rewrite SLAM's equations or supply answers.
"""
import ast
import itertools
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmark_reference import reference
from benchmark_sirenc import flatten_mean
from siren.inference.ssi import SSIState
from siren.libsiren import Particle, const

HERE = Path(__file__).resolve().parent


class ValueConditions(ast.NodeTransformer):
    def condition(self, node):
        return ast.Call(ast.Name('_condition_value', ast.Load()), [node], [])

    def visit_IfExp(self, node):
        self.generic_visit(node)
        node.test = self.condition(node.test)
        return node

    def visit_If(self, node):
        self.generic_visit(node)
        node.test = self.condition(node.test)
        return node


def load(path, adapt_conditions):
    tree = ast.parse(path.read_text())
    tree.body = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef))]
    if adapt_conditions:
        tree = ValueConditions().visit(tree)
    ast.fix_missing_locations(tree)
    namespace = {}
    exec(compile(tree, str(path), 'exec'), namespace)
    return namespace


def enumerate_motion(path, adapt_conditions):
    weights, values, paths = [], [], []
    for bits in itertools.product((False, True), repeat=2):
        namespace = load(path, adapt_conditions)
        draws = iter(bits)
        namespace['sample'] = lambda particle, distribution: const(next(draws))
        particle = Particle(None, SSIState(seed=1234))
        namespace['_condition_value'] = lambda expr: bool(particle.state.value_expr(expr).v)
        result = namespace['program'](particle)
        with_no_extra_draws = next(draws, None) is None
        if not with_no_extra_draws:
            raise AssertionError('Program did not consume both motion decisions')
        estimates = flatten_mean(result, particle.state)
        branch_prior = math.prod(.1 if bit else .9 for bit in bits)
        weight = branch_prior * math.exp(particle.score)
        weights.append(weight)
        values.append(estimates)
        paths.append(dict(slips=list(bits), conditional_mean=estimates, log_likelihood=particle.score))
    total = math.fsum(weights)
    mean = [math.fsum(w * value[i] for w, value in zip(weights, values)) / total
            for i in range(4)]
    truth = reference('slam')
    return dict(mean=mean, log_z=math.log(total), max_mean_error=max(abs(x-y) for x,y in zip(mean,truth['mean'])),
                log_z_error=abs(math.log(total)-truth['log_z']), paths=paths)


if __name__ == '__main__':
    results = {'exact': reference('slam')}
    for name, path, adapt in (
        ('transformed_as_emitted', HERE / 'generated.py', False),
        ('transformed_with_value_conditions', HERE / 'generated.py', True),
        ('original_source_with_value_conditions', HERE / 'original_generated.py', True),
    ):
        try:
            results[name] = enumerate_motion(path, adapt)
        except Exception as exc:
            results[name] = {'error': f'{type(exc).__name__}: {exc}'}
    (HERE / 'accuracy_diagnostic.json').write_text(json.dumps(results, indent=2))
    for name, value in results.items():
        if name != 'exact':
            print(name, {k:v for k,v in value.items() if k != 'paths'})
    original = results['original_source_with_value_conditions']
    assert original['max_mean_error'] < 1e-12
    assert original['log_z_error'] < 1e-12

"""The retained Bernoulli backend must not alter other code generation."""
from llcsiren.src.compiler import compile_source
from siren.compiler.bernoulli_folds import needs_dynamic_bernoulli_fold
from siren.parser import parse_program


def test_non_fold_bernoulli_program_uses_existing_path():
    source = "let sample b <- bernoulli(0.5) in if b then 1 else 0"
    assert not needs_dynamic_bernoulli_fold(parse_program(source))
    assert "_sirenc_sample_value" not in compile_source(source)


def test_mixed_gaussian_fold_uses_existing_path():
    source = "val step = fun (y, x) -> let sample b <- bernoulli(0.9) in let symbolic z <- gaussian(x, 1) in z in fold(step, [1, 2], 0)"
    assert not needs_dynamic_bernoulli_fold(parse_program(source))
    assert "_sirenc_sample_value" not in compile_source(source)

# import ast
# import subprocess
# import sys
# import tempfile
# import unittest
# from pathlib import Path

# from llcsiren.siren.compiler import CompilerError, compile_file, compile_program, compile_source
# from siren.parser import parse_program

# import siren.libsiren as libsiren
# from siren.libsiren import (
#     Particle,
#     bernoulli,
#     const,
#     delta,
#     div,
#     finalize,
#     fold,
#     gaussian,
#     get_distr,
#     get_mu,
#     get_p,
#     get_par,
#     get_var,
#     list_range,
#     new_rv,
#     observe,
#     observe_inner,
#     run,
#     sample,
#     set_distr,
#     times,
# )

# ROOT = Path(__file__).resolve().parents[1]
# FIXTURES = [
#   ROOT / "tests" / "test_llcsiren" / "example9",
#   ROOT / "tests" / "test_llcsiren" / "example9-compiled",
#   ROOT / "tests" / "test_llcsiren" / "example9-desired",
#   ROOT / "tests" / "test_llcsiren" / "example11",
#   ROOT / "tests" / "test_llcsiren" / "example11-compiled",
# ]


# class CompilerGoldenTests(unittest.TestCase):
#   def test_existing_fixtures_match_golden_targets(self):
#     for fixture in FIXTURES:
#       with self.subTest(fixture=fixture.name):
#         source = (fixture / "source.si").read_text()
#         expected = (fixture / "target.py").read_text()
#         self.assertEqual(compile_source(source), expected)

#   def test_generated_fixtures_parse_as_python(self):
#     for fixture in FIXTURES:
#       with self.subTest(fixture=fixture.name):
#         ast.parse(compile_source((fixture / "source.si").read_text()))


# class CompilerApiTests(unittest.TestCase):
#   def test_ast_passes_run_in_order(self):
#     calls = []
#     program = parse_program("1")

#     def first(program):
#       calls.append("first")
#       return program

#     def second(program):
#       calls.append("second")
#       return program

#     compile_program(program, passes=[first, second])

#     self.assertEqual(calls, ["first", "second"])

#   def test_ast_pass_must_return_program(self):
#     program = parse_program("1")

#     with self.assertRaisesRegex(CompilerError, "AST passes must return a Program"):
#       compile_program(program, passes=[lambda _program: None])

#   def test_compile_file_returns_and_writes_output(self):
#     source = ROOT / "tests" / "test_llcsiren" / "example9" / "source.si"
#     expected = (ROOT / "tests" / "test_llcsiren" / "example9" / "target.py").read_text()

#     with tempfile.TemporaryDirectory() as temp_dir:
#       output = Path(temp_dir) / "target.py"
#       compiled = compile_file(source, output)

#       self.assertEqual(compiled, expected)
#       self.assertEqual(output.read_text(), expected)

#   def test_unsupported_prob_call_is_clear(self):
#     with self.assertRaisesRegex(CompilerError, "Unsupported Prob call Prob.unknown"):
#       compile_source("Prob.unknown()")

#   def test_basic_constructs_compile_to_valid_python(self):
#     source = """
# val inc = fun x -> x + 1 in
# let pair = (1, 2) in
# let xs = [1, 2, 3] in
# let ys = 1 :: xs in
# if true then inc(1) else fold(inc, 0, ys)
# """
#     ast.parse(compile_source(source))

#   def test_resample_statement_compiles(self):
#     compiled = compile_source("let () = resample() in 1")

#     self.assertIn("resample(particle)", compiled)
#     ast.parse(compiled)

#   def test_unit_let_if_compiles_to_statement(self):
#     compiled = compile_source(
#       "let b = true in "
#       "let () = if b then observe(gaussian(0, 1), 0) "
#       "else observe(gaussian(1, 1), 1) in "
#       "b"
#     )

#     self.assertIn("if b:", compiled)
#     self.assertIn("else:", compiled)
#     self.assertIn("observe(particle, gaussian(const(0.0), const(1.0)), const(0.0))", compiled)
#     self.assertIn("observe(particle, gaussian(const(1.0), const(1.0)), const(1.0))", compiled)
#     self.assertNotIn(" if b else ", compiled)
#     ast.parse(compiled)

#   def test_value_if_compiles_to_expression(self):
#     compiled = compile_source("let b = true in let x = if b then 1 else 2 in x")

#     self.assertIn("(const(1.0) if b else const(2.0))", compiled)
#     self.assertNotIn("if b:\n", compiled)
#     ast.parse(compiled)

#   def test_main_program_marks_particle_finished_at_return(self):
#     direct = compile_source("1")
#     finalized = compile_source("Prob.finalize(1)")
#     helper = compile_source("val f = fun x -> x in f(1)")

#     self.assertIn("particle.finished = True", direct)
#     self.assertIn("particle.finished = True", finalized)
#     self.assertEqual(helper.count("particle.finished = True"), 1)

#   def test_ssa_renames_shadowed_let_bindings(self):
#     compiled = compile_source("let x = 1 in let x = x + 1 in x")

#     self.assertIn("x = const(1.0)", compiled)
#     self.assertIn("x_2 = add(x, const(1.0))", compiled)
#     self.assertIn("return x_2", compiled)

#   def test_ssa_renames_shadowed_function_locals(self):
#     compiled = compile_source("val f = fun x -> let x = x + 1 in x in f(1)")

#     self.assertIn("def f(x):\n  x_2 = add(x, const(1.0))\n  return x_2", compiled)
#     self.assertEqual(compiled.count("particle.finished = True"), 1)

#   def test_ssa_avoids_existing_name_collisions(self):
#     compiled = compile_source("let x_2 = 9 in let x = 1 in let x = x + x_2 in x")

#     self.assertIn("x_2 = const(9.0)", compiled)
#     self.assertIn("x = const(1.0)", compiled)
#     self.assertIn("x_3 = add(x, x_2)", compiled)
#     self.assertIn("return x_3", compiled)


# class CompilerCliTests(unittest.TestCase):
#   def test_cli_stdout(self):
#     source = ROOT / "tests" / "test_llcsiren" / "example9" / "source.si"
#     expected = (ROOT / "tests" / "test_llcsiren" / "example9" / "target.py").read_text()

#     result = subprocess.run(
#       [sys.executable, "-m", "siren.compiler", str(source)],
#       cwd=ROOT,
#       check=True,
#       text=True,
#       capture_output=True,
#     )

#     self.assertEqual(result.stdout, expected)
#     self.assertEqual(result.stderr, "")

#   def test_cli_output_file_and_particles(self):
#     source = ROOT / "tests" / "test_llcsiren" / "example9" / "source.si"

#     with tempfile.TemporaryDirectory() as temp_dir:
#       output = Path(temp_dir) / "compiled.py"
#       subprocess.run(
#         [
#           sys.executable,
#           "-m",
#           "siren.compiler",
#           str(source),
#           "-o",
#           str(output),
#           "--particles",
#           "17",
#         ],
#         cwd=ROOT,
#         check=True,
#         text=True,
#         capture_output=True,
#       )

#       compiled = output.read_text()
#       self.assertIn("run(program, 17)", compiled)
#       self.assertNotIn("print(run(program, 17))", compiled)


# if __name__ == "__main__":
#   unittest.main()

import ast
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

# Ensure project root is present in sys.path for module resolution
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llcsiren.siren.compiler import CompilerError, compile_file, compile_program, compile_source
from siren.parser import parse_program

# Import libsiren runtime components directly
import siren.libsiren as libsiren
from siren.libsiren import (
    Particle,
    assume,
    bernoulli,
    beta,
    const,
    delta,
    div,
    add,
    finalize,
    fold,
    gaussian,
    get_distr,
    get_mu,
    get_p,
    get_par,
    get_var,
    list_range,
    new_rv,
    observe,
    observe_inner,
    run,
    sample,
    set_distr,
    times,
)


FIXTURES = [
  ROOT / "tests" / "test_llcsiren" / "example9",
  ROOT / "tests" / "test_llcsiren" / "example9-compiled",
  ROOT / "tests" / "test_llcsiren" / "example9-desired",
  ROOT / "tests" / "test_llcsiren" / "example11",
  ROOT / "tests" / "test_llcsiren" / "example11-compiled",
]


class CompilerGoldenTests(unittest.TestCase):
  def test_existing_fixtures_match_golden_targets(self):
    for fixture in FIXTURES:
      with self.subTest(fixture=fixture.name):
        source = (fixture / "source.si").read_text()
        expected = (fixture / "target.py").read_text()
        self.assertEqual(compile_source(source), expected)

  def test_generated_fixtures_parse_as_python(self):
    for fixture in FIXTURES:
      with self.subTest(fixture=fixture.name):
        ast.parse(compile_source((fixture / "source.si").read_text()))


class CompilerApiTests(unittest.TestCase):
  def test_ast_passes_run_in_order(self):
    calls = []
    program = parse_program("1")

    def first(program):
      calls.append("first")
      return program

    def second(program):
      calls.append("second")
      return program

    compile_program(program, passes=[first, second])

    self.assertEqual(calls, ["first", "second"])

  def test_ast_pass_must_return_program(self):
    program = parse_program("1")

    with self.assertRaisesRegex(CompilerError, "AST passes must return a Program"):
      compile_program(program, passes=[lambda _program: None])

  def test_compile_file_returns_and_writes_output(self):
    source = ROOT / "tests" / "test_llcsiren" / "example9" / "source.si"
    expected = (ROOT / "tests" / "test_llcsiren" / "example9" / "target.py").read_text()

    with tempfile.TemporaryDirectory() as temp_dir:
      output = Path(temp_dir) / "target.py"
      compiled = compile_file(source, output)

      self.assertEqual(compiled, expected)
      self.assertEqual(output.read_text(), expected)

  def test_unsupported_prob_call_is_clear(self):
    with self.assertRaisesRegex(CompilerError, "Unsupported Prob call Prob.unknown"):
      compile_source("Prob.unknown()")

  def test_basic_constructs_compile_to_valid_python(self):
    source = """
val inc = fun x -> x + 1 in
let pair = (1, 2) in
let xs = [1, 2, 3] in
let ys = 1 :: xs in
if true then inc(1) else fold(inc, 0, ys)
"""
    ast.parse(compile_source(source))

  def test_resample_statement_compiles(self):
    compiled = compile_source("let () = resample() in 1")

    self.assertIn("resample(particle)", compiled)
    ast.parse(compiled)

  def test_unit_let_if_compiles_to_statement(self):
    compiled = compile_source(
      "let b = true in "
      "let () = if b then observe(gaussian(0, 1), 0) "
      "else observe(gaussian(1, 1), 1) in "
      "b"
    )

    self.assertIn("if b:", compiled)
    self.assertIn("else:", compiled)
    self.assertIn("observe(particle, gaussian(const(0.0), const(1.0)), const(0.0))", compiled)
    self.assertIn("observe(particle, gaussian(const(1.0), const(1.0)), const(1.0))", compiled)
    self.assertNotIn(" if b else ", compiled)
    ast.parse(compiled)

  def test_value_if_compiles_to_expression(self):
    compiled = compile_source("let b = true in let x = if b then 1 else 2 in x")

    self.assertIn("(const(1.0) if b else const(2.0))", compiled)
    self.assertNotIn("if b:\n", compiled)
    ast.parse(compiled)

  def test_main_program_marks_particle_finished_at_return(self):
    direct = compile_source("1")
    finalized = compile_source("Prob.finalize(1)")
    helper = compile_source("val f = fun x -> x in f(1)")

    self.assertIn("particle.finished = True", direct)
    self.assertIn("particle.finished = True", finalized)
    self.assertEqual(helper.count("particle.finished = True"), 1)

  def test_ssa_renames_shadowed_let_bindings(self):
    compiled = compile_source("let x = 1 in let x = x + 1 in x")

    self.assertIn("x = const(1.0)", compiled)
    self.assertIn("x_2 = add(x, const(1.0))", compiled)
    self.assertIn("return x_2", compiled)

  def test_ssa_renames_shadowed_function_locals(self):
    compiled = compile_source("val f = fun x -> let x = x + 1 in x in f(1)")

    self.assertIn("def f(x):\n  x_2 = add(x, const(1.0))\n  return x_2", compiled)
    self.assertEqual(compiled.count("particle.finished = True"), 1)

  def test_ssa_avoids_existing_name_collisions(self):
    compiled = compile_source("let x_2 = 9 in let x = 1 in let x = x + x_2 in x")

    self.assertIn("x_2 = const(9.0)", compiled)
    self.assertIn("x = const(1.0)", compiled)
    self.assertIn("x_3 = add(x, x_2)", compiled)
    self.assertIn("return x_3", compiled)


class LibsirenDirectExecutionTests(unittest.TestCase):
  def test_direct_libsiren_program_execution(self):
    def program(particle):
      rv1 = new_rv(particle)
      set_distr(particle, rv1, bernoulli(const(0.5)))
      set_distr(particle, rv1, bernoulli(get_p(get_distr(particle, rv1))))
      set_distr(particle, rv1, bernoulli(get_p(get_distr(particle, rv1))))
      S3 = assume(particle, get_distr(particle, rv1))
      set_distr(particle, rv1, delta(S3))
      b = S3
      particle.finished = True
      return b

    particles = run(program, 1000)
    self.assertEqual(len(particles), 1000)
    self.assertTrue(all(p.finished for p in particles))


class CompilerCliTests(unittest.TestCase):
  def _get_env(self):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    return env

  def test_cli_stdout(self):
    source = ROOT / "tests" / "test_llcsiren" / "example9" / "source.si"
    expected = (ROOT / "tests" / "test_llcsiren" / "example9" / "target.py").read_text()

    result = subprocess.run(
      [sys.executable, "-m", "siren.compiler", str(source)],
      cwd=ROOT,
      env=self._get_env(),
      check=True,
      text=True,
      capture_output=True,
    )

    self.assertEqual(result.stdout, expected)
    self.assertEqual(result.stderr, "")

  def test_cli_output_file_and_particles(self):
    source = ROOT / "tests" / "test_llcsiren" / "example9" / "source.si"

    with tempfile.TemporaryDirectory() as temp_dir:
      output = Path(temp_dir) / "compiled.py"
      subprocess.run(
        [
          sys.executable,
          "-m",
          "siren.compiler",
          str(source),
          "-o",
          str(output),
          "--particles",
          "17",
        ],
        cwd=ROOT,
        env=self._get_env(),
        check=True,
        text=True,
        capture_output=True,
      )

      compiled = output.read_text()
      self.assertIn("run(program, 17)", compiled)
      self.assertNotIn("print(run(program, 17))", compiled)


if __name__ == "__main__":
  unittest.main()
# from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from siren.grammar import (
  Annotation,
  Apply,
  Const,
  Expr,
  Fold,
  Function,
  GenericOp,
  Identifier,
  IfElse,
  Ite, 
  Let,
  LetRV,
  Observe,
  Operator,
  Program,
  Resample,
)
from siren.parser import parse_program
from llcsiren.siren.ssa import to_ssa


AstPass = Callable[[Program], Program]


class CompilerError(Exception):
  pass


def _make_gaussian_state():
  """Exact joint Gaussian conditioning for affine, unsampled programs."""
  import math
  import numpy as np
  from siren.grammar import Const, RandomVar, Normal, Add, Mul, Div, Eq, Lt

  class GaussianState:
    def __init__(self):
      self.mu = np.empty(0, dtype=np.longdouble)
      self.cov = np.empty((0, 0), dtype=np.longdouble)
      self.indices = {}

    def affine(self, expr):
      zero = np.zeros(len(self.mu), dtype=np.longdouble)
      if isinstance(expr, Const):
        if type(expr.v) not in (int, float, bool):
          raise TypeError("Gaussian backend requires numeric constants")
        return np.longdouble(expr.v), zero
      if isinstance(expr, RandomVar):
        zero[self.indices[expr]] = 1
        return np.longdouble(0), zero
      if isinstance(expr, (Add, Mul, Div)):
        b1, a1 = self.affine(expr.left)
        b2, a2 = self.affine(expr.right)
        if isinstance(expr, Add):
          return b1 + b2, a1 + a2
        if isinstance(expr, Mul):
          if np.any(a1) and np.any(a2):
            raise TypeError("Gaussian backend does not support nonlinear products")
          return b1 * b2, b1 * a2 + b2 * a1
        if np.any(a2) or b2 == 0:
          raise TypeError("Gaussian backend requires a nonzero constant divisor")
        return b1 / b2, a1 / b2
      raise TypeError(f"Unsupported Gaussian expression: {type(expr).__name__}")

    def value_expr(self, expr):
      if isinstance(expr, Const):
        return expr
      if isinstance(expr, (Eq, Lt)):
        left = self.value_expr(expr.left).v
        right = self.value_expr(expr.right).v
        return Const(left == right if isinstance(expr, Eq) else left < right)
      offset, coefficients = self.affine(expr)
      if np.any(coefficients):
        raise TypeError("Gaussian backend requires deterministic control flow")
      return Const(float(offset))

    def assume(self, name, annotation, distribution):
      if not isinstance(distribution, Normal):
        raise TypeError("Gaussian backend supports only Normal distributions")
      if annotation is not None and getattr(annotation, 'name', '') == 'sample':
        raise TypeError("Gaussian backend does not support sample annotations")
      offset, coefficients = self.affine(distribution.mu)
      variance = np.longdouble(self.value_expr(distribution.var).v)
      if not np.isfinite(variance) or variance < 0:
        raise ValueError("Invalid Gaussian variance")
      cross = self.cov @ coefficients
      marginal_variance = variance + coefficients @ cross
      marginal_mean = offset + coefficients @ self.mu
      n = len(self.mu)
      covariance = np.empty((n + 1, n + 1), dtype=np.longdouble)
      covariance[:n, :n] = self.cov
      covariance[:n, n] = cross
      covariance[n, :n] = cross
      covariance[n, n] = marginal_variance
      self.cov = covariance
      self.mu = np.append(self.mu, marginal_mean)
      rv = RandomVar(f"compiled_gaussian_{n}")
      self.indices[rv] = n
      return rv

    def observe(self, rv, value):
      i = self.indices[rv]
      observed = np.longdouble(self.value_expr(value).v)
      variance = self.cov[i, i]
      if not np.isfinite(variance) or variance <= 0:
        raise ValueError("Observation requires positive Gaussian variance")
      residual = observed - self.mu[i]
      column = self.cov[:, i].copy()
      self.mu += column * (residual / variance)
      self.cov -= np.outer(column, column) / variance
      self.cov = (self.cov + self.cov.T) / 2
      self.mu[i] = observed
      self.cov[i, :] = 0
      self.cov[:, i] = 0
      return float(-0.5 * (np.log(2 * math.pi * variance) + residual**2 / variance))

    def mean(self, expr):
      offset, coefficients = self.affine(expr)
      return float(offset + coefficients @ self.mu)

    def distr(self, rv):
      i = self.indices[rv]
      return Normal(Const(float(self.mu[i])), Const(float(self.cov[i, i])))

  return GaussianState()


@dataclass(frozen=True)
class CompilerOptions:
  run_particles: int = 1000
  linear_gaussian: bool = False


def compile_source(
  source: str,
  *,
  run_particles: int = 1000,
  passes: Sequence[AstPass] = (),
  linear_gaussian: bool = False,
) -> str:
  return compile_program(
    parse_program(source),
    run_particles=run_particles,
    passes=passes,
    linear_gaussian=linear_gaussian,
  )


def compile_program(
  program: Program,
  *,
  run_particles: int = 1000,
  passes: Sequence[AstPass] = (),
  linear_gaussian: bool = False,
) -> str:
  for ast_pass in passes:
    program = ast_pass(program)
    if not isinstance(program, Program):
      raise CompilerError("AST passes must return a Program")
  program = to_ssa(program)

  if linear_gaussian:
    from dataclasses import fields, is_dataclass

    def validate(node):
      if isinstance(node, Resample) or (
        isinstance(node, LetRV) and node.annotation is Annotation.sample
      ):
        raise CompilerError("Linear Gaussian mode does not support sampling or resampling")
      if isinstance(node, (list, tuple)):
        for item in node:
          validate(item)
      elif is_dataclass(node) and not isinstance(node, type):
        for field in fields(node):
          validate(getattr(node, field.name))

    validate(program)

  compiler = PythonCompiler(CompilerOptions(run_particles=run_particles, linear_gaussian=linear_gaussian))
  return compiler.compile(program)


def compile_file(
  input_path: str | Path,
  output_path: str | Path | None = None,
  *,
  run_particles: int = 1000,
  linear_gaussian: bool = False,
) -> str:
  source = Path(input_path).read_text()
  compiled = compile_source(source, run_particles=run_particles, linear_gaussian=linear_gaussian)
  if output_path is not None:
    Path(output_path).write_text(compiled)
  return compiled


class PythonCompiler:
  _DISTRIBUTIONS = {
    Operator.gaussian: "gaussian",
    Operator.beta: "beta",
    Operator.bernoulli: "bernoulli",
    Operator.binomial: "binomial",
    Operator.beta_binomial: "beta_binomial",
    Operator.negative_binomial: "negative_binomial",
    Operator.exponential: "exponential",
    Operator.gamma: "gamma",
    Operator.poisson: "poisson",
    Operator.delta: "delta",
    Operator.categorical: "categorical",
    Operator.uniform_int: "uniform_int",
    Operator.student_t: "student_t",
  }

  _OPERATORS = {
    Operator.add: "add",
    Operator.sub: "sub",
    Operator.mul: "times",
    Operator.div: "div",
    Operator.eq: "eq",
    Operator.lt: "lt",
    Operator.cons: "cons",
    Operator.lst: "lst",
    Operator.pair: "pair",
  }

  def __init__(self, options: CompilerOptions) -> None:
    self.options = options
    self._reserved_names: set[str] = set()
    self._temp_counter = 0
    self._function_names: set[Identifier] = set()

  def compile(self, program: Program) -> str:
    self._reset_temporaries(program)
    self._function_names = {function.name for function in program.functions}
    lines = ["from siren.libsiren import *", ""]

    if self.options.linear_gaussian:
      import inspect
      lines.extend(inspect.getsource(_make_gaussian_state).splitlines())
      lines.append("")

    for function in program.functions:
      lines.extend(self._emit_function(function))
      lines.append("")

    lines.append("def program(particle):")
    body = self._finish_program(self._emit_tail(program.main))
    if self.options.linear_gaussian:
      body.insert(0, "particle.state = _make_gaussian_state()")
    lines.extend(f"  {line}" for line in body)
    lines.append("")
    lines.append(f"run(program, {self.options.run_particles})")
    return self._inline_update_temporaries("\n".join(lines) + "\n")

  def _inline_update_temporaries(self, source: str) -> str:
    """Inline generated distribution values without crossing state mutations."""
    tree = ast.parse(source)
    lines = source.splitlines()
    pure_calls = set(self._DISTRIBUTIONS.values()) | set(self._OPERATORS.values()) | {
      "const", "sub", "subst_rv", "ifelse", "get_par", "get_mu", "get_var",
      "get_distr", "get_p",
    }

    def pure(node: ast.AST) -> bool:
      return all(
        isinstance(part.func, ast.Name) and part.func.id in pure_calls
        for part in ast.walk(node) if isinstance(part, ast.Call)
      )

    edits = {}
    removed = set()
    for scope in tree.body:
      if not isinstance(scope, ast.FunctionDef):
        continue
      for block in ast.walk(scope):
        statements = getattr(block, "body", None)
        if not isinstance(statements, list):
          continue
        for index, statement in enumerate(statements):
          if not (isinstance(statement, ast.Assign) and len(statement.targets) == 1
                  and isinstance(statement.targets[0], ast.Name)):
            continue
          name = statement.targets[0].id
          if not name.startswith(("__siren_update_posterior_", "__siren_update_marginal_")):
            continue
          uses = [n for n in ast.walk(scope)
                  if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id == name]
          if len(uses) != 1 or not pure(statement.value):
            continue
          # Constructors and arithmetic on existing values do not read the
          # particle's distributions, so they may cross a set_distr update.
          state_independent = all(
            part.func.id not in {"get_par", "get_mu", "get_var", "get_distr", "get_p"}
            for part in ast.walk(statement.value) if isinstance(part, ast.Call)
          )
          dependencies = {part.id for part in ast.walk(statement.value)
                          if isinstance(part, ast.Name) and isinstance(part.ctx, ast.Load)}
          for following in statements[index + 1:]:
            if isinstance(following, ast.Assign) and pure(following.value):
              if any(isinstance(part, ast.Name) and part.id in dependencies
                     for target in following.targets for part in ast.walk(target)):
                break
              continue
            if isinstance(following, ast.Expr) and isinstance(following.value, ast.Call):
              call = following.value
              if (isinstance(call.func, ast.Name) and call.func.id == "set_distr"
                  and len(call.args) == 3 and call.args[2] is uses[0]
                  and statement.lineno == statement.end_lineno
                  and following.lineno == following.end_lineno):
                line = edits.get(following.lineno, lines[following.lineno - 1])
                value = ast.get_source_segment(source, statement.value)
                edits[following.lineno] = line[:uses[0].col_offset] + value + line[uses[0].end_col_offset:]
                removed.add(statement.lineno)
              elif (state_independent and isinstance(call.func, ast.Name)
                    and call.func.id == "set_distr" and len(call.args) == 3
                    and all(pure(arg) for arg in call.args)):
                continue
            break
    return "\n".join(edits.get(i, line) for i, line in enumerate(lines, 1) if i not in removed) + "\n"

  def _finish_program(self, body: list[str]) -> list[str]:
    if not body:
      return ["particle.finished = True", "return const(None)"]
    if body[-1].startswith("return "):
      return body[:-1] + ["particle.finished = True", body[-1]]
    return body + ["particle.finished = True"]

  def _emit_function(self, function: Function) -> list[str]:
    name = self._identifier_name(function.name)
    parameters = []
    bindings = []
    for arg in function.args:
      target = self._pattern_target(arg)
      if target.startswith("("):
        parameter = self._fresh_temp()
        parameters.append(parameter)
        bindings.append(f"{target} = {parameter}")
      else:
        parameters.append(target)
    args = ", ".join(parameters)
    if args:
      args = f"particle, {args}"
    else:
      args = "particle"
    lines = [f"def {name}({args}):"]
    lines.extend(f"  {binding}" for binding in bindings)
    lines.extend(f"  {line}" for line in self._emit_tail(function.body))
    return lines

  def _emit_tail(self, expr: Expr) -> list[str]:
    match expr:
      case LetRV(var=var, annotation=annotation, distribution=distribution, body=body):
        target = self._identifier_name(var)
        value = self._emit_rv_binding(annotation, distribution)
        return [f"{target} = {value}"] + self._emit_tail(body)

      case Let(var=var, value=value, body=body):
        lines: list[str] = []
        if self._is_unit_pattern(var):
          lines.extend(self._emit_effect(value))
        else:
          value_lines, value_expr = self._emit_value(value)
          lines.extend(value_lines)
          lines.append(f"{self._pattern_target(var)} = {value_expr}")
        lines.extend(self._emit_tail(body))
        return lines

      case Observe() | Apply(func=Identifier(module=None, name="observe")):
        return self._emit_effect(expr) + ["return const(None)"]

      case Apply(func=Identifier(module="Prob", name="finalize"), args=[value]):
        return [f"return {self._emit_expr(value)}"]

      case Apply(func=Identifier(module="Prob", name="null")):
        return []

      case Apply(func=Identifier(module="Prob", name="finalize")):
        raise CompilerError("Prob.finalize expects one argument")

      case _:
        lines, value = self._emit_value(expr)
        lines.append(f"return {value}")
        return lines

  def _emit_condition(self, rendered):
    if self.options.linear_gaussian:
      return f"particle.state.value_expr({rendered}).v"
    return rendered

  def _emit_value(self, expr: Expr) -> tuple[list[str], str]:
    match expr:
      case IfElse(cond=cond, then=then, else_=else_):
        cond_lines, cond_value = self._emit_value(cond)
        then_lines, then_value = self._emit_value(then)
        else_lines, else_value = self._emit_value(else_)
        if not cond_lines and not then_lines and not else_lines:
          return [], f"({then_value} if {self._emit_condition(cond_value)} else {else_value})"

        result = self._fresh_temp()
        return (
          cond_lines
          + [f"if {self._emit_condition(cond_value)}:"]
          + self._indent(then_lines + [f"{result} = {then_value}"])
          + ["else:"]
          + self._indent(else_lines + [f"{result} = {else_value}"])
        ), result

      case Let(var=var, value=value, body=body):
        lines: list[str] = []
        if self._is_unit_pattern(var):
          lines.extend(self._emit_effect(value))
        else:
          value_lines, value_expr = self._emit_value(value)
          lines.extend(value_lines)
          lines.append(f"{self._pattern_target(var)} = {value_expr}")
        body_lines, body_value = self._emit_value(body)
        lines.extend(body_lines)
        return lines, body_value

      case LetRV(var=var, annotation=annotation, distribution=distribution, body=body):
        target = self._identifier_name(var)
        lines = [f"{target} = {self._emit_rv_binding(annotation, distribution)}"]
        body_lines, body_value = self._emit_value(body)
        lines.extend(body_lines)
        return lines, body_value

      case Observe() | Apply(func=Identifier(module=None, name="observe")):
        return self._emit_effect(expr), "const(None)"

      case Apply(func=Identifier(module="Prob", name="finalize"), args=[value]):
        return [], self._emit_expr(value)

      case Apply(func=Identifier(module="Prob", name="finalize")):
        raise CompilerError("Prob.finalize expects one argument")

      case _:
        return [], self._emit_expr(expr)

  def _emit_effect(self, expr: Expr) -> list[str]:
    match expr:
      case IfElse(cond=cond, then=then, else_=else_):
        return self._emit_if_statement(cond, then, else_)

      case Observe(condition=condition, observation=observation):
        return [
          "observe("
          f"particle, {self._emit_distribution(condition)}, "
          f"{self._emit_expr(observation)})"
        ]

      case Apply(func=Identifier(module=None, name="observe"), args=[condition, observation]):
        return [
          "observe("
          f"particle, {self._emit_distribution(condition)}, "
          f"{self._emit_expr(observation)})"
        ]

      case Apply(func=Identifier(module=None, name="observe")):
        raise CompilerError("observe expects two arguments")

      case Fold(func=func, init=init, acc=acc):
        return [
          "fold("
          "particle, "
          f"{self._identifier_name(func)}, "
          f"{self._emit_expr(init)}, "
          f"{self._emit_expr(acc)})"
        ]

      case Apply(func=Identifier(module=None, name="fold"), args=[func, lst, acc]):
        return [
          "fold("
          "particle, "
          f"{self._identifier_name(func)}, "
          f"{self._emit_expr(lst)}, "
          f"{self._emit_expr(acc)})"
        ]

      case Apply(func=Identifier(module=None, name="fold")):
        raise CompilerError("fold expects exactly three arguments")

      case Apply(func=Identifier(module=None, name="resample"), args=[]):
        return ["resample(particle)"]

      case Apply(func=Identifier(module=None, name="resample")):
        raise CompilerError("resample expects 0 arguments")

      case Apply(func=Identifier(module="Prob")):
        return self._emit_prob_effect(expr)

      case Resample():
        return ["resample(particle)"]

      case Let() | LetRV():
        return self._emit_discard(expr)

      case _:
        return [self._emit_expr(expr)]

  def _emit_if_statement(self, cond: Expr, then: Expr, else_: Expr) -> list[str]:
    cond_lines, cond_value = self._emit_value(cond)
    then_lines = self._emit_if_branch(then)
    else_lines = self._emit_if_branch(else_)
    return (
      cond_lines
      + [f"if {self._emit_condition(cond_value)}:"]
      + self._indent(then_lines)
      + ["else:"]
      + self._indent(else_lines)
    )

  def _emit_if_branch(self, expr: Expr) -> list[str]:
    return self._emit_effect(expr)

  def _emit_discard(self, expr: Expr) -> list[str]:
    match expr:
      case LetRV(var=var, annotation=annotation, distribution=distribution, body=body):
        target = self._identifier_name(var)
        value = self._emit_rv_binding(annotation, distribution)
        return [f"{target} = {value}"] + self._emit_discard(body)

      case Let(var=var, value=value, body=body):
        lines: list[str] = []
        if self._is_unit_pattern(var):
          lines.extend(self._emit_effect(value))
        else:
          value_lines, value_expr = self._emit_value(value)
          lines.extend(value_lines)
          lines.append(f"{self._pattern_target(var)} = {value_expr}")
        lines.extend(self._emit_discard(body))
        return lines

      case _:
        return self._emit_effect(expr)

  def _indent(self, lines: list[str]) -> list[str]:
    return [f"  {line}" for line in lines] if lines else ["  pass"]

  def _emit_prob_effect(self, expr: Apply) -> list[str]:
    match expr:
      case Apply(func=Identifier(module="Prob", name="set_distr"), args=[target_expr, distribution_expr]):
        target = self._emit_expr(target_expr)
        distribution = self._emit_distribution(distribution_expr)
        return [f"set_distr(particle, {target}, {distribution})"]

      case Apply(func=Identifier(module="Prob", name="set_distr")):
        raise CompilerError("Prob.set_distr expects two arguments")

      case Apply(func=Identifier(module="Prob", name="observe_inner"), args=[rv, value]):
        return [
          "observe_inner("
          f"particle, {self._emit_expr(rv)}, {self._emit_expr(value)})"
        ]

      case Apply(func=Identifier(module="Prob", name="observe_inner")):
        raise CompilerError("Prob.observe_inner expects two arguments")

      case Apply(func=Identifier(module="Prob", name="set_lookup"), args=[_, _]):
        return []

      case Apply(func=Identifier(module="Prob", name="set_lookup")):
        raise CompilerError("Prob.set_lookup expects two arguments")

      case Apply(func=Identifier(module="Prob", name="null"), args=[]):
        return []

      case Apply(func=Identifier(module="Prob", name="null")):
        raise CompilerError("Prob.null expects 0 arguments")

      case Apply(func=Identifier(module="Prob", name="finalize"), args=[_]):
        return []

      case Apply(func=Identifier(module="Prob", name="finalize")):
        raise CompilerError("Prob.finalize expects one argument")

      case _:
        return [self._emit_expr(expr)]

  def _emit_expr(self, expr: Expr) -> str:
    match expr:
      case Const():
        return self._emit_const(expr)

      case Identifier():
        return self._identifier_name(expr)

      case GenericOp(op=op) if op in self._DISTRIBUTIONS:
        return self._emit_distribution(expr)

      case GenericOp(op=op, args=args) if op in self._OPERATORS:
        name = self._OPERATORS[op]
        rendered_args = ", ".join(self._emit_expr(arg) for arg in args)
        return f"{name}({rendered_args})"

      case GenericOp(op=op):
        raise CompilerError(f"Unsupported operator {op}")

      case Apply():
        return self._emit_apply(expr)

      case Ite(cond=cond, true=true, false=false):
        return (
            f"ifelse({self._emit_expr(cond)}, "
            f"{self._emit_expr(true)}, "
            f"{self._emit_expr(false)})"
        )

      case IfElse(cond=cond, then=then, else_=else_):
        return (
          f"({self._emit_expr(then)} if {self._emit_condition(self._emit_expr(cond))} "
          f"else {self._emit_expr(else_)})"
        )

      case Fold(func=func, init=init, acc=acc):
        return (
          "fold(particle, "
          f"{self._identifier_name(func)}, "
          f"{self._emit_expr(init)}, "
          f"{self._emit_expr(acc)})"
        )

      case Resample():
        return "resample(particle)"

      case Let() | LetRV() | Observe():
        lines, value = self._emit_value(expr)
        if lines:
          raise CompilerError("Nested effectful expressions are only supported in let bindings")
        return value

      case _:
        raise CompilerError(f"Unsupported expression {expr!r}")


  def _emit_symbolic_expr(self, expr: Expr) -> str:
    def lower(node):
        match node:
            case IfElse(cond=cond, then=then, else_=else_):
                return Apply(
                    Identifier("Prob", "ifelse"),
                    [lower(cond), lower(then), lower(else_)],
                )

            case GenericOp(op=op, args=args):
                return GenericOp(op, [lower(arg) for arg in args])

            case Apply(func=func, args=args):
                return Apply(func, [lower(arg) for arg in args])

            case _:
                return node

    return self._emit_expr(lower(expr))

  def _emit_apply(self, expr: Apply) -> str:
    match expr:
      case Apply(func=Identifier(module=None, name="resample"), args=[]):
        return "resample(particle)"

      case Apply(func=Identifier(module=None, name="resample")):
        raise CompilerError("resample expects 0 arguments")

      case Apply(func=Identifier(module=None, name="fold"), args=[func, lst, acc]):
        return (
          "fold("
          f"particle, {self._identifier_name(func)}, "
          f"{self._emit_expr(lst)}, "
          f"{self._emit_expr(acc)})"
        )

      case Apply(func=Identifier(module=None, name="fold")):
        raise CompilerError("fold expects exactly three arguments")

      case Apply(func=Identifier(module="List", name="range"), args=[start, stop]):
        return f"list_range({self._emit_range_bound(start)}, {self._emit_range_bound(stop)})"

      case Apply(func=Identifier(module="List", name="range")):
        raise CompilerError("List.range expects exactly two arguments")
      
      case Apply(func=Identifier(module="List", name="hd"), args=[lst]):
        return f"list_hd({self._emit_expr(lst)})"
 
      case Apply(func=Identifier(module="List", name="hd")):
        raise CompilerError("List.hd expects exactly one argument")

      case Apply(func=Identifier(module=None, name="lt"), args=[left, right]):
        return f"lt({self._emit_expr(left)}, {self._emit_expr(right)})"

      case Apply(func=Identifier(module=None, name="lt")):
        raise CompilerError("lt expects exactly two arguments")
 
      case Apply(func=Identifier(module="List", name="tl"), args=[lst]):
        return f"list_tl({self._emit_expr(lst)})"
 
      case Apply(func=Identifier(module="List", name="tl")):
        raise CompilerError("List.tl expects exactly one argument")
 
      case Apply(func=Identifier(module="List", name="rev"), args=[lst]):
        return f"list_rev({self._emit_expr(lst)})"
 
      case Apply(func=Identifier(module="List", name="rev")):
        raise CompilerError("List.rev expects exactly one argument")

      case Apply(func=Identifier(module="List", name="map"), args=[func, lst]):
        return (
          f"list_map(particle, {self._identifier_name(func)}, "
          f"{self._emit_expr(lst)})"
        )

      case Apply(func=Identifier(module="List", name="map")):
        raise CompilerError("List.map expects exactly two arguments")

      case Apply(func=Identifier(module="List", name="len"), args=[lst]):
        return f"list_len({self._emit_expr(lst)})"

      case Apply(func=Identifier(module="List", name="len")):
        raise CompilerError("List.len expects exactly one argument")
 

      case Apply(func=Identifier(module="Prob")):
        return self._emit_prob_expr(expr)

      case Apply(func=func, args=args):
        name = self._qualified_name(func)
        rendered = [self._emit_expr(arg) for arg in args]
        if func in self._function_names:
          rendered.insert(0, "particle")
        rendered_args = ", ".join(rendered)
        return f"{name}({rendered_args})"

  def _emit_prob_expr(self, expr: Apply) -> str:
    match expr:
      case Apply(func=Identifier(module="Prob", name="ex_mul"), args=[left, right]):
        return f"times({self._emit_expr(left)}, {self._emit_expr(right)})"

      case Apply(func=Identifier(module="Prob", name="ex_mul")):
        raise CompilerError("Prob.ex_mul expects two arguments")

      case Apply(func=Identifier(module="Prob", name="new_var"), args=[]):
        return "new_rv(particle)"

      case Apply(func=Identifier(module="Prob", name="new_var")):
        raise CompilerError("Prob.new_var expects 0 arguments")

      case Apply(func=Identifier(module="Prob", name="get_distr"), args=[rv]):
        return f"get_distr(particle, {self._emit_expr(rv)})"

      case Apply(func=Identifier(module="Prob", name="get_distr")):
        raise CompilerError("Prob.get_distr expects one argument")

      case Apply(func=Identifier(module="Prob", name=name), args=[distribution]) if name in {"get_a", "get_b"}:
        return f"{name}({self._emit_expr(distribution)})"

      case Apply(func=Identifier(module="Prob", name=name)) if name in {"get_a", "get_b"}:
        raise CompilerError(f"Prob.{name} expects one argument")

      case Apply(func=Identifier(module="Prob", name="get_mu"), args=[distribution]):
        return f"get_mu({self._emit_expr(distribution)})"

      case Apply(func=Identifier(module="Prob", name="get_mu")):
        raise CompilerError("Prob.get_mu expects one argument")

      case Apply(func=Identifier(module="Prob", name="get_var"), args=[distribution]):
        return f"get_var({self._emit_expr(distribution)})"


      case Apply(func=Identifier(module="Prob", name="get_var")):
        raise CompilerError("Prob.get_var expects one argument")

      case Apply(func=Identifier(module="Prob", name="subst_rv"), args=[expr, rv, value]):
        return (f"subst_rv({self._emit_expr(expr)}, "f"{self._emit_expr(rv)}, "f"{self._emit_expr(value)})")

      case Apply(func=Identifier(module="Prob", name="subst_rv")):
        raise CompilerError("Prob.subst_rv expects three arguments")

      case Apply(func=Identifier(module="Prob", name="get_p"), args=[distribution]):
        return f"get_p({self._emit_expr(distribution)})"

      case Apply(func=Identifier(module="Prob", name="get_p")):
        raise CompilerError("Prob.get_p expects one argument")

      case Apply(func=Identifier(module="Prob", name="get_par"), args=[rv, index]):
        return (
          "get_par("
          f"particle, {self._emit_expr(rv)}, "
          f"{self._emit_index_arg(index)})"
        )

      case Apply(func=Identifier(module="Prob", name="get_par")):
        raise CompilerError("Prob.get_par expects two arguments")

      case Apply(func=Identifier(module="Prob", name="ifelse"), args=[condition, true_value, false_value]):
        return (f"ifelse({self._emit_expr(condition)}, "f"{self._emit_expr(true_value)}, "f"{self._emit_expr(false_value)})"
)

      case Apply(func=Identifier(module="Prob", name="finalize"), args=[value]):
        return self._emit_expr(value)

      case Apply(func=Identifier(module="Prob", name="finalize")):
        raise CompilerError("Prob.finalize expects one argument")

      case Apply(func=Identifier(module="Prob", name=name)) if name in {
        "set_distr",
        "observe_inner",
        "set_lookup",
        "null",
      }:
        raise CompilerError(f"Prob.{name} can only be emitted as a statement")

      case Apply(func=Identifier(module="Prob", name=name)):
        raise CompilerError(f"Unsupported Prob call Prob.{name}")

  def _emit_distribution(self, expr: Expr) -> str:
    if (
        not isinstance(expr, GenericOp)
        or expr.op not in self._DISTRIBUTIONS
    ):
        return self._emit_symbolic_expr(expr)

    name = self._DISTRIBUTIONS[expr.op]
    args = ", ".join(
        self._emit_symbolic_expr(arg)
        for arg in expr.args
    )
    return f"{name}({args})"

  def _emit_rv_binding(self, annotation: Annotation | None, distribution: Expr) -> str:
    return (
      f"{self._rv_binding_function(annotation)}"
      f"(particle, {self._emit_distribution(distribution)})"
    )

  def _rv_binding_function(self, annotation: Annotation | None) -> str:
    match annotation:
      case Annotation.sample:
        return "sample"
      case Annotation.symbolic | None:
        return "assume"
      case _:
        raise CompilerError(f"Unsupported random variable annotation {annotation}")

  def _emit_const(self, expr: Const) -> str:
    value = expr.v
    return f"const({self._literal(value)})"

  def _emit_index_arg(self, expr: Expr) -> str:
    if isinstance(expr, Const) and isinstance(expr.v, float) and expr.v.is_integer():
      return f"const({int(expr.v)})"
    return self._emit_expr(expr)

  def _emit_range_bound(self, expr: Expr) -> str:
    if isinstance(expr, Const) and isinstance(expr.v, (int, float)) and not isinstance(expr.v, bool):
      if isinstance(expr.v, int) or expr.v.is_integer():
        return str(int(expr.v))
    return self._emit_expr(expr)

  def _reset_temporaries(self, program: Program) -> None:
    self._reserved_names = set()
    self._temp_counter = 0
    for function in program.functions:
      self._collect_identifier(function.name)
      self._collect_pattern(function.args)
      self._collect_expr(function.body)
    self._collect_expr(program.main)

  def _fresh_temp(self) -> str:
    while True:
      self._temp_counter += 1
      name = f"__siren_tmp_{self._temp_counter}"
      if name not in self._reserved_names:
        self._reserved_names.add(name)
        return name

  def _collect_expr(self, expr: Expr) -> None:
    match expr:
      case Identifier():
        self._collect_identifier(expr)
      case GenericOp(args=args) | Apply(args=args):
        for arg in args:
          self._collect_expr(arg)
      case Fold(func=func, init=init, acc=acc):
        self._collect_identifier(func)
        self._collect_expr(init)
        self._collect_expr(acc)
      case IfElse(cond=cond, then=then, else_=else_):
        self._collect_expr(cond)
        self._collect_expr(then)
        self._collect_expr(else_)
      case Let(var=var, value=value, body=body):
        self._collect_pattern(var)
        self._collect_expr(value)
        self._collect_expr(body)
      case LetRV(var=var, distribution=distribution, body=body):
        self._collect_identifier(var)
        self._collect_expr(distribution)
        self._collect_expr(body)
      case Observe(condition=condition, observation=observation):
        self._collect_expr(condition)
        self._collect_expr(observation)

  def _collect_pattern(self, pattern) -> None:
    match pattern:
      case Identifier():
        self._collect_identifier(pattern)
      case list():
        for part in pattern:
          self._collect_pattern(part)

  def _collect_identifier(self, identifier: Identifier) -> None:
    if identifier.module is None and identifier.name is not None:
      self._reserved_names.add(identifier.name)

  def _literal(self, value) -> str:
    if isinstance(value, str):
      return repr(value).replace("'", '"')
    if value is None:
      return "None"
    if value is True:
      return "True"
    if value is False:
      return "False"
    if isinstance(value, float):
      return repr(value)
    if isinstance(value, list):
      return "[" + ", ".join(self._literal(v) for v in value) + "]"
    if isinstance(value, tuple):
      return "(" + ", ".join(self._literal(v) for v in value) + ")"
    return repr(value)

  def _pattern_target(self, pattern) -> str:
    if isinstance(pattern, Identifier):
      return self._identifier_name(pattern)
    if isinstance(pattern, list):
      if len(pattern) == 0:
        return "_"
      if len(pattern) == 1:
        return self._pattern_target(pattern[0])
      return "(" + ", ".join(self._pattern_target(part) for part in pattern) + ")"
    raise CompilerError(f"Unsupported pattern {pattern!r}")

  def _is_unit_pattern(self, pattern) -> bool:
    return isinstance(pattern, list) and len(pattern) == 0

  def _identifier_name(self, identifier: Identifier) -> str:
    if identifier.module is not None:
      raise CompilerError(f"Expected local identifier, got {identifier}")
    if identifier.name is None:
      return "_"
    return identifier.name

  def _qualified_name(self, identifier: Identifier) -> str:
    if identifier.module is None:
      if identifier.name is None:
        raise CompilerError("Cannot call wildcard identifier")
      return identifier.name
    return f"{identifier.module}.{identifier.name}"

  def _is_plain_call(self, expr: Apply, name: str) -> bool:
    return expr.func.module is None and expr.func.name == name

  def _is_prob_call(self, expr: Apply, name: str) -> bool:
    return expr.func.module == "Prob" and expr.func.name == name

  def _expect_arity(self, expr: Apply, expected: int) -> None:
    if len(expr.args) != expected:
      raise CompilerError(f"{self._qualified_name(expr.func)} expects {expected} arguments")


def main(argv: Sequence[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description="Compile Siren source to Python.")
  parser.add_argument("input", help="Path to the .si source file")
  parser.add_argument("-o", "--output", help="Path to write generated Python")
  parser.add_argument(
    "--particles",
    type=int,
    default=1000,
    help="Number of particles used in the generated run call",
  )
  parser.add_argument("--linear-gaussian", action="store_true",
                      help="Exact affine Gaussian conditioning")
  args = parser.parse_args(argv)
  if args.linear_gaussian and (
    args.output is None or not Path(args.output).is_file()
  ):
    parser.error("--linear-gaussian requires an existing --output file")

  compiled = compile_file(
    args.input,
    args.output,
    run_particles=args.particles,
    linear_gaussian=args.linear_gaussian,
  )
  if args.output is None:
    print(compiled, end="")
  return 0

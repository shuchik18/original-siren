from __future__ import annotations

from dataclasses import replace

from siren.grammar import (
  Apply,
  Expr,
  Fold,
  Function,
  GenericOp,
  Identifier,
  IfElse,
  Let,
  LetRV,
  Observe,
  Program,
  Resample,
)


Env = dict[str, Identifier]


class SSATransformer:
  def __init__(self, program: Program) -> None:
    self.function_names = {
      function.name.name
      for function in program.functions
      if function.name.module is None and function.name.name is not None
    }
    self.reserved_names = self._collect_names(program)

  def transform(self, program: Program) -> Program:
    return Program(
      [self._function(function) for function in program.functions],
      self._expr(program.main, {}, set(self.function_names), {}),
    )

  def _function(self, function: Function) -> Function:
    env: Env = {}
    taken = set(self.function_names)
    counters: dict[str, int] = {}
    args, env = self._bind_pattern(function.args, env, taken, counters)
    return Function(
      function.name,
      args,
      self._expr(function.body, env, taken, counters),
    )

  def _expr(
    self,
    expr: Expr,
    env: Env,
    taken: set[str],
    counters: dict[str, int],
  ) -> Expr:
    match expr:
      case Identifier(module=None, name=name) if name in env:
        return env[name]

      case Identifier() | Resample():
        return expr

      case GenericOp(args=args):
        return replace(
          expr,
          args=[self._expr(arg, env, taken, counters) for arg in args],
        )

      case Apply(func=func, args=args):
        return Apply(
          func,
          [self._expr(arg, env, taken, counters) for arg in args],
        )

      case Fold(func=func, init=init, acc=acc):
        return Fold(
          func,
          self._expr(init, env, taken, counters),
          self._expr(acc, env, taken, counters),
        )

      case IfElse(cond=cond, then=then, else_=else_):
        return IfElse(
          self._expr(cond, env, taken, counters),
          self._expr(then, env.copy(), taken.copy(), counters.copy()),
          self._expr(else_, env.copy(), taken.copy(), counters.copy()),
        )

      case Let(var=var, value=value, body=body):
        renamed_value = self._expr(value, env, taken, counters)
        body_env = env.copy()
        renamed_var, body_env = self._bind_pattern(var, body_env, taken, counters)
        return Let(
          renamed_var,
          renamed_value,
          self._expr(body, body_env, taken, counters),
        )

      case LetRV(var=var, annotation=annotation, distribution=distribution, body=body):
        renamed_distribution = self._expr(distribution, env, taken, counters)
        body_env = env.copy()
        renamed_var, body_env = self._bind_identifier(var, body_env, taken, counters)
        return LetRV(
          renamed_var,
          annotation,
          renamed_distribution,
          self._expr(body, body_env, taken, counters),
        )

      case Observe(condition=condition, observation=observation):
        return Observe(
          self._expr(condition, env, taken, counters),
          self._expr(observation, env, taken, counters),
        )

      case _:
        return expr

  def _bind_pattern(
    self,
    pattern,
    env: Env,
    taken: set[str],
    counters: dict[str, int],
  ):
    match pattern:
      case Identifier():
        return self._bind_identifier(pattern, env, taken, counters)

      case list():
        renamed = []
        for part in pattern:
          renamed_part, env = self._bind_pattern(part, env, taken, counters)
          renamed.append(renamed_part)
        return renamed, env

      case _:
        return pattern, env

  def _bind_identifier(
    self,
    identifier: Identifier,
    env: Env,
    taken: set[str],
    counters: dict[str, int],
  ) -> tuple[Identifier, Env]:
    if identifier.module is not None or identifier.name is None:
      return identifier, env

    original_name = identifier.name
    if original_name not in env and original_name not in taken:
      renamed = identifier
    else:
      renamed = Identifier(None, self._fresh_name(original_name, taken, counters))

    taken.add(renamed.name)
    env[original_name] = renamed
    return renamed, env

  def _fresh_name(
    self,
    name: str,
    taken: set[str],
    counters: dict[str, int],
  ) -> str:
    counter = counters.get(name, 1) + 1
    while True:
      candidate = f"{name}_{counter}"
      counters[name] = counter
      if candidate not in taken and candidate not in self.reserved_names:
        return candidate
      counter += 1

  def _collect_names(self, program: Program) -> set[str]:
    names: set[str] = set()
    for function in program.functions:
      self._collect_identifier(function.name, names)
      self._collect_pattern(function.args, names)
      self._collect_expr(function.body, names)
    self._collect_expr(program.main, names)
    return names

  def _collect_expr(self, expr: Expr, names: set[str]) -> None:
    match expr:
      case Identifier():
        self._collect_identifier(expr, names)
      case GenericOp(args=args) | Apply(args=args):
        for arg in args:
          self._collect_expr(arg, names)
      case Fold(func=func, init=init, acc=acc):
        self._collect_identifier(func, names)
        self._collect_expr(init, names)
        self._collect_expr(acc, names)
      case IfElse(cond=cond, then=then, else_=else_):
        self._collect_expr(cond, names)
        self._collect_expr(then, names)
        self._collect_expr(else_, names)
      case Let(var=var, value=value, body=body):
        self._collect_pattern(var, names)
        self._collect_expr(value, names)
        self._collect_expr(body, names)
      case LetRV(var=var, distribution=distribution, body=body):
        self._collect_identifier(var, names)
        self._collect_expr(distribution, names)
        self._collect_expr(body, names)
      case Observe(condition=condition, observation=observation):
        self._collect_expr(condition, names)
        self._collect_expr(observation, names)

  def _collect_pattern(self, pattern, names: set[str]) -> None:
    match pattern:
      case Identifier():
        self._collect_identifier(pattern, names)
      case list():
        for part in pattern:
          self._collect_pattern(part, names)

  def _collect_identifier(self, identifier: Identifier, names: set[str]) -> None:
    if identifier.module is None and identifier.name is not None:
      names.add(identifier.name)


def to_ssa(program: Program) -> Program:
  return SSATransformer(program).transform(program)


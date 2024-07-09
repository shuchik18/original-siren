from typing import Any, Optional, List, Dict, Tuple
import numpy as np
import os
from multiprocessing import Pool, cpu_count, Queue, Process
import time
from copy import copy

from siren.grammar import *
from siren.utils import get_pair, get_lst
from siren.inference.interface import SymState, Context, ProbState, Particle

def assume(name: Identifier, annotation: Optional[Annotation], distribution: SymExpr,
           state: SymState) -> Const | RandomVar:
  assert isinstance(distribution, SymDistr)
  rv = state.assume(name, annotation, distribution)
  # If the annotation is sample, sample the value
  if annotation is Annotation.sample:
    return state.value(rv)
  return rv

def observe(score: float, distribution: SymExpr, v: SymExpr, state: SymState) -> float:
  assert isinstance(distribution, SymDistr)
  rv = state.assume(None, None, distribution)
  # the conditioned value must be a constant
  v = state.value_expr(v)
  s = state.observe(rv, v)
  return score + s

# true if expr does not contain resample and observe
def pure(expr: Expr[SymExpr], functions: Dict[Identifier, Function]) -> bool:
  def _pure(expr: Expr[SymExpr]) -> bool:
    # All symbolic expressions are pure
    if isinstance(expr, SymExpr):
      return True

    match expr:
      case Resample():
        return False
      case Observe(_, _):
        return False
      case Identifier(_, _):
        return True
      case GenericOp(op, args):
        return all(_pure(arg) for arg in args)
      # TODO: save purity of functions in functions to avoid recomputation
      case Fold(func, init, acc):
        return _pure(functions[func].body) and _pure(init) and _pure(acc)
      case Apply(func, args):
        return all(_pure(arg) for arg in args) and _pure(functions[func].body)
      case Lst(exprs):
        return all(_pure(expr) for expr in exprs)
      case Pair(fst, snd):
        return _pure(fst) and _pure(snd)
      case IfElse(cond, true, false):
        return _pure(cond) and _pure(true) and _pure(false)
      case Let(_, value, body):
        return _pure(value) and _pure(body)
      case LetRV(_, _, distribution, expression):
        return _pure(distribution) and _pure(expression)
      case _:
        raise ValueError(expr)

  return _pure(expr)

# Match pattern to expression
def match_pattern(pattern: List[Any], expr: SymExpr) -> Context:
    if len(pattern) == 0:
      return Context()
    elif len(pattern) == 1:
      return Context({pattern[0]: expr})
    else:
      try:
        fst, snd = get_pair(expr)
      except ValueError:
        raise ValueError(pattern, expr)
      if len(pattern) == 2:
        return match_pattern(pattern[0], fst) | match_pattern(pattern[1], snd)
      else:
        return match_pattern(pattern[0], fst) | match_pattern(pattern[1:], snd)

# Evaluates a single particle
# file_dir is the directory of the file being evaluated, used for file operations for relative paths
def evaluate_particle(particle: Particle, functions: Dict[Identifier, Function[SymExpr]], file_dir: str) -> Particle:
  # Evaluates arguments, from left to right. Returning the particle, the remaining arguments, and the evaluated arguments
  # If an argument is not finished, the particle is returned with the remaining arguments
  def _evaluate_args(particle: Particle,
                     args: List[Expr[SymExpr]],
                     new_args: List[SymExpr]) -> Tuple[Particle, List[Expr[SymExpr]], List[SymExpr]]:
    if len(args) == 0:
      return particle, args, new_args
    p1 = _evaluate(particle.update(cont=args[0]))
    if not p1.finished:
      args[0] = p1.cont
      return p1, args, new_args
    new_args.append(p1.final_expr)
    return _evaluate_args(p1, args[1:], new_args)

  # Evaluate built-in operators
  def _evaluate_ops(particle: Particle, op: Operator, args: SymExpr) -> Particle:
    def _evaluate_unops(particle: Particle, constructor: Any, args: SymExpr) -> Particle:
      return particle.update(cont=constructor(args), finished=True)

    def _evaluate_binops(particle: Particle, constructor: Any, args: SymExpr) -> Particle:
      fst, snd = get_pair(args)
      return particle.update(cont=constructor(fst, snd), finished=True)

    def _evaluate_triops(particle: Particle, constructor: Any, args: SymExpr) -> Particle:
      fst, args2 = get_pair(args)
      snd, trd = get_pair(args2)
      return particle.update(cont=constructor(fst, snd, trd), finished=True)

    # Map to the correct operator
    match op.name:
      case "add":
        return _evaluate_binops(particle, Add, args)
      case "sub":
        # a - b = a + (-1 * b)
        return _evaluate_binops(particle, lambda fst,snd: Add(fst, Mul(Const(-1), snd)), args)
      case "mul":
        return _evaluate_binops(particle, Mul, args)
      case "div":
        return _evaluate_binops(particle, Div, args)
      case "eq":
        return _evaluate_binops(particle, Eq, args)
      case "lt":
        return _evaluate_binops(particle, Lt, args)
      case "cons":
        return _evaluate_binops(particle, lambda fst,snd: Lst([fst] + get_lst(snd)), args)
      case "lst":
        # Empty list is considered a Constant
        def _make_list(x):
          if isinstance(x, Const):
            if x.v is None:
              return Const([])
          return Lst([x])
        return _evaluate_unops(particle, _make_list, args)
      case "pair":
        return _evaluate_binops(particle, Pair, args)
      case "gaussian":
        return _evaluate_binops(particle, Normal, args)
      case "beta":
        return _evaluate_binops(particle, Beta, args)
      case "bernoulli":
        return _evaluate_unops(particle, Bernoulli, args)
      case "binomial":
        return _evaluate_binops(particle, Binomial, args)
      case "beta_binomial":
        return _evaluate_triops(particle, BetaBinomial, args)
      case "negative_binomial":
        return _evaluate_binops(particle, NegativeBinomial, args)
      case "exponential":
        # Exponential is a special case of Gamma
        # Represented as a Gamma so can be detected when applying
        # conjugacy rules
        return _evaluate_binops(particle, Gamma, Pair(Const(1.0), args))
      case "gamma":
        return _evaluate_binops(particle, Gamma, args)
      case "poisson":
        return _evaluate_unops(particle, Poisson, args)
      case "delta":
        return _evaluate_unops(particle, Delta, args)
      case "categorical":
        return _evaluate_triops(particle, Categorical, args)
      case "uniform_int":
        a, b = get_pair(args)
        # For now, uniform only takes constants
        # [a, b]
        # Represented as a categorical distribution
        match (particle.state.eval(a), particle.state.eval(b)):
          case (Const(a), Const(b)):
            assert isinstance(a, Number) and isinstance(b, Number)\
              and round(a) == a and round(b) == b and a <= b
            a, b = int(a), int(b)
            probs = Const(list(np.ones(b - a + 1) / (b - a + 1)))
            return _evaluate_ops(particle, Operator.categorical,
                                  Pair(Const(a), Pair(Const(b), probs)))
          case _:
            raise ValueError(args)
      case "student_t":
        return _evaluate_triops(particle, StudentT, args)
      case _:
        raise ValueError(op.name)

  # Args can be a list of expressions, so need to extract them
  # if only a single argument (or none)
  def _convert_args(args: List[SymExpr]) -> SymExpr:
    if len(args) == 0:
      return Const(None)
    elif len(args) == 1:
      return args[0]
    else:
      return Pair(args[0], _convert_args(args[1:]))

  # Evaluate list operations. Evaluates arguments first, interrupting if args did not finish evaluation
  def _evaluate_list(particle: Particle, func: Identifier, args: List[Expr[SymExpr]]) -> Particle:
    assert func.module == 'List'
    match func.name:
      case 'hd':
        (p1, old_args, new_args) = _evaluate_args(particle, args, [])
        if len(old_args) != 0:
          return p1.update(cont=Apply(func, old_args + new_args), finished=False)
        new_args = _convert_args(new_args)

        exprs = get_lst(new_args)
        if len(exprs) == 0:
          raise ValueError(new_args)
        return p1.update(cont=exprs[0], finished=True)
      case 'tl':
        (p1, old_args, new_args) = _evaluate_args(particle, args, [])
        if len(old_args) != 0:
          return p1.update(cont=Apply(func, old_args + new_args), finished=False)
        new_args = _convert_args(new_args)

        exprs = get_lst(new_args)
        if len(exprs) == 0:
          raise ValueError(new_args)
        return p1.update(cont=Lst(exprs[1:]), finished=True)
      case 'len':
        (p1, old_args, new_args) = _evaluate_args(particle, args, [])
        if len(old_args) != 0:
          return p1.update(cont=Apply(func, old_args + new_args), finished=False)
        new_args = _convert_args(new_args)

        exprs = get_lst(new_args)
        return p1.update(cont=Const(len(exprs)), finished=True)
      case 'range':
        (p1, old_args, new_args) = _evaluate_args(particle, args, [])
        if len(old_args) != 0:
          return p1.update(cont=Apply(func, old_args + new_args), finished=False)
        new_args = _convert_args(new_args)

        # Range only takes constants
        a, b = get_pair(new_args)
        match p1.state.eval(a), p1.state.eval(b):
          case Const(a), Const(b):
            assert isinstance(a, Number) and isinstance(b, Number) and a <= b
            l : List[SymExpr] = list(map(Const, range(int(a), int(b))))
            return p1.update(cont=Lst(l), finished=True)
          case _:
            raise ValueError(new_args)
      case 'rev':
        (p1, old_args, new_args) = _evaluate_args(particle, args, [])
        if len(old_args) != 0:
          return p1.update(cont=Apply(func, old_args + new_args), finished=False)
        new_args = _convert_args(new_args)

        exprs = get_lst(new_args)
        return p1.update(cont=Lst(exprs[::-1]), finished=True)
      case 'map':
        map_func = args[0]
        assert isinstance(map_func, Identifier)

        (p1, old_args, new_args) = _evaluate_args(particle, args[1:], [])
        if len(old_args) != 0:
          return p1.update(cont=Apply(func, old_args + new_args), finished=False)
        new_args = _convert_args(new_args)

        # Map is syntactic sugar of a list, calling the function on each element
        exprs = get_lst(new_args)
        new_e = Lst([])
        for e in exprs[::-1]:
          new_e = GenericOp(Operator.cons, [
            Apply(map_func, [e]),
            new_e,
          ])
        return _evaluate(p1.update(cont=new_e))
      case _:
        raise ValueError(func)

  # Evaluate file operations. These only take constants as arguments
  def _evaluate_file(particle: Particle, func: Identifier, args: List[Expr[SymExpr]]) -> Particle:
    assert func.module == 'File'
    match func.name:
      case 'read':
        (p1, old_args, new_args) = _evaluate_args(particle, args, [])
        if len(old_args) != 0:
          return p1.update(cont=Apply(func, old_args + new_args), finished=False)
        new_args = _convert_args(new_args)

        match new_args:
          case Const(filename):
            if os.path.isabs(filename):
              path = filename
            else:
              path = os.path.join(file_dir, filename)
            data = []
            with open(path, 'r') as f:
              lines = f.readlines()
              for line in lines[1:]:
                line_list = line.strip().split(',')
                line_list = list(map(lambda x: float(x), line_list))
                data.append(line_list)
            return p1.update(cont=Const(data), finished=True)
          case _:
            raise ValueError(new_args)
      case _:
        raise ValueError(func)

  # Evaluate the particle, returning an evaluated particle
  # or a particle with the next expression to evaluate if interrupted by resample
  def _evaluate(particle: Particle) -> Particle:
    if isinstance(particle.cont, SymExpr):
      return particle.update(finished=True)
    match particle.cont:
      case Identifier(_, _):
        return particle.update(cont=particle.state.ctx[particle.cont], finished=True)
      case GenericOp(op, args):
        (p1, old_args, new_args) = _evaluate_args(particle, args, [])
        if len(old_args) != 0:
          return p1.update(cont=GenericOp(op, old_args + new_args), finished=False)
        new_args = _convert_args(new_args)
        return _evaluate_ops(p1, op, new_args)
      case Fold(func, lst, acc):
        # Fold is syntactic sugar for a chain of let expressions
        # It's a bounded loop
        p1 = _evaluate(particle.update(cont=lst))
        if not p1.finished:
          return p1.update(cont=Fold(func, p1.cont, acc), finished=False)
        lst_val = p1.final_expr

        p2 = _evaluate(p1.update(cont=acc))
        if not p2.finished:
          return p2.update(cont=Fold(func, lst_val, p2.cont), finished=False)
        acc_val = p2.final_expr
        match lst_val:
          case Lst(exprs):
            if len(exprs) == 0:
              return p2.update(cont=acc_val, finished=True)
            else:
              hd, tl = exprs[0], exprs[1:]
              # Create a temporary variable to store the result of the function
              tempvar = p2.state.ctx.temp_var()
              e = Let([tempvar],
                      Apply(func, [hd, acc_val]),
                      Fold(func, Lst(tl), tempvar))
              return _evaluate(p2.update(cont=e))
          case _:
            raise ValueError(p1.cont)
      case Apply(func, args):
        if func.module is not None:
          # use lib functions
          match func.module:
            case "List":
              return _evaluate_list(particle, func, args)
            case "File":
              return _evaluate_file(particle, func, args)
            case _:
              raise ValueError(func.module)
        else:
          (p1, old_args, new_args) = _evaluate_args(particle, args, [])
          if len(old_args) != 0:
            return p1.update(cont=Apply(func, old_args + new_args), finished=False)
          converted_args = _convert_args(new_args)
          function = functions[func]
          ctx = copy(p1.state.ctx)
          p1.state.ctx = match_pattern(function.args, converted_args)
          p2 = _evaluate(p1.update(cont=function.body))
          if p2.finished:
            p2.state.ctx = ctx
          return p2
      case IfElse(cond, true, false):
        p1 = _evaluate(particle.update(cont=cond))
        if not p1.finished:
          return p1.update(cont=IfElse(p1.cont, true, false), finished=False)
        cond_val = p1.final_expr

        # If both branches are pure, evaluate them and represent as ite symbolic expression
        if len(cond_val.rvs()) > 0 and pure(true, functions) and pure(false, functions):
          p2 = _evaluate(p1.update(cont=true))
          if not p2.finished:
            return p2.update(cont=IfElse(cond_val, p2.cont, false), finished=False)
          then_val = p2.final_expr
          p3 = _evaluate(p2.update(cont=false))
          if not p3.finished:
            return p3.update(cont=IfElse(p1.cont, p2.cont, p3.cont), finished=False)
          return p3.update(cont=p3.state.ex_ite(cond_val, then_val, p3.final_expr), finished=True)
        else:
          # If not pure, fully evaluate the condition, sampling RVs if necessary,
          # and then evaluate only the branch that is taken
          cond_value = p1.state.value_expr(p1.final_expr)
          match cond_value:
            case Const(v):
              if v:
                return _evaluate(p1.update(cont=true))
              else:
                return _evaluate(p1.update(cont=false))
            case _:
              raise ValueError(cond_value)
      case Let(pattern, v, body):
        ctx = copy(particle.state.ctx)
        p1 = _evaluate(particle.update(cont=v))
        if not p1.finished:
          return p1.update(cont=Let(pattern, p1.cont, body), finished=False)
        val = p1.final_expr
        p1.state.ctx |= match_pattern(pattern, val)
        p2 = _evaluate(p1.update(cont=body))
        # If the body is finished, restore the original context
        if p2.finished:
          p2.state.ctx = ctx
        return p2
      case LetRV(identifier, annotation, distribution, expression):
        p1 = _evaluate(particle.update(cont=distribution))
        assert isinstance(p1.cont, Op) # should still be an Op even if not finished
        if not p1.finished:
          return p1.update(cont=LetRV(identifier, annotation, p1.cont, expression), finished=False)
        assert identifier.name is not None # RVs should always be named
        rv = assume(identifier, annotation, p1.final_expr, p1.state)
        # After creating the RV, it is just a let expression
        return _evaluate(p1.update(cont=Let([identifier], rv, expression)))
      case Observe(expression, v):
        p1 = _evaluate(particle.update(cont=expression))
        assert isinstance(p1.cont, Op) # should still be an Op even if not finished
        if not p1.finished:
          return p1.update(cont=Observe(p1.cont, v), finished=False)
        d = p1.final_expr
        assert isinstance(d, Op) # should still be an Op
        p2 = _evaluate(p1.update(cont=v))
        if not p2.finished:
          return p2.update(cont=Observe(d, p2.cont), finished=False)
        w = observe(p2.score, d, p2.final_expr, p2.state)
        # Update the particle with the new score
        return p2.update(score=w)
      case Resample():
        # Resample interrupts the evalution
        return particle.update(cont=Const(None), finished=False)
      case _:
        raise ValueError(particle.cont)

  return _evaluate(particle)

def evaluate(
  program: Program,
  n_particles: int,
  method: type[SymState],
  file_dir: str,
  seed: Optional[int] = None,
) -> Tuple[SymExpr, ProbState]:
  # functions, expression = program.functions, program.main
  #
  # # Make lookup for functions
  # functions = {f.name: f for f in functions}
  #
  # # Initialize particles
  # particles = ProbState(n_particles, expression, method, seed)
  # # Evaluate particles until all are finished
  # while True:
  #   for i, particle in enumerate(particles):
  #     if particle.finished:
  #       continue
  #     else:
  #       particles[i] = evaluate_particle(particle, functions, file_dir)
  #
  #   # If not all particles are finished, resample the particles
  #   if particles.finished:
  #     break
  #   else:
  #     particles.resample()


  res =[]
  for i in range(n_particles):
    x = np.random.normal(0,1)
    res.append(x)
  return res

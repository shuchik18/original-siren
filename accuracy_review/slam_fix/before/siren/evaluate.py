import operator
from contextlib import nullcontext
from importlib.metadata import distribution
from logging import raiseExceptions
from pickle import FLOAT
from typing import Any, Optional, List, Dict, Tuple, Callable, Type
from weakref import finalize
from xxlimited_35 import error

import numpy as np
import os
from multiprocessing import Pool, cpu_count, Queue, Process
import time
from copy import copy
from collections.abc import Callable

from scipy.constants import value

from siren.analysis.conjugate import abs_is_affine
from siren.analysis.interface import AbsContext
from siren.analyze import analyze
from siren.grammar import *
from siren.grammar import Expr, SymExpr, Const, AbsConst, IfElse, TopE, Let, Lst, AbsLst, UnkE, Identifier, Function
from siren.utils import get_pair, get_lst
from siren.compiler.interface import SymState, Context, Particle
from siren.compiler.ssi import SSIState, NonConjugate
from siren.inference.interface import ProbState
from siren.analysis.ssi import AbsSSIState
from siren.analysis.interface import *
from siren.inference_plan import DistrEnc

def assume(name: Identifier, annotation: Optional[Annotation], distribution: SymExpr,
           state: SymState,Abs_distribution:AbsSymExpr) -> tuple[Callable[[Expr], Expr], Expr,Expr]:
  # assert isinstance(distribution, SymDistr)
  # (k2,res) = state.distr(distribution)

  (k, rv, rv1) = state.assume(name, annotation, distribution, Abs_distribution)
  # print("the value of the annotation in the assume func", annotation)

  print("the value of the rv1", rv1)
  print("value of the rv", rv)
  # print("name ..................", name)
  # print("the value of the distribution", distribution)
  # print("the value of the abs_distr in the assume ", Abs_distribution)
  # state.register_expr_for_abs_rv(distribution, rv)
  s = state.set_lookup(name, rv,rv1)


  # If the annotation is sample, sample the value
  if annotation is Annotation.sample:
    # raise ValueError("sampling is not supported")
    # if rv is Operator:
    # print("asdfasdfasdfas",rv)
    print("rv in assume",rv)
    # print("absrv in assume", abs_rv)
    # print("absssss",rv1)
    (k1, distr,abs_distr) = state.value(rv,rv1)
    print("distr in assume", distr)
    print("absdistr in assume", abs_distr)


    def kret(res1: Expr) -> Expr:

      return k(s(k1(res1)))





    # print("the value of the k is.................. ",kret)
    # print("value of the distr in assume in evaluate ",abs_distr)
    # print("final abs_distr is ",rv1)





    return kret,distr,abs_distr


  def kres(res: Expr) -> Expr:
      return k(s(res))
  # (k, rv1) = state.assume(name, annotation, distr)
  #   return state.value(rv)
  # def kret(res: Expr) -> Expr:
  #   # return k1(k(res))
  #   return Let([],rv,k(res))
  # print("k..............",  k)
  return kres, rv,rv1

def observe(score: Expr, distribution: SymExpr, abs_distribution: AbsSymExpr,val:SymExpr,abs_val:AbsSymExpr, state: SymState) -> \
Callable[[Expr], Expr]:
  # assert isinstance(distribution, SymDistr)
  print("distrbution in observe", distribution)
  print("abs distribution in observe", abs_distribution)
  print("val in observe..............................",val)
  print("abs_val in observe...........................", abs_val)
  
  # while isinstance(abs_distribution, AbsRandomVar):
  #     abs_distribution = state.Abs_distr(abs_distribution)

  # while isinstance(abs_val, AbsRandomVar):
  #     abs_val = state.Abs_distr(abs_val)
  
  (conc_val,rv,abs_rv) = state.assume(None, None, distribution,abs_distribution)
  print("----------------------------------In observe func-------------------------------------")
  print(conc_val)
  print(rv)
  print(abs_rv)
  print("----------------------------------end In observe func-------------------------------------")
  # the conditioned value must be a constant

  # assert isinstance(distribution, AbsSymDistr)
  # rv1 = AbsSymState.assume(None, None, distribution)
  # v1 = AbsSymState.value_expr(v)
  # AbsSymState.observe(rv1, v1)
  # print(state.value_expr(val,abs_val))
  (v,v_abs) = state.value_expr(val,abs_val)
  
  print("abs val in observe.........................",v_abs)
  print("v in observe ..............................",v)
  s = state.observe(rv, abs_rv,v, v_abs)
  
  print("s val in observe in evaluate",s)
  #abstract state
  # state1: AbsSymState
  # expr1: AbsSymExpr
  # assert isinstance(distribution, AbsSymDistr)
  # rv1 = state1.assume(None, None, distribution)
  # v1 = state1.value_expr(expr1)
  # state1.observe(rv1, v1)
  s_val  = score

  def val(res:Expr)->Expr:
    # print("===============================in observe======================================")
    # print(conc_val(s(res)))
    # print("=====================================================================")
    return conc_val(s(res))



  # return score + s
  return val

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
def match_pattern(pattern: List[Any],abs_expr:AbsSymExpr) -> AbsContext:
    if len(pattern) == 0:
      return AbsContext()
    elif len(pattern) == 1:
      return AbsContext({pattern[0]:(pattern[0],abs_expr)})
    elif isinstance(abs_expr, AbsLst):
      if len(pattern) != len(abs_expr.exprs):
        raise ValueError(pattern, abs_expr)
      context = AbsContext()
      for part, value in zip(pattern, abs_expr.exprs):
        context |= match_pattern(part, value)
      return context
    else:
      try:
        # print("in the match patter", abs_expr)
        fst, snd = get_abs_pair(abs_expr)
      except ValueError:
        raise ValueError(pattern, abs_expr)
      if len(pattern) == 2:
        # print("into match pattern this case")
        return match_pattern(pattern[0], fst) | match_pattern(pattern[1], snd)
      else:
        return match_pattern(pattern[0], fst) | match_pattern(pattern[1:], snd)

# Evaluates a single particle
# file_dir is the directory of the file being evaluated, used for file operations for relative paths
def evaluate_expr(expr: Expr, functions: Dict[Identifier, Function[SymExpr]], file_dir: str) -> Callable[[[Expr], Expr],Expr]:
  fold_function_counter = 0

  def new_fold_function() -> Identifier:
    nonlocal fold_function_counter
    while True:
      fold_function_counter += 1
      name = Identifier(None, f"new_make_obs{fold_function_counter}")
      if name not in functions:
        return name

  # Evaluates arguments, from left to right. Returning the particle, the remaining arguments, and the evaluated arguments
  # If an argument is not finished, the particle is returned with the remaining arguments
  def _evaluate_args(state: SymState,
                     args: List[Expr[SymExpr]],
                     new_args: List[Expr]) -> tuple[
    Callable[[Any], Any], SymState, list[Expr[SymExpr]], list[Expr], dict[Identifier, Function[SymExpr]]]:
    if len(args) == 0:
      return lambda x: x, state, args, new_args,functions
    # print("args list",args)
    # print("in evalaute args args value",args[0])
    (k0, arg0,abs_arg0,func) = _evaluate(args[0], state)
    #p1 = _evaluate(particle.update(cont=args[0]))
    #if not p1.finished:
    #  args[0] = p1.cont
    #  return p1, args, new_args
    # if arg0 is Const:
    #   print("arg0 is constant ",arg0)
    # else:
    #   print("arg0 is not constant ",arg0)


    # print("arg0 value is the : ", arg0)
    # # print("value of k0",k0)
    # print("abs_arg0 is ", abs_arg0)

    new_args.append(arg0)
    new_args.append(abs_arg0)
    k1, new_state, new_args, new_new_args,func1 = _evaluate_args(state, args[1:], new_args)
    def k(rest: Expr):
      return k0(k1(rest))
    # print("val of k1",k1)
    # print("value of the k is : ", k)
    # # print("value of the new_state is :", new_state)
    # print("value of the new_args is :", new_args )
    # print("value of the new_new_args is :", new_new_args)




    return k, new_state, new_args, new_new_args,func

  # Evaluate built-in operators
  def _evaluate_ops(op: Operator, args: List[Expr],abs_args:AbsSymExpr) -> tuple[Callable[[[Expr], Expr],Expr]]:
    #def _evaluate_unops(particle: Particle, constructor: Any, args: SymExpr) -> Particle:
    #  return particle.update(cont=constructor(args), finished=True)

    # def _evaluate_binops(particle: Particle, constructor: Any, args: SymExpr) -> Particle:
    #  fst, snd = get_pair(args)
    #  return particle.update(cont=constructor(fst, snd), finished=True)

    #def _evaluate_triops(particle: Particle, constructor: Any, args: SymExpr) -> Particle:
    #  fst, args2 = get_pair(args)
    #  snd, trd = get_pair(args2)
    #  return particle.update(cont=constructor(fst, snd, trd), finished=True)
    # print("list of the abstract args in the evaluate ops", abs_args)
    # print("list  evaluate ops", args)
    # Map to the correct operator
    match op.name:
      case "add":
        # raise ValueError("not implemented")
        return GenericOp(Operator.add, [args[0], args[1]]), AbsAdd(abs_args[0], abs_args[1])
        #return _evaluate_binops(particle, Add, args)
      case "sub":
        return GenericOp(Operator.sub, args), AbsAdd(abs_args[0], AbsMul(AbsConst(-1), abs_args[1]))
        # a - b = a + (-1 * b)
        #return _evaluate_binops(particle, lambda fst,snd: Add(fst, Mul(Const(-1), snd)), args)
      case "mul":
        # raise ValueError("Not Implemented")
        return GenericOp(Operator.mul,[args[0],args[1]]), AbsMul(abs_args[0],abs_args[1])
        #return _evaluate_binops(particle, Mul, args)
      case "div":
        # raise ValueError("Not Implemented")
        return GenericOp(Operator.div, [args[0], args[1]]), AbsDiv(abs_args[0], abs_args[1])
        #return _evaluate_binops(particle, Div, args)
      case "eq":
        return GenericOp(Operator.eq,[args[0],args[1]]), AbsEq(abs_args[0], abs_args[1])
        # raise ValueError("Not Implemented")
        #return _evaluate_binops(particle, Eq, args)
      case "lt":
        return GenericOp(Operator.lt,[args[0],args[1]]), AbsLt(abs_args[0], abs_args[1])
        # raise ValueError("Not Implemented")
        #return _evaluate_binops(particle, Lt, args)
      case "cons":
        head, tail = abs_args
        if isinstance(tail, UnkE):
          abstract_list = UnkE(list(dict.fromkeys(head.rvs() + tail.rvs())))
        else:
          abstract_list = AbsLst([head] + get_abs_lst(tail))
        return GenericOp(Operator.cons, args), abstract_list
          # def _make_cons(fst, snd):
          #   l = get_abs_lst(snd)
          #   # If the list is a list of unknown length (UnkE), the result is also an unknown length list
          #   if isinstance(l, UnkE):
          #     parents = []
          #     for p in fst.rvs() + l.rvs():
          #       parents += [p]
          #     return UnkE(parents)
          #   return AbsLst([fst] + l)
        
          # return GenericOp(Operator.lst,(lambda fst,snd: Lst([fst] + get_lst(snd)))), AbsLst(_make_cons)
        #return _evaluate_binops(particle, lambda fst,snd: Lst([fst] + get_lst(snd)), args)
      case "lst":
        return GenericOp(Operator.lst, args), AbsLst(abs_args)
        #return _evaluate_unops(particle, _make_list, args)
      case "pair":
        return Pair(args[0], args[1]), AbsPair(abs_args[0], abs_args[1])
        # return Pair,[args[0],args[1]], AbsPair(abs_args[0],abs_args[1])
        # raise ValueError("Not Implemented")
        #return _evaluate_binops(particle, Pair, args)
      case "gaussian":
        # GenericOp(Operator.gaussian, [
        #   args[0],
        #   args[1],
        # ])


        return GenericOp(Operator.gaussian,[args[0], args[1]]),(AbsNormal(abs_args[0], abs_args[1]))
        # return (Normal(args[0], args[1]))
        # raise ValueError("not implmented")
        #return _evaluate_binops(particle, Normal, args)
      case "beta":
        return GenericOp(Operator.beta,[args[0], args[1]]),(AbsBeta(abs_args[0], abs_args[1]))
        # raise ValueError("not implmented")
        #return _evaluate_binops(particle, Beta, args)
      case "bernoulli":
        return GenericOp(Operator.bernoulli,(args[0])),AbsBernoulli(abs_args[0])
        # raise ValueError("not implmented")
        #return _evaluate_unops(particle, Bernoulli, args)
      case "binomial":
        return GenericOp(Operator.binomial,[args[0], args[1]]),(AbsBinomial(abs_args[0], abs_args[1]))
        # raise ValueError("not implmented")
        #return _evaluate_binops(particle, Binomial, args)
      case "beta_binomial":
        return GenericOp(Operator.beta_binomial,[args[0], args[1], args[2]]),(AbsBetaBinomial(abs_args[0], abs_args[1], abs_args[2]))
        # raise ValueError("not implmented")
        #return _evaluate_triops(particle, BetaBinomial, args)
      case "negative_binomial":
        raise ValueError("not implmented")
        #return _evaluate_binops(particle, NegativeBinomial, args)
      case "exponential":
        raise ValueError("not implmented")
        # Exponential is a special case of Gamma
        # Represented as a Gamma so can be detected when applying
        # conjugacy rules
        #return _evaluate_binops(particle, Gamma, Pair(Const(1.0), args))
      case "gamma":
        return GenericOp(Operator.gamma,[args[0],args[1]]),AbsGamma(abs_args[0],abs_args[1])
        #return _evaluate_binops(particle, Gamma, args)
      case "poisson":
        raise ValueError("not implmented")
        #return _evaluate_unops(particle, Poisson, args)
      case "delta":
        return GenericOp(Operator.delta, (args[0])), AbsDelta(abs_args[0])
        # raise ValueError("not implmented")
        #return _evaluate_unops(particle, Delta, args)
      case "categorical":
        raise ValueError("not implmented")
        #return _evaluate_triops(particle, Categorical, args)
      case "uniform_int":
        raise ValueError("not implmented")
        #a, b = get_pair(args)
        # For now, uniform only takes constants
        # [a, b]
        # Represented as a categorical distribution
        #match (particle.state.eval(a), particle.state.eval(b)):
        #  case (Const(a), Const(b)):
        #    assert isinstance(a, Number) and isinstance(b, Number)\
        #      and round(a) == a and round(b) == b and a <= b
        #    a, b = int(a), int(b)
        #    probs = Const(list(np.ones(b - a + 1) / (b - a + 1)))
        #    return _evaluate_ops(particle, Operator.categorical,
        #                          Pair(Const(a), Pair(Const(b), probs)))
        #  case _:
        #    raise ValueError(args)
      case "student_t":
        return GenericOp(Operator.student_t,[args[0], args[1], args[2]]),(AbsStudentT(abs_args[0], abs_args[1], abs_args[2]))
        # raise ValueError("not implmented")
        #return _evaluate_triops(particle, StudentT, args)
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
      # print("1")
      return Pair(args[0], _convert_args(args[1:]))

  # Evaluate list operations. Evaluates arguments first, interrupting if args did not finish evaluation
  def _evaluate_list( func: Identifier, args: List[Expr[SymExpr]]) -> tuple[Callable[[Any], Any], UnkE] | AbsLst | \
                                                                      tuple[Callable[[Expr], Any], Any, Any] | Any:
    assert func.module == 'List'
    match func.name:
      case 'hd':
        print("args in the list", args)
        (p1, _,old_args, new_args,func) = _evaluate_args(state,args, [])
        # if len(old_args) != 0:
        #   return p1.update(cont=Apply(func, old_args + new_args), finished=False)
        print("new args in the list before convert args", new_args)
        print("old args in the list", old_args)
        new_args = _convert_args(new_args)
        print("new_args in hd", new_args)
        new_args1, abs_new_args1 = get_pair(new_args)
        # exprs = get_lst(new_args1)
        abs_exprs = get_abs_lst(abs_new_args1)
        print("value of the new_args1 in the evaluate list.hd", new_args1)
        # If the list is unknown expression, the result is also an unknown expression
        if isinstance(abs_exprs, UnkE):
          return p1, Apply(Identifier("List", "hd"), [new_args1]), abs_exprs, func
        # if len(exprs) == 0:
        #   raise ValueError(new_args)
        # return p1.update(cont=exprs[0], finished=True)
        print("value of the abs_exprs in the evaluate list.hd", abs_exprs)
        return p1, Apply(Identifier("List", "hd"), [new_args1]), abs_exprs[0], func
      case 'tl':
        (p1, _,old_args, new_args,func) = _evaluate_args( state,args, [])
        # if len(old_args) != 0:
        #   return p1.update(cont=Apply(func, old_args + new_args), finished=False)
        new_args = _convert_args(new_args)
        new_args1, abs_new_args1 = get_pair(new_args)
        # exprs = get_lst(new_args1)
        abs_exprs = get_abs_lst(abs_new_args1)
        
        # If the list is unknown expression, the result is also an unknown expression
        if isinstance(abs_exprs, UnkE):
          return p1, Apply(Identifier("List", "tl"), [new_args1]), abs_exprs, func
        if isinstance(abs_exprs[0], UnkE):
          return p1, Apply(Identifier("List", "tl"), [new_args1]), AbsLst(abs_exprs), func
        # if len(exprs) == 0:
        #   raise ValueError(new_args)
        # return p1.update(cont=Lst(exprs[1:]), finished=True)
        return p1, Apply(Identifier("List", "tl"), [new_args1]), AbsLst(abs_exprs[1:]), func
      case 'len':
        (p1, _, old_args, new_args,func) = _evaluate_args( state,args, [])
        # if len(old_args) != 0:
        #   return p1.update(cont=Apply(func, old_args + new_args), finished=False)
        new_args = _convert_args(new_args)
        new_args1, abs_new_args1 = get_pair(new_args)

        abs_exprs = get_abs_lst(abs_new_args1)
        length = AbsConst(UnkC()) if isinstance(abs_exprs, UnkE) else AbsConst(len(abs_exprs))
        return p1, Apply(Identifier("List", "len"), [new_args1]), length, func
      case 'range':
        (p1, _,old_args, new_args,func) = _evaluate_args( state,args, [])
        # if len(old_args) != 0:
        #   return p1.update(cont=Apply(func, old_args + new_args), finished=False)
        # print("new_args before convert args", new_args)
        new_args = _convert_args(new_args)

        # Range only takes constants
        # print("old_args",old_args)
        # print("new_args", new_args)
        # a,abs_a,b,abs_b = new_args
        [a, abs_val] = get_pair(new_args)
        [abs_a, abs_val1]= get_pair(abs_val)
        [b, abs_b] = get_pair(abs_val1)
        # print("cons_val", cons_val)
        # print("abs_val", abs_val)
        # a,abs_a,b,abs_b = new_args
        # abs_a,abs_b = abs_val
        # print("a",a)
        # print("b",b)
        # print("abs_a",abs_a)
        # print("abs_b",abs_b)
        if isinstance(abs_a, UnkE) or isinstance(abs_b, UnkE):
          return p1,Apply(Identifier("List", "range"), [a,b]) ,UnkE([]), func
        match state.Abs_eval(abs_a), state.Abs_eval(abs_b):
          case AbsConst(abs_a), AbsConst(abs_b):
            assert isinstance(abs_a, Number) and isinstance(abs_b, Number) and abs_a <= abs_b
            l : List[AbsSymExpr] = list(map(AbsConst, range(int(abs_a), int(abs_b) )))
            return p1, Apply(Identifier("List", "range"), [a,b]), AbsLst(l), func
          case _:
            raise ValueError(new_args)
        #     assert isinstance(abs_a, Number) and isinstance(abs_b, Number) and abs_a <= abs_b
        #     l : List[AbsSymExpr] = list(map(AbsConst, range(int(abs_a), int(abs_b))))
        #     print("value of l",l)
        #     # return l
        #     return p1,[a,b],(l),func
          # case _:
          #   raise ValueError(new_args)
      case 'rev':
        (p1, state2,old_args, new_args,func) = _evaluate_args( state,args, [])
        # if len(old_args) != 0:
        #   return p1.update(cont=Apply(func, old_args + new_args), finished=False)
        new_args = _convert_args(new_args)

        exprs, abs_list = get_pair(new_args)
        abs_exprs = get_abs_lst(abs_list)
        # If the list is unknown expression, the result is also an unknown expression
        if isinstance(abs_exprs, UnkE):
          return p1, Apply(Identifier("List", "rev"), [exprs]), abs_exprs, func
        l: List[AbsSymExpr] = abs_exprs[::-1]
        # return p1.update(cont=Lst(exprs[::-1]), finished=True)
        return p1, Apply(Identifier("List", "rev"), [exprs]), AbsLst(l), func
      case 'map':
        map_func = args[0]
        assert isinstance(map_func, Identifier)

        (p1, old_args, new_args,func) = _evaluate_args( state,args[1:], [])
        # if len(old_args) != 0:
        #   return p1.update(cont=Apply(func, old_args + new_args), finished=False)
        new_args = _convert_args(new_args)

        # Map is syntactic sugar of a list, calling the function on each element
        exprs, abs_exprs = get_lst(new_args)
        new_e = Lst([])
        for e in exprs[::-1]:
          new_e = GenericOp(Operator.cons, [
            Apply(map_func, [e]),
            new_e,
          ])
          
        # If the list is unknown expression, the result is also an unknown expression
        if isinstance(exprs, UnkE):
          return p1.update(cont=exprs)
        new_e = AbsLst([])
        for e in exprs[::-1]:
          new_e = GenericOp(Operator.cons, [
            Apply(map_func, [e]),
            new_e,
          ])  
        return _evaluate(p1.update(cont=new_e))
      case _:
        raise ValueError(func)
      
  def evaluate_map(func: Identifier, args: List[Expr[SymExpr]]) -> tuple[Callable[[Any], Any], UnkE] | AbsLst | \
                                                                      tuple[Callable[[Expr], Any], Any, Any] | Any:
        map_func = args[0]
        assert isinstance(map_func, Identifier)

        # Keep the existing argument evaluation.
        (p1, _, old_args, new_args, func) = _evaluate_args(
          state, args[1:], []
        )
        new_args = _convert_args(new_args)
        new_args1, abs_new_args1 = get_pair(new_args)

        abs_exprs = get_abs_lst(abs_new_args1)

        # Preserve existing behavior for an unknown list.
        if isinstance(abs_exprs, UnkE):
          return (p1,Apply(Identifier("List", "map"),[map_func, new_args1]),abs_exprs,func)

        function = functions[map_func]

        if len(function.args) != 1:
          raise ValueError(
            "Map compilation requires exactly one parameter"
          )

        parameter = function.args[0]
        while isinstance(parameter, list) and len(parameter) == 1:
          parameter = parameter[0]

        if not isinstance(parameter, Identifier):
          raise ValueError(
            "Map compilation requires an identifier parameter"
          )

        concrete_values = []
        abstract_values = []

        mapped_functions = (
          list(func)
          if isinstance(func, list)
          else ([func] if func else [])
        )

        function_emitted = False
        tail = new_args1

        for abs_e in abs_exprs:
          element = Apply(Identifier("List", "hd"), [tail])

          if isinstance(abs_e, AbsConst) and not isinstance(abs_e.v, UnkC):
            element = Const(abs_e.v)

          # Evaluate each invocation to track its abstract result.
          k, concrete, abstract, generated = _evaluate(
            Apply(map_func, [(element, abs_e)]),
            state,
          )

          # Compile the body once under the original function name.
          if not function_emitted:
            if generated:
              mapped_functions.extend(
                generated
                if isinstance(generated, list)
                else [generated]
              )

            mapped_functions.append(Function(map_func,[parameter],k(concrete)))
            function_emitted = True

          # Emit a call instead of placing the body in the main program.
          concrete_values.append(Apply(map_func, [element]))
          abstract_values.append(abstract)

          tail = Apply(Identifier("List", "tl"), [tail])

        return (p1,Lst(concrete_values),AbsLst(abstract_values),mapped_functions)

  # Evaluate file operations. These only take constants as arguments
  def _evaluate_file(particle: Particle, func: Identifier, args: List[Expr[SymExpr]]) -> Particle:
    assert func.module == 'File'
    match func.name:
      case 'read':
        (p1, old_args, new_args,func) = _evaluate_args(particle, args, [])
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

  # Convert pattern to an expression producing a value
  # def _convert_pattern(pattern):
  #   if len(pattern) == 1:
  #     return pattern[0]
  #   else:
  #     raise ValueError("Not implemented")




  #evaluate fold for fixed point and looping
  def _evaluate_fold(state: SymState, func: Identifier, lst: Expr, acc: Expr, abs_lst: AbsSymExpr, abs_acc: AbsSymExpr) -> tuple[
    Callable[[Expr], Any], Any, Any, Any]:

    # print(f"--- FOLD LOOP STEP ---")
    # print(f" incoming abs_acc : {abs_acc}")
    state_old = copy(state)
    state_new = copy(state)
    # print("state_old", state_old.Abs_state)
    # print("state_new", state_new.Abs_state)
    # print("state",state.Abs_state)
    # state_old_ctx = copy(state.ctx)
    val_lst = [(lst, abs_lst), (acc, abs_acc)]
    # aaply function for the val_lst
    new_args = _convert_args(val_lst)
    # print("args in the evalute fold ",new_args)
    (l_val, r_val) = get_pair(new_args)
    (conc_l_val, abs_l_val) = l_val
    (conc_r_val, abs_r_val) = r_val
    function = functions[func]
    # print("before state_new context val", state_new.ctx)
    ctx = copy(state_new.ctx)
    # saved_ctx = copy(state.ctx)
    combined_abs_vals = AbsPair(abs_l_val, abs_r_val)
    # print("function args ", (function.args[0]))
    state_new.ctx = match_pattern(function.args, combined_abs_vals)
    # state.ctx = match_pattern(function.args, combined_abs_vals)

    # (p3, conc_val, abs_val, func_val1) = _evaluate(function.body, state_new)
    # print("state new before the evaluate of function.body")
    # print("state_new", state_new.Abs_state)
    (p3, conc_val, abs_val, func_val1) = _evaluate(function.body, state_new)
    # print("state new after the evaluate of function.body")
    # print("state_new", state_new.Abs_state)

    state_new.ctx = ctx
    # print(f"  body evaluated   : {abs_val}")
    # print("state_new", state_new.ctx)
    # state.ctx = saved_ctx

    acc_new = state.narrow_join_expr(abs_acc, abs_val, state_new)
    state.counter = max(state.counter, state_new.counter)

    # acc_new = state.narrow_join_expr(abs_acc, abs_val, state)
    # state.counter = max(state.counter, state.counter)
    state.abs_clean(acc_new)
    # print(f"   acc_new   : {acc_new}")
    # print(f"abs_acc: {abs_acc}")
    # print(f"  Match checks     : abs_acc==acc_new is {abs_acc == acc_new} | state_old==state is {state.__eq__(state_old) == state.__eq__(state)}")
    # print("state_old", state_old.Abs_state)
    # print("state_new", state_new.Abs_state)
    # print("state", state.Abs_state)
    if abs_acc == acc_new and state.__eq__(state_old) == state.__eq__(state):
    # if abs_acc == acc_new and state_old_ctx == state.ctx:

      func_body_val = p3(conc_val)

      func_name = Identifier(None, f"{func.name}{1}")
      (val_args1, val_args2) = function.args
      # print("val_args1", val_args1[0])
      # print("val_args2", val_args2[0])
      func_val = Function(func_name, (val_args1[0],val_args2[0]), func_body_val)
      # print("what is function args", function.args[0])

      

      # print("value of the funcval1 is : ", func_val1)
      # print("func_val in evaluate fold", func_val)

      # (p4,new_conc_val,new_abs_val,func_val2) = _evaluate(Apply(func,[conc_lst_val,acc_new]),state_old)
      return lambda x: x , Fold(func_name, lst, acc), acc_new, func_val
      # raise ValueError("Infinite loop.")
    else:
      # print("In the else condition")
      # print("state_old", state_old.Abs_state)
      # print("state_new", state_new.Abs_state)
      # print("state", state.Abs_state)
      # print("end in else condition")
      # raise ValueError("Infinite loop.")
      (p4, conc_val_fold, abs_val_fold, fold_func) = _evaluate_fold(state, func, lst, acc, abs_lst, acc_new)
      # print("fold func value in the else case of evaluate fold", fold_func)
      #
      # # (p4, conc_val_fold, abs_val_fold, fold_func)=_evaluate(Fold(func, abs_lst_val, acc_new), state)
      # # def cont_fold(res:Expr):
      # #   return p3(p4(res))
      #
      return lambda x: x, conc_val_fold, abs_val_fold, fold_func
    # return p_val,conc_fold, abs_fold, func_fold


  # Evaluate the particle, returning an evaluated particle
  # or a particle with the next expression to evaluate if interrupted by resample
  def _evaluate(expr: Expr, state: SymState) -> tuple[
    Callable[[Expr], Any], Any, Any, Any]:
    #if isinstance(particle.cont, SymExpr):
    #  return particle.update(finished=True)
    # state1: AbsSymState
    # particle: AbsParticle
    # print("evaluating ", expr)
    match expr:
      case Const(v):
        concrete = Const(v) if isinstance(v, bool) else v
        return lambda x: x, concrete,AbsConst(v), []
      case Identifier(_, _):

        # res = particle.update(cont=particle.state.ctx[particle.cont])
        # print("value of the state.ctx", state.ctx[expr])
        # print("expr in evaluate file ", expr)
        (sysval, abssymval) = state.ctx[expr]
        # print("val of the sysval", sysval)
        # print("val of the abssymval", abssymval)


        return lambda x: x, sysval,abssymval,[]
        #return particle.update(cont=particle.state.ctx[particle.cont], finished=True)
      # it returns the pair of expr and abs_expr should state.ctx should get the expr and state1.ctx should get the abs_expr
      case GenericOp(op, args):
        (k1, new_state, _, new_args,func_args) = _evaluate_args(state, args, [])
        # print("k1 and new_state",_evaluate_args(state, args, []))
        #(p1, old_args, new_args) = _evaluate_args(particle, args, [])
        #if len(old_args) != 0:
        #  return p1.update(cont=GenericOp(op, old_args + new_args), finished=False)
        #new_args = _convert_args(new_args)
        # print("values:: ", (k1, _evaluate_ops(op, new_args)))
        # print("new args value", new_args)

        # const_list = []
        # absconst_list = []
        #
        # # Separate elements by type
        # for item in new_args:
        #   if isinstance(item, Const):
        #     const_list.append(item)
        #   elif isinstance(item, AbsConst):
        #     absconst_list.append(item)
        #
        # # Combine lists in desired order
        # rearranged_list = const_list + absconst_list
        # print("new_args is ",new_args)
        # val = state.get_name_by_rv(new_args[1])
        # print("val.............................in new args",val)


        ret, abs_ret = _evaluate_ops(op, new_args[0::2],new_args[1::2])
        # print("what is op",op)
        # print("the new value of the new args list is ",ret)


        # bernoulli_value = next((item for item in new_value_args if isinstance(item, Bernoulli)), None)



        # print("genop in abs_ret", abs_ret)

        return k1,ret,abs_ret, func_args  #changes done here for the value not reaching
        #return _evaluate_ops(p1, op, new_args)
      case Fold(func, lst, acc):
        # print("in fold case")
        # print("func", func)
        # print("lst", lst)
        # print("acc", acc)
        (p1,conc_lst,abs_lst,func_lst) = _evaluate(lst,state)
        # print("conc_lst",conc_lst)
        # print("abs_lst",abs_lst)
        conc_lst_val = conc_lst
        # print("conc_lst_val",conc_lst_val)
        abs_lst_val = abs_lst
        (p2,conc_acc,abs_acc,func_acc) = _evaluate(acc,state)
        # print("conc_acc",conc_acc)
        # print("abs_acc",abs_acc)
        acc_val = abs_acc

        # Specialize scalar folds exactly like list folds, retaining the
        # generated function and Fold node for the Python compiler.
        scalar_input = not isinstance(abs_lst_val, (AbsLst, UnkE))
        if scalar_input:
          abs_lst_val = AbsLst([abs_lst_val])

        match abs_lst_val:
          case AbsLst(abs_exprs):
            calls = []

            def k_fold(final_expr: Expr):
              for target, call in reversed(calls):
                final_expr = Let([target], call, final_expr)
              return p1(p2(final_expr))

            if not abs_exprs:
              return (
                k_fold,
                conc_acc,
                acc_val,
                func_acc + func_lst,
              )

            abs_hd = abs_exprs[0]
            function = functions[func]
            runtime_parameters = False

            # Recognize the scalar Gaussian transition/observation pattern.
            match function.args, function.body:
              case (
                [[obs_arg], [previous_arg]],
                LetRV(
                  var=x,
                  annotation=annotation,
                  distribution=GenericOp(
                    op=Operator.gaussian,
                    args=[previous, Const(q)],
                  ),
                  body=Let(
                    var=[],
                    value=Observe(
                      condition=GenericOp(
                        op=Operator.gaussian,
                        args=[observed, Const(r)],
                      ),
                      observation=observation,
                    ),
                    body=result,
                  ),
                ),
              ):
                runtime_parameters = (
                  annotation is not Annotation.sample
                  and previous == previous_arg
                  and observation == obs_arg
                  and observed == x == result
                  and isinstance(acc_val, AbsRandomVar)
                  and isinstance(state.Abs_distr(acc_val), AbsNormal)
                  and not state.Abs_distr(acc_val).rvs()
                  and type(q) in (int, float) and q >= 0
                  and type(r) in (int, float) and r > 0
                )

            # Keep formal parameters in the generated helper.
            expand_fold = False
            match acc_val:
              case AbsPair(
                AbsConst(bool()),
                AbsPair(AbsLst(), AbsLst()),
              ):
                expand_fold = True

            emitted = []
            current = conc_acc
            tail = conc_lst_val

            for abs_hd in (
              abs_exprs if expand_fold else abs_exprs[:1]
            ):
              caller_ctx = state.ctx
              saved_runtime = getattr(
                state, "_fold_runtime_parameters", False
              )
              state.ctx = match_pattern(
                function.args, AbsPair(abs_hd, acc_val)
              )
              state._fold_runtime_parameters = runtime_parameters

              try:
                p_apply, conc_body, abs_body, a_func = _evaluate(
                  function.body, state
                )
              finally:
                state._fold_runtime_parameters = saved_runtime
                state.ctx = caller_ctx

              new_func = new_fold_function()
              parameters = [
                Identifier(
                  None,
                  str(group[0]) if len(group) == 1
                  else "(" + ", ".join(str(part[0]) for part in group) + ")",
                )
                for group in function.args
              ]
              emitted.append(
                Function(new_func, parameters, p_apply(conc_body))
              )
              emitted.extend(a_func)
              acc_val = abs_body

              if expand_fold:
                state.counter += 1
                target = Identifier(None, f"fold_acc_{state.counter}")
                item = Apply(Identifier("List", "hd"), [tail])
                calls.append((target, Apply(new_func, [item, current])))
                current = target
                tail = Apply(Identifier("List", "tl"), [tail])
              else:
                current = Fold(new_func, conc_lst_val, conc_acc)

            return (
              k_fold, current, acc_val,
              emitted + func_acc + func_lst,
            )
            # if len(abs_exprs) ==0:
            #   return p2,conc_acc,acc_val,func_acc
            # else:
            #   abs_hd,abs_tl = abs_exprs[0],abs_exprs[1:]
            #   # print("hd",abs_hd)
            #   # print("tl",abs_tl)
            #   # match lst_val:
            #   #   case Lst(exprs):
            #   #     conc_hd, conc_tl = exprs[0], exprs[1:]

            #   if scalar_input:
            #     conc_hd = conc_lst
            #     conc_tl = []
            #   elif hasattr(conc_lst, 'exprs'):
            #     conc_hd = conc_lst.exprs[0]
            #     conc_tl = conc_lst.exprs[1:] if len(conc_lst.exprs) > 1 else []
            #   elif isinstance(conc_lst, list) and len(conc_lst) > 0:
            #     conc_hd = conc_lst[0]
            #     conc_tl = conc_lst[1:]
            #   else:
            #     conc_hd = Const(abs_hd.v) if hasattr(abs_hd, 'v') else abs_hd
            #     conc_tl = []

            #   # The emitted helper runs for every element. Keep its formal
            #   # parameters in generated code, while using the first element
            #   # and accumulator only for abstract inference.
            #   caller_ctx = state.ctx
            #   state.ctx = match_pattern(
            #     functions[func].args, AbsPair(abs_hd, acc_val)
            #   )
            #   try:
            #     (p_apply, conc_body, abs_body, a_func) = _evaluate(
            #       functions[func].body, state
            #     )
            #   finally:
            #     state.ctx = caller_ctx
            #   new_func = new_fold_function()

            #   state.counter += 1
            #   tempvar = Identifier(None, f"x{state.counter}")
              
            #   func_body_val = p_apply(conc_body)
            #   val_func_args1, val_func_args2 = functions[func].args
            #   if isinstance(val_func_args1, list) and len(val_func_args1) == 1:
            #     val_func_args1 = val_func_args1[0]
            #   if isinstance(val_func_args2, list) and len(val_func_args2) == 1:
            #     val_func_args2 = val_func_args2[0]
            #   # print("func_val in evaluate fold", "(" + ", ".join(
            #   #     "(" + ", ".join(str(part[0]) for part in group) + ")"
            #   #     for group in functions[func].args
            #   # ) + ")")
            #   func_val = Function(new_func, [Identifier(None, str(group[0]) if len(group) == 1 else "(" + ", ".join(str(part[0]) for part in group) + ")") for group in functions[func].args], func_body_val)
              
            #   # func_val = Function(new_func, functions[func].args, func_body_val)
              

            #   val_fold = Fold(new_func, conc_lst_val, conc_acc)

            #   def k_fold(final_expr: Expr):
            #     return p1(p2(final_expr))
              
            #   # def val2(res:Expr):
            #   #   return p2(functions)
            #   # (p5, conc_e, abs_e, e_func) = _evaluate(val2.body, state)
            #   return k_fold, val_fold,abs_body, [func_val] + a_func + func_acc +func_lst
              # return p5,conc_e,abs_e,e_func

          case UnkE():
              # print("unknown case in fold")
              (p4, conc_val_fold, abs_val_fold, fold_func) = _evaluate_fold(state, func, conc_lst, conc_acc, abs_lst, abs_acc)
              # print("p4: ",fold_func)

              return p4, conc_val_fold, abs_val_fold, fold_func
            # print("UnkE")
            # state_old = copy(state)
            # state_new = copy(state)
            # val_lst = [(conc_lst_val,abs_lst_val),(conc_acc,abs_acc)]
            # #aaply function for the val_lst
            # new_args = _convert_args(val_lst)
            # function = functions[func]
            # ctx = copy(state_new.ctx)
            # state_new.ctx = match_pattern(function.args, new_args)
            # (p3,conc_val,abs_val,func_val1) = _evaluate(function.body, state_new)
            # state_new.ctx = ctx
            #
            # acc_new = state.narrow_join_expr(acc_val,abs_val,state_new)
            # state.counter = max(state.counter, state_new.counter)
            # state.clean(acc_new)
            # # state.counter_func += 1 need to work on this for dynaminc
            #
            # if acc_val == acc_new and state_old == state:
            #
            #   def p_val(res:Expr):
            #     return p1(p2(res))
            #
            #
            #   func_name = Identifier(None, f"{func.name}{1}")
            #   func_val = Function(func_name, function.args, p3(conc_val))
            #
            #   # (p4,new_conc_val,new_abs_val,func_val2) = _evaluate(Apply(func,[conc_lst_val,acc_new]),state_old)
            #   return p_val,Fold(func_name,conc_lst,conc_acc),acc_new,func_val
            # else:
            #   print("elseee")
            #   (p1, conc_lst, abs_lst, func_lst) = _evaluate(abs_lst, state)
            #
            #   (p2, acc_conc, acc_abs, func_acc) = _evaluate(acc_new, state)
            #   (p4, conc_val_fold, abs_val_fold, fold_func) = _evaluate_fold(state, func, conc_lst,conc_acc, abs_lst, abs_acc)
              # (p4, conc_val_fold, abs_val_fold, fold_func)=_evaluate(Fold(func, abs_lst_val, acc_new), state)
              # def cont_fold(res:Expr):
              #   return p3(p4(res))

              # return p4, conc_val_fold, acc_val, func_acc
        # raise ValueError("Not implemented")
        # Fold is syntactic sugar for a chain of let expressions
        # It's a bounded loop
        #p1 = _evaluate(particle.update(cont=lst))
        #if not p1.finished:
        #  return p1.update(cont=Fold(func, p1.cont, acc), finished=False)
        #lst_val = p1.final_expr

        #p2 = _evaluate(p1.update(cont=acc))
        #if not p2.finished:
        #  return p2.update(cont=Fold(func, lst_val, p2.cont), finished=False)
        #acc_val = p2.final_expr
        #match lst_val:
        #  case Lst(exprs):
        #    if len(exprs) == 0:
        #      return p2.update(cont=acc_val, finished=True)
        #    else:
        #      hd, tl = exprs[0], exprs[1:]
        #      # Create a temporary variable to store the result of the function
        #      tempvar = p2.state.ctx.temp_var()
        #      e = Let([tempvar],
        #              Apply(func, [hd, acc_val]),
        #              Fold(func, Lst(tl), tempvar))
        #      return _evaluate(p2.update(cont=e))
        #  case _:
        #    raise ValueError(p1.cont)
      case Apply(func, args):
          #abstarct
        if func.module is not None:

          match func:
            case Identifier(None, "new_var"):
              # return (lambda x: x,expr,args)
              return lambda x: x, expr,func
            case Identifier("List","range"):
              # print("func", func)
              # print("args", args)
              (p1,conc_args,new_args,func_args) = _evaluate_list(func, args)
              return p1,conc_args,new_args, func_args
            case Identifier("List","hd" | "len"):
              # print("func", func)
              # print("args", args)
              (p1,conc_args,new_args,func_args) = _evaluate_list(func, args)
              return p1,conc_args,new_args, func_args
            case Identifier("List","tl" | "rev"):
              # print("func", func)
              # print("args", args)
              (p1,conc_args,new_args,func_args) = _evaluate_list(func, args)
              return p1,conc_args,new_args, func_args
            case Identifier("List","map"):
              # print("func", func)
              # print("args", args)
              (p1,conc_args,new_args,func_args) = evaluate_map(func, args)
              return p1,conc_args,new_args, func_args
            


        else:
          # print("going in apply else case")
          if len(args) > 0 and all(isinstance(a, tuple) and len(a) == 2 for a in args):
            # Fold supplies values that have already been evaluated.
            p1 = lambda x: x
            abs_args = [value for pair in args for value in pair]
            func_args = []
          else:
            conc_vals = list(args)
            (p1,_,conc_args,abs_args,func_args) = _evaluate_args(state, conc_vals, [])
          # print("args", args)
          # (new_args1,new_args2) = args
          # print("new_args1", new_args1)
          # conc_val,abs_val = new_args1
          # conc_val1, abs_val1 = new_args2
          # (p1,_,conc_args,abs_args,func_args) = _evaluate_args(state,[conc_val,conc_val1],[])
          # (p1, state2,conc_args, abs_args) = _evaluate_args(state, [conc_val,conc_val1], [])
          # assert len(abs_args) == 0 #mapping concrete to x_pre to abstract and do this for all the values in the context
          # abs_args1 = [abs_args[i] for i in range(1, len(abs_args), 2)]
          abs_args1 = abs_args[1::2]
          # print("abs_args1 in the apply", abs_args1)
          conc_args_extracted = abs_args[0::2]
          new_args = _convert_args(abs_args1)
          function = functions[func]
          # val = [p[0] if isinstance(p, list) and len(p) == 1 else p for p in function.args]
          ctx = copy(state.ctx)
          ctx_old = copy(state.ctx.context)
          
          def bundle_to_abs_pairs(lst):
              if len(lst) == 0:
                  return AbsConst(None)
              elif len(lst) == 1:
                  return lst[0]
              else:
                  return AbsPair(lst[0], bundle_to_abs_pairs(lst[1:]))

          abstract_args_structured = bundle_to_abs_pairs(abs_args1)
          
          # for i in range(len(val)):
          #   pattern_name = val[i]
          #
          #   val_abstract = abs_args1[i]
          #
          #   if isinstance(val_abstract, AbsLst) and len(val_abstract.exprs) == 1:
          #     val_abstract = val_abstract.exprs[0]
          # [val1, val2] = abs_args
          
          
          
            

          state.ctx = match_pattern(function.args, abstract_args_structured)


            # state.ctx.context[pattern_name] = (pattern_name, val_abstract)

          for pattern, concrete_arg in zip(function.args, conc_args_extracted):
            parameter = pattern

            while isinstance(parameter, list) and len(parameter) == 1:
              parameter = parameter[0]

            if isinstance(parameter, Identifier):
              _, abstract_arg = state.ctx[parameter]
              state.ctx[parameter] = (concrete_arg,abstract_arg)

          # print(f"Executing {func} body with bound vars: {val}")
          # print("state.ctx val after match pattern",state.ctx1)
          # print(f"DEBUG: Context before body evaluation: {state.ctx.context}")

          (p2, conc_func, abs_func,func_val) = _evaluate(function.body, state)
          state.ctx = ctx
          # state.ctx.context = ctx_old
          # print("state.ctx val after again evaluate func is called",state.ctx1)
          # print("===============================================================")

          # new_func = Identifier(None, f"new_make_obs1")
          #
          # func_body_val = p2(func_val)
          # func_lst = Function(new_func,functions[func].args, func_body_val)
          # print(func_val)
          # print("new_func list ", [func_lst])
          def k_final(final_res:Expr):
            return (p1(p2(final_res)))
          
          
          if not func_val:
              combined_apply_funcs = func_args if isinstance(func_args, list) else [func_args]
          elif not func_args:
              combined_apply_funcs = func_val if isinstance(func_val, list) else [func_val]
          else:
              val_list = func_val if isinstance(func_val, list) else [func_val]
              args_list = func_args if isinstance(func_args, list) else [func_args]
              combined_apply_funcs = val_list + args_list

          return k_final, conc_func,abs_func, combined_apply_funcs

              # new_args = _convert_args(new_args)
              #
              #
              # print("new_args", new_args)
              #
              # a, abs_val = get_pair(new_args)
              # abs_a, abs_val1 = get_pair(abs_val)
              # b, abs_b = get_pair(abs_val1)
              #
              # const_val_a,abs_cons_val_a = state.eval(a, abs_a)
              # cons_val_b,abs_const_val_b = state.eval(b, abs_b)
              # print("const_val_a", const_val_a)
              # print("abs_cons_val_a", abs_cons_val_a)
              # print("cons_val_b", cons_val_b)
              # print("abs_const_val_b", abs_const_val_b)


              # match state.eval(a, abs_a), state.eval(b, abs_b):
              #   case AbsConst(abs_a), AbsConst(abs_b):
              #     assert isinstance(abs_a, Number) and isinstance(abs_b, Number) and abs_a <= abs_b
              #     l: List[SymExpr] = list(map(AbsConst, range(int(abs_a), int(abs_b))))
              #     return p1, Lst(l),AbsLst(l)
              #   case _:
              #     raise ValueError(new_args)

              # return lambda x: x, Lst([const_val_a,cons_val_b]),AbsLst([abs_cons_val_a,abs_const_val_b])


              # return p1, Lst(a)
              # return p1,
              # (p1,lst,abs_lst) = _evaluate_list(func,args)
              # return p1, lst,abs_lst
            #   return p1,lst_val,abs_lst
            # case Identifier(None, "File"):
            #   (p2,lst_val1,abs_lst1) = _evaluate_file(func,args)
            #   return p2,lst_val1,abs_lst1



          # raise ValueError("Not implemented.")
        #if func.module is not None:
          # use lib functions
        #  match func.module:
        #    case "List":
        #      return _evaluate_list(particle, func, args)
        #    case "File":
        #      return _evaluate_file(particle, func, args)
        #    case _:
        #      raise ValueError(func.module)
        #else:
        #  (p1, old_args, new_args) = _evaluate_args(particle, args, [])
        #  if len(old_args) != 0:
        #    return p1.update(cont=Apply(func, old_args + new_args), finished=False)
        #  converted_args = _convert_args(new_args)
        #  function = functions[func]
        #  ctx = copy(p1.state.ctx)
        #  p1.state.ctx = match_pattern(function.args, converted_args)
        #  p2 = _evaluate(p1.update(cont=function.body))
        #  if p2.finished:
        #    p2.state.ctx = ctx
        #  return p2
      case IfElse(cond, true, false):
        # raise ValueError("Not implemented")
        # p1 = _evaluate(particle.update(cont=cond))
        #if not p1.finished:
        #  return p1.update(cont=IfElse(p1.cont, true, false), finished=False)
        #cond_val = p1.final_expr

        (p1,res,abs_res,func_res) = _evaluate(cond,state)

        # A known condition must not evaluate the unreachable branch (which
        # may contain partial operations such as List.hd on an empty list).
        if not isinstance(abs_res, AbsConst):
          res, abs_res = state.eval(res, abs_res)
        if isinstance(abs_res, AbsConst) and not isinstance(abs_res.v, UnkC):
          p_branch, branch_res, branch_abs, branch_funcs = _evaluate(
            true if abs_res.v else false, state
          )
          return lambda result: p1(p_branch(result)), branch_res, branch_abs, branch_funcs

        # print("abs_res in ifelse",abs_res)
        # print("res in ifelse",res)
        def p(pret:Expr):
          return p1(pret)

        (p2, res1,abs_res1,func_res1) = _evaluate(true,state)

        (p3, res2,abs_res2,func_res2) = _evaluate(false,state)

        if pure(true,functions) and pure(false,functions):
          val_true = abs_res1
          val_false = abs_res2
          conc_ex_ite_val, abs_ex_ite_val = state.ex_ite(res,res1,res2,abs_res,val_true,val_false)
          return p, conc_ex_ite_val, abs_ex_ite_val,func_res

        else:
          conc_cond_val,abs_cond_val = state.value_expr(res,abs_res)
          match abs_cond_val:
            case AbsConst(v):
              if isinstance(v, UnkC):
                # (p2, res1,abs_res1) = _evaluate(true,state)
                then_val = abs_res1

                # (p3, res2,abs_res2) = _evaluate(false,state)
                else_val = abs_res2
                e = state.narrow_join_expr(then_val,else_val,state)

                return p,IfElse(conc_cond_val, p2(res1), p3(res2)),e, func_res
              elif v:
                return p,res1,abs_res1, func_res1
              else:
                return p,res2,abs_res2, func_res2
            case _:
              raise ValueError(abs_cond_val)



        # return p,IfElse(res, p2(res1), p3(res2)),
        # If both branches are pure, evaluate them and represent as ite symbolic expression
        #if len(cond_val.rvs()) > 0 and pure(true, functions) and pure(false, functions):
        #  p2 = _evaluate(p1.update(cont=true))
        #  if not p2.finished:
        #    return p2.update(cont=IfElse(cond_val, p2.cont, false), finished=False)
        #  then_val = p2.final_expr
        #  p3 = _evaluate(p2.update(cont=false))
        #  if not p3.finished:
        #    return p3.update(cont=IfElse(p1.cont, p2.cont, p3.cont), finished=False)
        #  return p3.update(cont=p3.state.ex_ite(cond_val, then_val, p3.final_expr), finished=True)
        #else:
          # If not pure, fully evaluate the condition, sampling RVs if necessary,
          # and then evaluate only the branch that is taken
        #  cond_value = p1.state.value_expr(p1.final_expr)
        #  match cond_value:
        #    case Const(v):
        #      if v:
        #        return _evaluate(p1.update(cont=true))
        #      else:
        #        return _evaluate(p1.update(cont=false))
        #    case _:
        #      raise ValueError(cond_value)
      case Let(pattern, v, body):
        ctx = copy(state.ctx)
        # print("what is pattern ", pattern)

        # print("what is the value of v",v)
        # print("value of the evaluate",_evaluate(v, state))
        # x=0
        # x = x+1
        # print("value of v",v)
        (k1, val,abs_val,func_val) = _evaluate(v, state)
        # print("x", x+1)
        # print("--------------------inside evaluate----------------")
        # print("val",val)
        # print("abs_val",abs_val)
        # print("k1",k1(expr))
        # print("--------------------------------------------------")
        #p1 = _evaluate(particle.update(cont=v))
        #if not p1.finished:
        #  return p1.update(cont=Let(pattern, p1.cont, body), finished=False)
        #val = p1.final_expr
        # state.ctx |= match_pattern(pattern,val,abs_val)
        state.ctx |= match_pattern(pattern, abs_val)
        # print("=============in the let case===============")
        # print("pattern",pattern)
        # print("abs_val",abs_val)
        # print("===========================================")
        (k2, res,abs_res,func_res) = _evaluate(body, state)
        state.ctx = ctx
        #p2 = _evaluate(p1.update(cont=body))
        # If the body is finished, restore the original context
        #if p2.finished:
        #  p2.state.ctx = ctx
        #return p2
        # print("value of the pattern", pattern)
        def k(rest: Expr):
            # print("the value of the pattern  is :",pattern)

            
            # print(f"the value of the pattern is: ({pattern_str})")
            # print("the value of the val variable is :",val)
            # print("the value of the rest variable is :",rest)
            # # print("the value of the identifier variable is :",identifier)
            # print("the value of the k2(rest value is :",k2(rest))
            # print("the value of the k1(rest value is :", k1(rest))
            # print("x before funcres",x+1)
            # print("func_res in the let case::::::::",func_res)
            # print("x after func res and before func val",x+1)

            # print("func_val in the let case::::::::",func_val)
            # print("x after both",x+1)
            # # print("--------------------------------------------")
            # print("continuation val of k", Let(pattern, val,k2(rest)))
            # print("--------------------------------------------")
            
            #changed here regarding the extra let statement
            # return k1(k2(rest))
            # return k1(Let(pattern, val,k2(rest)))
            emit_pattern = [Identifier(None, "(" + ", ".join(str(x[0]) if isinstance(x, (list, tuple)) and len(x) == 1 else str(x) for x in pattern) + ")")] if len(pattern) > 1 else pattern
            val1 = Let(emit_pattern, val, (k2(rest)))
            # val1 = Let((pattern), val,(k2(rest)))
            return k1(val1)

        def k4(rest1: Expr):
          return k(rest1)
          # print("abs_res in let",abs_res)
        # print("-------------------value in the let case------------------------")
        # print(k4(expr))
        # print(res)
        # print(abs_res)
        # print("-----------------------------------------------------------------")
        if not func_val:
            combined_let_funcs = func_res
        elif not func_res:
            combined_let_funcs = func_val if isinstance(func_val, list) else [func_val]
        else:
            if isinstance(func_val, list):
                combined_let_funcs = func_val + func_res
            else:
                combined_let_funcs = [func_val] + func_res
        # combined_let_funcs = [func_val] + func_res
        # print("combined_let_funcs",combined_let_funcs)

        return k4,res,abs_res, combined_let_funcs
        # return (lambda x: x, k(res),abs_res)
      case LetRV(identifier, annotation, distribution, expression):

        # print("in let rv")
        # # print("functions",functions.values())
        # print("the value of the distribution is", distribution)
        # print("the value of the expression is", expression)
        # print("before evaluate distribution the value of the state", state.Abs_state)
        # # print("the value of the state1", state1)
        # print("the dystr is ",distribution)


        # (k4, val_func,abs_val_func,func) = _evaluate(functions.values(), state)
        (k1, distr,abs_distr,func_distr) = _evaluate(distribution, state)
        # print("after evaluate distribution the value of the state", state.Abs_state)

        #p1 = _evaluate(particle.update(cont=distribution))
        #assert isinstance(p1.cont, Op) # should still be an Op even if not finished
        #if not p1.finished:
        #  return p1.update(cont=LetRV(identifier, annotation, p1.cont, expression), finished=False)
        #assert identifier.name is not None # RVs should always be named

        # print("the value of the abs_state", state1)
        # print("the value of the distr ", distr)
        # print("the distribution is", abs_distr)
        # new_distr = list(distr)
        # print("making the value of the ditr into the lsit", new_distr.count(0))

        # bernoulli_value = next((item for item in distr if isinstance(item, Bernoulli)), None)
        # abs_distri = next((item for item in distr if isinstance(item, AbsBernoulli)), None)
        # print("new abs_disrtibute vval", abs_distri)
        # print("before assume the value of the state", state.Abs_state)
        (k2, rv,abs_rv) = assume(identifier, annotation, distr, state, abs_distr)
        # print("after assume the value of the state", state.Abs_state)
        # After creating the RV, it is just a let expression
        # return _evaluate(p1.update(cont=Let([identifier], rv, expression)))
        # print("identifier : ", identifier)
        # print("distribution : ", distribution)
        # print("expression : ", expression)
        # print("rv : ",rv)

        ctx = copy(state.ctx)

        # p1 = _evaluate(particle.update(cont=v))
        # if not p1.finished:
        #  return p1.update(cont=Let(pattern, p1.cont, body), finished=False)
        # val = p1.final_expr
        # state.ctx |= match_pattern([identifier],rv,abs_rv)
        state.ctx |= match_pattern([identifier], abs_rv)
        # print("=============in the letrv case===============")
        # print("identifier",identifier)
        # print("abs_rv",abs_rv)
        # print("=============================================")
        # print("before evaluate expression the value of the state", state.Abs_state)
        (k4, res,abs_res,func_res) = _evaluate(expression, state)
        # print("after evaluate expression the value of the state", state.Abs_state)
        state.ctx = ctx
        # p2 = _evaluate(p1.update(cont=body))
        # If the body is finished, restore the original context
        # if p2.finished:
        #  p2.state.ctx = ctx
        # return p2
        # print("value of the pattern", pattern)

        def k3(rest: Expr):

          # print("the value of the pattern  is :", pattern)
          # print("the value of the val variable is :", val)
          # print("the value of the rest variable is :", rest)
          # print(" identifier variable is............ :",identifier)
          # print("rv........",rv)
          # print("the value of the k4(rest value is :",k4(rest))
          return Let([identifier], rv, k4(rest))


        def k(rest: Expr):
          # print("k1", k1(rest))
          # print("k2", k2(rest))
          # print("k3", k3(rest))
          return k1(k2(k3(rest)))


        # print("what is the value of the res in the letrv ",abs_res)
        # print("============================")
        # print("func_res in letrv, ", func_res)
        # print("============================")
        return k, res,abs_res,func_res
      case Observe(expression, v):
        # raise ValueError("not implemented")
        # print("expression....",expression)
        # print("in observe case")
        # print("before evaluate expression state is ", state.Abs_state)
        (p1,expr,abs_expr, func_expr) = _evaluate(expression, state)
        # print("after evaluate expression state is ", state.Abs_state)
        # print("p1............",p1)
        # print("evaluate expr....",expr)
        # print("abs_expr....",abs_expr)
        # print("v",v)
        # print("before evaluate v state is ", state.Abs_state)
        (p2,conc_v,abs_v, func_v) = _evaluate(v,state)
        # print("after evaluate v state is ", state.Abs_state)
        #assert isinstance(p1.cont, Op) # should still be an Op even if not finished
        #if not p1.finished:
        #  return p1.update(cont=Observe(p1.cont, v), finished=False)
        #d = p1.final_expr
        #assert isinstance(d, Op) # should still be an Op
        #p2 = _evaluate(p1.update(cont=v))
        #if not p2.finished:
        #  return p2.update(cont=Observe(d, p2.cont), finished=False)
        # print("p2..........",p2)
        # print("val expr in observe is ..............",expr)
        # print("val abs expr in observe is ..............", abs_expr)
        # print("val conc_v in observe is ..............",conc_v)
        # print("val abs_v in observe is ..............", abs_v)
        # print("func in observe", func_v)
        # print("before observe state is ", state.Abs_state)
        val = observe(Observe(expr, p2),expr,abs_expr,conc_v,abs_v,state)
        # print("after observe state is ", state.Abs_state)

        # Update the particle with the new score
        #return p2.update(score=w)
        def k1(rest:Expr):
          return p1(p2(rest))
        def k(rest: Expr):
          return k1(val(rest))
        #let () = gaussian(x,1.0) is coming from the return of the final observe
        return k,Apply(Identifier("Prob","null"),[]),abs_expr, func_v
      case Resample():
        raise ValueError("Not implemented)")
        # Resample interrupts the evalution
        #return particle.update(cont=Const(None), finished=False)
      case _:
        raise ValueError(expr)

  state = SSIState()
  # state1 = AbsSSIState()

  ret = _evaluate(expr, state)
  # print("ret ::",ret)
  # print("::::::::::::::::::::")
  # print("ret",ret)
  # print("ret1",ret1)
  # print("abs_ret",abs_ret)
  # print("fun",fun)
  # print("::::::::::::::::::::")

  return ret


def evaluate(
  program: Program,
  # n_particles: int,
  # method: type[SymState],
  file_dir: str,
  # seed: Optional[int] = None,
) ->Callable[[[Expr], Expr], Expr]:
  functions, expression = program.functions, program.main



  functions = {f.name: f for f in functions}



  # (k1,k2,k3) = Function(functions.keys,functions.values,functions.items)
  # Initialize particles
  #particles = ProbState(n_particles, expression, method, seed)
  # Evaluate particles until all are finished
  #while True:
  #  for i, particle in enumerate(particles):
  #    if particle.finished:
  #      continue
  #    else:
  #      particles[i] = evaluate_particle(particle, functions, file_dir)

    # If not all particles are finished, resample the particles
  #  if particles.finished:
  #    break
  #  else:
  #    particles.resample()


  # k1,val ,avs_val = evaluate_expr(functions,functions.values,file_dir)
  # print("================")
  # print("val",val)
  # print("avs_val",avs_val)
  # print("================")

  k, res,abs_res,func = evaluate_expr(expression, functions, file_dir)

  finalize = Apply(Identifier("Prob","finalize"), res)
  # new_finalize = Apply(Identifier("SSIState",""), [finalize])


  # ret = (k(finalize))
  # print("func.....",func)
  # print("ret",ret)

  main_trace_str = str(k(finalize))

  if func:
    flat_funcs = [f for item in func for f in (item if isinstance(item, list) else [item])]
    functions_prefix = "".join(str(f) for f in flat_funcs)
  else:
    # If func is a dict (original functions) or empty, no prefix is needed
    functions_prefix = ""

  # 3. Combine them cleanly
  ret = functions_prefix + main_trace_str











  # print("result is being printed",ret)

  return ret

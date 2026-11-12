from logging import exception
from typing import Optional, Tuple, Any

# from siren.analysis.interface import AbsSymState
from siren.grammar import *
from siren.grammar import Normal, AbsNormal, Const, SymExpr, AbsConst, AbsSymExpr
from siren.compiler.interface import SymState


# Helper functions for manipulating conjugate distributions

# If expr scales rv, returns the scaling factor
# Not complete
def is_scaled(state: SymState, expr: SymExpr, e: SymExpr,abs_expr: AbsSymExpr, abs_e: AbsSymExpr) -> [Expr, AbsSymExpr]:
    if abs_expr == abs_e:
        return Const(1),AbsConst(1)

    match abs_expr:
        case AbsConst(_):
            return None, None
        case AbsRandomVar(_):
            return None, None
        case AbsAdd(abs_e1, abs_e2):
            s1, abs_s1 = is_scaled(state, abs_e1, e, abs_e1, abs_e)
            s2, abs_s2 = is_scaled(state, abs_e2, e, abs_e2, abs_e)
            if ((s1 and abs_s1) is None) or ((s2 and abs_s2) is None):
                return None, None
            else:
                return state.ex_add(s1, s2, abs_s1, abs_s2)
        case AbsMul(abs_e1, abs_e2):
            s1, abs_s1 = is_scaled(state, abs_e1, e, abs_e1, abs_e)
            s2, abs_s2 = is_scaled(state, abs_e2, e, abs_e2, abs_e)
            if ((s1 and abs_s1) is None) and ((s2 and abs_s2) is None):
                return None, None
            elif ((s1 and abs_s1) is None) and ((s2 and abs_s2) is not None):
                return state.ex_mul(s2, e1, abs_s2, abs_e1)
            elif ((s1 and abs_s1) is not None) and ((s2 and abs_s2) is None):
                return state.ex_mul(s1, e2, abs_s1, abs_e2)
            else:
                assert ((s1 and abs_s1) is not None) and ((s2 and abs_s2) is not None)
                return state.ex_mul(s1, s2, abs_s1, abs_s2)
        case AbsDiv(abs_e1, abs_e2):
            s1, abs_s1 = is_scaled(state, abs_e1, e, abs_e1, abs_e)
            s2, abs_s2 = is_scaled(state, abs_e2, e, abs_e2, abs_e)
            if ((s1 and abs_s1) is not None) and ((s2 and abs_s2) is not None):
                return None, None  # e cancels out
            elif ((s1 and abs_s1) is None) and ((s2 and abs_s2) is not None):
                return state.ex_div(e1, s2, abs_e1, abs_s2)
            elif ((s1 and abs_s1) is not None) and ((s2 and abs_s2) is None):
                return state.ex_div(s1, e2, abs_s1, abs_e2)
            else:
                return None, None
        case AbsIte(_):
            return None, None
        case AbsEq(_):
            return None, None
        case AbsLt(_):
            return None, None
        case AbsLst(_):
            return None, None
        case AbsPair(_):
            return None, None
        case UnkE(s):
            return None, None
        case TopE():
            return None, None
        case _:
            raise ValueError(expr)


def abs_is_scaled(state: SymState, expr: AbsSymExpr, e: AbsSymExpr) -> Optional[AbsSymExpr]:
    if expr == e:
        return AbsConst(1)

    match expr:
        case AbsConst(_):
            return None
        case AbsRandomVar(_):
            return None
        case AbsAdd(e1, e2):
            s1 = abs_is_scaled(state, e1, e)
            s2 = abs_is_scaled(state, e2, e)
            if s1 is None or s2 is None:
                return None
            else:
                return state.abs_ex_add(s1, s2)
        case AbsMul(e1, e2):
            s1 = abs_is_scaled(state, e1, e)
            s2 = abs_is_scaled(state, e2, e)
            if s1 is None and s2 is None:
                return None
            elif s1 is None and s2 is not None:
                return state.abs_ex_mul(s2, e1)
            elif s1 is not None and s2 is None:
                return state.abs_ex_mul(s1, e2)
            else:
                assert s1 is not None and s2 is not None
                return state.abs_ex_mul(s1, s2)
        case AbsDiv(e1, e2):
            s1 = abs_is_scaled(state, e1, e)
            s2 = abs_is_scaled(state, e2, e)
            if s1 is not None and s2 is not None:
                return None  # e cancels out
            elif s1 is None and s2 is not None:
                return state.abs_ex_div(e1, s2)
            elif s1 is not None and s2 is None:
                return state.abs_ex_div(s1, e2)
            else:
                return None
        case AbsIte(_):
            return None
        case AbsEq(_):
            return None
        case AbsLt(_):
            return None
        case AbsLst(_):
            return None
        case AbsPair(_):
            return None
        case UnkE(s):
            return None
        case TopE():
            return None
        case _:
            raise ValueError(expr)


# Returns (a, b) such that expr = a * rv + b
def is_affine(state: SymState, expr: SymExpr, rv: RandomVar, abs_expr:AbsSymExpr, abs_rv: AbsRandomVar) -> tuple[
    Expr, Expr, AbsConst, AbsSymExpr]:
    # Distribution parameters are runtime projections, not concrete Add/Mul
    # trees. Use the abstract affine coefficient and substitute the actual
    # parent out of the runtime mean to retain the remaining terms.
    if isinstance(expr, Apply):
        coefficients = abs_is_affine(state, abs_expr, abs_rv)
        if coefficients is None:
            return None
        abs_a, abs_b = coefficients
        match state.Abs_eval(abs_a):
            case AbsConst(value) if not isinstance(value, UnkC):
                b = Apply(Identifier("Prob", "subst_rv"), [expr, rv, Const(0)])
                return Const(value), b, abs_a, abs_b
            case _:
                return None

    match abs_expr:
        case AbsConst(_):
            return (Const(0), expr,AbsConst(0),abs_expr)
        case AbsRandomVar(_):
            if abs_expr == abs_rv:
                return (Const(1), Const(0),AbsConst(1),AbsConst(0))
            else:
                return (Const(0), expr,AbsConst(0),abs_expr)
        case AbsAdd(abs_e1, abs_e2):
            if not isinstance(expr, Add):
                return None
            coefs1 = is_affine(state, expr.left, rv, abs_e1,abs_rv)
            coefs2 = is_affine(state, expr.right, rv, abs_e2,abs_rv)
            if coefs1 is None or coefs2 is None:
                return None
            else:
                a1, b1,abs_a1,abs_b1 = coefs1
                a2, b2,abs_a2,abs_b2 = coefs2
                return state.conc_ex_add(a1, a2), state.conc_ex_add(b1, b2), state.abs_ex_add(abs_a1,abs_a2), state.abs_ex_add(abs_b1,abs_b2)
        case AbsMul(abs_e1, abs_e2):
            if not isinstance(expr, Mul):
                return None
            coefs1 = is_affine(state, expr.left, rv,abs_e1,abs_rv)
            coefs2 = is_affine(state, expr.right, rv,abs_e2,abs_rv)
            if coefs1 is None or coefs2 is None:
                return None
            else:
                a1, b1,abs_a1,abs_b1 = coefs1
                a2, b2,abs_a2,abs_b2 = coefs2
                match state.Abs_eval(abs_a1), state.Abs_eval(abs_a2):
                    case AbsConst(0), AbsConst(0):
                        return Const(0), state.conc_ex_mul(b1, b2),AbsConst(0),state.abs_ex_mul(abs_b1, abs_b2)
                    case abs_a1, AbsConst(0):
                        return state.conc_ex_mul(a1, b2), state.conc_ex_mul(b1, b2),state.abs_ex_mul(abs_a1,abs_b2),state.abs_ex_mul(abs_b1,abs_b2)
                    case AbsConst(0), abs_a2:
                        return state.conc_ex_mul(b1, a2), state.conc_ex_mul(b1, b2), state.abs_ex_mul(abs_b1,abs_a2),state.abs_ex_mul(abs_b1,abs_b2)
                    case _:
                        return None
        case AbsDiv(abs_e1, abs_e2):
            if not isinstance(expr, Div):
                return None
            coefs1 = is_affine(state, expr.left, rv,abs_e1,abs_rv)
            coefs2 = is_affine(state, expr.right, rv,abs_e2,abs_rv)
            if coefs1 is None or coefs2 is None:
                return None
            else:
                a1, b1, abs_a1, abs_b1 = coefs1
                a2, b2, abs_a2, abs_b2 = coefs2
                match state.Abs_eval(abs_a2):
                    case AbsConst(0):
                        return state.conc_ex_div(a1,b2),state.conc_ex_div(b1,b2),state.abs_ex_div(abs_a1, abs_b2), state.abs_ex_div(abs_b1, abs_b2)
                    case _:
                        return None
        case AbsIte(_):
            return None,None
        case AbsEq(_):
            return None
        case AbsLt(_):
            return None
        case AbsLst(_):
            return None
        case AbsPair(_):
            return None
        case UnkE(s):
            if len(s) ==0:
                return ("SWAP","SWAP",AbsConst(0),abs_expr)

            else:
                return None,None,None,None
        case TopE():
            return None,None,None,None
        case _:
            raise ValueError(expr)



def conc_is_affine(state: SymState, expr: SymExpr, rv: RandomVar) -> Optional[Tuple[SymExpr, SymExpr]]:
    match expr:
        case Const(_):
            return Const(0), expr
        case RandomVar(_):
            if expr == rv:
                return Const(1), Const(0)
            else:
                return Const(0), expr
        case Add(e1, e2):
            coefs1 = conc_is_affine(state, e1, rv)
            coefs2 = conc_is_affine(state, e2, rv)
            if coefs1 is None or coefs2 is None:
                return None
            else:
                a1, b1 = coefs1
                a2, b2 = coefs2
                return state.conc_ex_add(a1, a2), state.conc_ex_add(b1, b2)
        case Mul(e1, e2):
            coefs1 = conc_is_affine(state, e1, rv)
            coefs2 = conc_is_affine(state, e2, rv)
            if coefs1 is None or coefs2 is None:
                return None
            else:
                a1, b1 = coefs1
                a2, b2 = coefs2
                match state.conc_eval(a1), state.conc_eval(a2):
                    case Const(0), Const(0):
                        return Const(0), state.conc_ex_mul(b1, b2)
                    case a1, Const(0):
                        return state.conc_ex_mul(a1, b2), state.conc_ex_mul(b1, b2)
                    case Const(0), a2:
                        return state.conc_ex_mul(b1, a2), state.conc_ex_mul(b1, b2)
                    case _:
                        return None
        case Div(e1, e2):
            coefs1 = conc_is_affine(state, e1, rv)
            coefs2 = conc_is_affine(state, e2, rv)
            if coefs1 is None or coefs2 is None:
                return None
            else:
                a1, b1 = coefs1
                a2, b2 = coefs2
                match state.conc_eval(a2):
                    case Const(0):
                        return state.conc_ex_div(a1, b2), state.conc_ex_div(b1, b2)
                    case _:
                        return None
        case Ite(_):
            return None
        case Eq(_):
            return None
        case Lt(_):
            return None
        case Lst(_):
            return None
        case Pair(_):
            return None
        case _:
            raise ValueError(expr)

def abs_is_affine(state: SymState, expr: AbsSymExpr, rv: AbsRandomVar) -> Optional[Tuple[AbsSymExpr, AbsSymExpr]]:
  match expr:
    case AbsConst(_):
      return AbsConst(0), expr
    case AbsRandomVar(_):
      if expr == rv:
        return AbsConst(1), AbsConst(0)
      else:
        return AbsConst(0), expr
    case AbsAdd(e1, e2):
      coefs1 = abs_is_affine(state, e1, rv)
      coefs2 = abs_is_affine(state, e2, rv)
      if coefs1 is None or coefs2 is None:
        return None
      else:
        a1, b1 = coefs1
        a2, b2 = coefs2
        return state.abs_ex_add(a1, a2), state.abs_ex_add(b1, b2)
    case AbsMul(e1, e2):
      coefs1 = abs_is_affine(state, e1, rv)
      coefs2 = abs_is_affine(state, e2, rv)
      if coefs1 is None or coefs2 is None:
        return None
      else:
        a1, b1 = coefs1
        a2, b2 = coefs2
        match state.Abs_eval(a1), state.Abs_eval(a2): #changes done here in the eval
          case AbsConst(0), AbsConst(0):
            return AbsConst(0), state.abs_ex_mul(b1, b2)
          case a1, AbsConst(0):
            return state.abs_ex_mul(a1, b2), state.abs_ex_mul(b1, b2)
          case AbsConst(0), a2:
            return state.abs_ex_mul(b1, a2), state.abs_ex_mul(b1, b2)
          case _:
            return None
    case AbsDiv(e1, e2):
      coefs1 = abs_is_affine(state, e1, rv)
      coefs2 = abs_is_affine(state, e2, rv)
      if coefs1 is None or coefs2 is None:
        return None
      else:
        a1, b1 = coefs1
        a2, b2 = coefs2
        match state.Abs_eval(a2):# changes done here for the eval
          case AbsConst(0):
            return state.abs_ex_div(a1, b2), state.abs_ex_div(b1, b2)
          case _:
            return None
    case AbsIte(_):
      return None
    case AbsEq(_):
      return None
    case AbsLt(_):
      return None
    case AbsLst(_):
      return None
    case AbsPair(_):
      return None
    case UnkE(s):
      # Doesn't depend on rv so it's affine
      if len(s) == 0:
        return AbsConst(0), expr
      else:
        return None
    case TopE():
      return None
    case _:
      raise ValueError(expr)

### Affine Gaussian Conjugate ###

# def gaussian_conjugate_check(state: SymState, prior: SymDistr, likelihood: SymDistr,
#                              rv_par: RandomVar, rv_child: RandomVar) -> bool:
#     match prior, likelihood:
#         case Normal(mu0, var0), Normal(mu, var):
#             coefs = is_affine(state, mu, rv_par)
#             if coefs is None:
#                 return False
#
#             return not mu0.depends_on(rv_child, True) \
#                 and not var0.depends_on(rv_child, True) \
#                 and not var.depends_on(rv_par, True)
#         case _:
#             return False
#
#
# def gaussian_marginal(state: SymState, prior: Normal, likelihood: Normal,
#                       rv_par: RandomVar, rv_child: RandomVar) -> Optional[Normal]:
#     if not gaussian_conjugate_check(state, prior, likelihood, rv_par, rv_child):
#         return None
#
#     mu0, var0 = prior.mu, prior.var
#     mu, var = likelihood.mu, likelihood.var
#
#     coefs = is_affine(state, mu, rv_par)
#     if coefs is None:
#         return None
#
#     a, b = coefs
#
#     mu01 = state.ex_add(state.ex_mul(a, mu0), b)
#     var01 = state.ex_mul(state.ex_mul(a, a), var0)
#
#     mu0_new = mu01
#     var0_new = state.ex_add(var01, var)
#
#     return Normal(mu0_new, var0_new)
#
#
# def gaussian_posterior(state: SymState, prior: Normal, likelihood: Normal,
#                        rv_par: RandomVar, rv_child: RandomVar, obs: Optional[SymExpr] = None) -> Optional[Normal]:
#     if not gaussian_conjugate_check(state, prior, likelihood, rv_par, rv_child):
#         return None
#
#     mu0, var0 = prior.mu, prior.var
#     mu, var = likelihood.mu, likelihood.var
#
#     coefs = is_affine(state, mu, rv_par)
#     if coefs is None:
#         return None
#
#     a, b = coefs
#
#     x = rv_child if obs is None else obs
#
#     mu01 = state.ex_add(state.ex_mul(a, mu0), b)
#     var01 = state.ex_mul(state.ex_mul(a, a), var0)
#
#     denom = state.ex_add(state.ex_div(Const(1), var01), state.ex_div(Const(1), var))
#     var02 = state.ex_div(Const(1), denom)
#
#     sum1 = state.ex_add(state.ex_div(mu01, var01), state.ex_div(x, var))
#     mu02 = state.ex_mul(sum1, var02)
#
#     mu1_new = state.ex_div(state.ex_add(mu02, state.ex_mul(Const(-1), b)), a)
#     var1_new = state.ex_div(var02, state.ex_mul(a, a))
#
#     return Normal(mu1_new, var1_new)
#
#
# # Returns (marginal, posterior) distributions
# def gaussian_conjugate(state: SymState, rv_par: RandomVar, rv_child: RandomVar) -> Optional[Tuple[Normal, Normal]]:
#     prior, likelihood = state.get_entry(rv_par, 'distribution'), state.get_entry(rv_child, 'distribution')
#     match prior, likelihood:
#         case Normal(mu0, var0), Normal(mu, var):
#             marginal = gaussian_marginal(state, prior, likelihood, rv_par, rv_child)
#             posterior = gaussian_posterior(state, prior, likelihood, rv_par, rv_child)
#             if marginal is None or posterior is None:
#                 return None
#             return (marginal, posterior)
#         case _:
#             return None

def gaussian_conjugate_check(state: SymState, prior: SymDistr, likelihood: SymDistr,
                             rv_par: RandomVar, rv_child: RandomVar, abs_prior: AbsSymDistr, abs_likelihood: AbsSymDistr, abs_rv_par: AbsRandomVar, abs_rv_child:AbsRandomVar) -> bool:
  match abs_prior, abs_likelihood:
    case AbsNormal(mu0, var0), AbsNormal(mu, var):
      coefs = is_affine(state, likelihood.mu, rv_par,mu,abs_rv_par)
      if coefs is None:
        return False

      return not mu0.depends_on(abs_rv_child, True) \
            and not var0.depends_on(abs_rv_child, True) \
            and not var.depends_on(abs_rv_par, True)
    case _:
      return False

def abs_gaussian_conjugate_check(state: SymState, prior: AbsSymDistr, likelihood: AbsSymDistr,
                             rv_par: AbsRandomVar, rv_child: AbsRandomVar) -> bool:
  match prior, likelihood:
    case AbsNormal(mu0, var0), AbsNormal(mu, var):
      coefs = abs_is_affine(state, mu, rv_par)
      if coefs is None:
        return False

      return not mu0.depends_on(rv_child, True) \
            and not var0.depends_on(rv_child, True) \
            and not var.depends_on(rv_par, True)
    case _:
      return False


def gaussian_marginal(state: SymState, prior: Normal, likelihood: Normal,
                      rv_par: RandomVar, rv_child: RandomVar,abs_prior: AbsNormal, abs_likelihood: AbsNormal,
                      abs_rv_par: AbsRandomVar, abs_rv_child: AbsRandomVar) -> tuple[Normal, AbsNormal] | None:

    if not abs_gaussian_conjugate_check(state, abs_prior, abs_likelihood, abs_rv_par, abs_rv_child):
        return None

    mu0 = Apply(Identifier("Prob", "get_mu"), [prior])
    var0 = Apply(Identifier("Prob", "get_var"), [prior])
    mu = Apply(Identifier("Prob", "get_mu"), [likelihood])
    var = Apply(Identifier("Prob", "get_var"), [likelihood])
    abs_mu0, abs_var0 = abs_prior.mu, abs_prior.var
    abs_mu, abs_var = abs_likelihood.mu, abs_likelihood.var

    coefs = is_affine(state, mu, rv_par,abs_mu,abs_rv_par)
    if coefs is None:
        return None

    a, b, abs_a, abs_b = coefs
    val_mul,abs_val_mul = state.ex_mul(a, mu0,abs_a,abs_mu0)
    mu01,abs_mu01 = state.ex_add(val_mul, b,abs_val_mul,abs_b)
    value_mul, abs_value_mul = state.ex_mul(a, a,abs_a,abs_a)
    var01,abs_var01 = state.ex_mul(value_mul, var0,abs_value_mul,abs_var0)

    mu0_new = mu01
    abs_mu0_new = abs_mu01
    var0_new,abs_var0_new = state.ex_add(var01, var,abs_var01,abs_var)


    return Normal(mu0_new, var0_new), AbsNormal(abs_mu0_new, abs_var0_new)


def abs_gaussian_marginal(state: SymState, prior: AbsNormal, likelihood: AbsNormal,
                      rv_par: AbsRandomVar, rv_child: AbsRandomVar) -> Optional[AbsNormal]:
    if not abs_gaussian_conjugate_check(state, prior, likelihood, rv_par, rv_child):
        return None

    mu0, var0 = prior.mu, prior.var
    mu, var = likelihood.mu, likelihood.var

    coefs = abs_is_affine(state, mu, rv_par)
    if coefs is None:
        return None

    a, b = coefs

    mu01 = state.abs_ex_add(state.abs_ex_mul(a, mu0), b)
    var01 = state.abs_ex_mul(state.abs_ex_mul(a, a), var0)

    mu0_new = mu01
    var0_new = state.abs_ex_add(var01, var)

    return AbsNormal(mu0_new, var0_new)


def gaussian_posterior(state: SymState, prior: Normal, likelihood: Normal,
                       rv_par: RandomVar, rv_child: RandomVar,abs_prior: AbsNormal, abs_likelihood: AbsNormal,
                       abs_rv_par: AbsRandomVar, abs_rv_child: AbsRandomVar, obs: Optional[SymExpr] = None, abs_obs: Optional[AbsSymExpr] = None) -> \
tuple[Normal, AbsNormal] | None:

    if not abs_gaussian_conjugate_check(state, abs_prior, abs_likelihood, abs_rv_par, abs_rv_child):
        return None

    mu0 = Apply(Identifier("Prob", "get_mu"), [prior])
    var0 = Apply(Identifier("Prob", "get_var"), [prior])
    mu = Apply(Identifier("Prob", "get_mu"), [likelihood])
    var = Apply(Identifier("Prob", "get_var"), [likelihood])
    abs_mu0, abs_var0 = abs_prior.mu, abs_prior.var
    abs_mu, abs_var = abs_likelihood.mu, abs_likelihood.var

    coefs = is_affine(state, mu, rv_par,abs_mu,abs_rv_par)
    if coefs is None:
        return None

    a, b, abs_a, abs_b = coefs

    x = rv_child if obs is None else obs
    abs_x = abs_rv_child if abs_obs is None else abs_obs

    val_mul, abs_val_mul = state.ex_mul(a, mu0,abs_a,abs_mu0)
    mu01, abs_mu01 = state.ex_add(val_mul, b, abs_val_mul, abs_b)
    value_mul, abs_value_mul = state.ex_mul(a, a,abs_a,abs_a)
    var01, abs_var01 = state.ex_mul(value_mul, var0, abs_value_mul, abs_var0)

    val_div_var01, abs_val_div_var01 = state.ex_div(Const(1), var01, AbsConst(1),abs_var01)
    val_div_var, abs_val_div_var = state.ex_div(Const(1), var, AbsConst(1),abs_var)
    denom, abs_denom = state.ex_add(val_div_var01, val_div_var, abs_val_div_var01, abs_val_div_var)
    var02, abs_var02 = state.ex_div(Const(1), denom, AbsConst(1),abs_denom)

    val_div_mu01, abs_val_div_mu01 = state.ex_div(mu01, var01,abs_mu01,abs_var01)
    val_div_x, abs_val_div_x = state.ex_div(x, var,abs_x,abs_var)
    sum1, abs_sum1 = state.ex_add(val_div_mu01, val_div_x, abs_val_div_mu01,abs_val_div_x)
    mu02, abs_mu02 = state.ex_mul(sum1, var02,abs_sum1,abs_var02)



    val_mul_b, abs_val_mul_b = state.ex_mul(Const(-1), b,AbsConst(-1),abs_b)
    val_add_mu02, abs_val_add_mu02 = state.ex_add(mu02, val_mul_b, abs_mu02, abs_val_mul_b)
    mu1_new, abs_mu1_new = state.ex_div(val_add_mu02, a, abs_val_add_mu02, abs_a)
    val_mul_a, abs_val_mul_a = state.ex_mul(a, a, abs_a, abs_a)
    var1_new, abs_var1_new = state.ex_div(var02, val_mul_a, abs_var02,abs_val_mul_a)

    # print("x", x)
    # print("mu01", mu01)
    # print("var01", var01)
    # print("denom",denom)
    # print("var02",var02)
    # print("sum1",sum1)
    # print("abs_sum1",abs_sum1)
    # print("mu02", mu02)
    # print("mu1_new", mu1_new)
    # print("var1_new", var1_new)
    # print("val_mul", val_mul)
    # print("abs_val_mul", abs_val_mul)
    # print("abs_val_div_var", abs_val_div_var)
    # print("val_div_var",val_div_var)

    #
    # print("normal", Normal(mu1_new, var1_new))
    return Normal(mu1_new, var1_new), AbsNormal(abs_mu1_new, abs_var1_new)


def abs_gaussian_posterior(state: SymState, prior: AbsNormal, likelihood: AbsNormal,
                       rv_par: AbsRandomVar, rv_child: AbsRandomVar, obs: Optional[AbsSymExpr] = None) -> Optional[
    AbsNormal]:
    if not abs_gaussian_conjugate_check(state, prior, likelihood, rv_par, rv_child):
        return None

    mu0, var0 = prior.mu, prior.var
    mu, var = likelihood.mu, likelihood.var

    coefs = abs_is_affine(state, mu, rv_par)
    if coefs is None:
        return None

    a, b = coefs

    x = rv_child if obs is None else obs

    mu01 = state.abs_ex_add(state.abs_ex_mul(a, mu0), b)
    var01 = state.abs_ex_mul(state.abs_ex_mul(a, a), var0)

    denom = state.abs_ex_add(state.abs_ex_div(AbsConst(1), var01), state.abs_ex_div(AbsConst(1), var))
    var02 = state.abs_ex_div(AbsConst(1), denom)

    sum1 = state.abs_ex_add(state.abs_ex_div(mu01, var01), state.abs_ex_div(x, var))
    mu02 = state.abs_ex_mul(sum1, var02)

    mu1_new = state.abs_ex_div(state.abs_ex_add(mu02, state.abs_ex_mul(AbsConst(-1), b)), a)
    var1_new = state.abs_ex_div(var02, state.abs_ex_mul(a, a))

    return AbsNormal(mu1_new, var1_new)


def gaussian_conjugate(state: SymState, rv_par: RandomVar, rv_child: RandomVar,abs_rv_par:AbsRandomVar, abs_rv_child:AbsRandomVar) -> \
tuple[Expr, Expr, AbsSymExpr, AbsSymExpr] | None:
  prior, likelihood = state.distr(rv_par), state.distr(rv_child)

  abs_prior, abs_likelihood = state.Abs_distr(abs_rv_par), state.Abs_distr(abs_rv_child)
  match abs_prior, abs_likelihood:
    case AbsNormal(mu0, var0), AbsNormal(mu, var):
      marginal,abs_marginal = gaussian_marginal(state, prior, likelihood, rv_par, rv_child,abs_prior, abs_likelihood, abs_rv_par, abs_rv_child)
      posterior,abs_posterior = gaussian_posterior(state, prior, likelihood, rv_par, rv_child,abs_prior, abs_likelihood, abs_rv_par, abs_rv_child)
      if marginal is None or posterior is None:
        return None
      return marginal, posterior, abs_marginal,abs_posterior
    case _:
      return None

# Returns (marginal, posterior) distributions
def abs_gaussian_conjugate(state: SymState, rv_par: AbsRandomVar, rv_child: AbsRandomVar) -> Optional[
    Tuple[AbsNormal, AbsNormal]]:
    prior, likelihood = state.Abs_distr(rv_par), state.Abs_distr(rv_child)
    match prior, likelihood:
        case AbsNormal(mu0, var0), AbsNormal(mu, var):
            marginal = abs_gaussian_marginal(state, prior, likelihood, rv_par, rv_child)
            posterior = abs_gaussian_posterior(state, prior, likelihood, rv_par, rv_child)
            if marginal is None or posterior is None:
                return None
            return marginal, posterior
        case _:
            return None


### Bernoulli Conjugate ###

def bernoulli_conjugate_check(state: SymState, prior: SymDistr, likelihood: SymDistr,
                              rv_par: RandomVar, rv_child: RandomVar, abs_prior: AbsSymDistr, abs_likelihood: AbsSymDistr,
                              abs_rv_par: AbsRandomVar, abs_rv_child: AbsRandomVar) -> bool:
    match abs_prior, abs_likelihood:
        case AbsBernoulli(p1), AbsBernoulli(p2):
            return p2.depends_on(abs_rv_par, False) and \
                not p1.depends_on(abs_rv_child, True)
        case _:
            return False

def abs_bernoulli_conjugate_check(state: SymState, prior: AbsSymDistr, likelihood: AbsSymDistr,
                              rv_par: AbsRandomVar, rv_child: AbsRandomVar) -> bool:
  match prior, likelihood:
    case AbsBernoulli(p1), AbsBernoulli(p2):
      return p2.depends_on(rv_par, False) and \
            not p1.depends_on(rv_child, True)
    case _:
      return False


def bernoulli_marginal(state: SymState, prior: Bernoulli, likelihood: Bernoulli,
                       rv_par: RandomVar, rv_child: RandomVar, abs_prior: AbsBernoulli, abs_likelihood: AbsBernoulli,
                       abs_rv_par: AbsRandomVar, abs_rv_child: AbsRandomVar) -> tuple[Bernoulli, AbsBernoulli] | None:
    if not bernoulli_conjugate_check(state, prior, likelihood, rv_par, rv_child, abs_prior, abs_likelihood, abs_rv_par, abs_rv_child):
        return None

    p1 = Apply(Identifier("Prob", "get_p"),[prior])
    p2 = Apply(Identifier("Prob", "get_p"),[likelihood])
    abs_p1, abs_p2 = abs_prior.p, abs_likelihood.p

    mul_val, abs_mul_val = state.ex_mul(GenericOp(Operator.sub,[Const(0), Const(1)]), p1, AbsConst(-1), abs_p1)
    add_val, abs_add_val = state.ex_add(Const(1), mul_val, AbsConst(1), abs_mul_val)
    print("value of the p2", p2)
    print("value of the abs_p2", abs_p2)
    print("value of the abs_p2 subset", abs_p2.subst_rv(abs_rv_par, AbsConst(True)))
    mul_val1, abs_mul_val1 = state.ex_mul(p1, Apply(Identifier("Prob","subst_rv"),[p2,rv_par,Const(True)]), abs_p1, abs_p2.subst_rv(abs_rv_par, AbsConst(True)))
    mul_val2, abs_mul_val2 = state.ex_mul(add_val, Apply(Identifier("Prob","subst_rv"),[p2,rv_par,Const(False)]), abs_add_val, abs_p2.subst_rv(abs_rv_par, AbsConst(False)))
    p2_new, abs_p2_new = state.ex_add(mul_val1, mul_val2, abs_mul_val1, abs_mul_val2)
    return Bernoulli(p2_new), AbsBernoulli(abs_p2_new) 


def abs_bernoulli_marginal(state: SymState, prior: AbsBernoulli, likelihood: AbsBernoulli,
                       rv_par: AbsRandomVar, rv_child: AbsRandomVar) -> Optional[AbsBernoulli]:
    if not abs_bernoulli_conjugate_check(state, prior, likelihood, rv_par, rv_child):
        return None

    p1, p2 = prior.p, likelihood.p
    p2_new = state.abs_ex_add(state.abs_ex_mul(p1, p2.subst_rv(rv_par, AbsConst(True))),
                          state.abs_ex_mul(state.abs_ex_add(AbsConst(1), state.abs_ex_mul(AbsConst(-1), p1)),
                                       p2.subst_rv(rv_par, AbsConst(False))))
    return AbsBernoulli(p2_new)


def bernoulli_posterior(state: SymState, prior: Bernoulli, likelihood: Bernoulli,
                        rv_par: RandomVar, rv_child: RandomVar, abs_prior: AbsBernoulli, abs_likelihood: AbsBernoulli,
                        abs_rv_par: AbsRandomVar, abs_rv_child: AbsRandomVar, obs: Optional[SymExpr] = None,abs_obs: Optional[AbsSymExpr] = None) -> tuple[Bernoulli, AbsBernoulli] | None:
    if not bernoulli_conjugate_check(state, prior, likelihood, rv_par, rv_child, abs_prior, abs_likelihood, abs_rv_par, abs_rv_child):
        return None

    x = rv_child if obs is None else obs
    abs_x = abs_rv_child if abs_obs is None else abs_obs

    p1 = Apply(Identifier("Prob", "get_p"),[prior])
    p2 = Apply(Identifier("Prob", "get_p"),[likelihood])
    abs_p1, abs_p2 = abs_prior.p, abs_likelihood.p

    mul_val, abs_mul_val = state.ex_mul(GenericOp(Operator.sub,[Const(0), Const(1)]), p1, AbsConst(-1), abs_p1)
    add_val, abs_add_val = state.ex_add(Const(1), mul_val, AbsConst(1), abs_mul_val)
    mul_val1, abs_mul_val1 = state.ex_mul(add_val, Apply(Identifier("Prob","subst_rv"),[p2,rv_par,Const(False)]), abs_add_val, abs_p2.subst_rv(abs_rv_par, AbsConst(False)))
    mul_val2, abs_mul_val2 = state.ex_mul(p1, Apply(Identifier("Prob","subst_rv"),[p2,rv_par,Const(True)]), abs_p1, abs_p2.subst_rv(abs_rv_par, AbsConst(True)))

    p2_new, abs_p2_new = state.ex_add(mul_val2, mul_val1, abs_mul_val2, abs_mul_val1)

    p1_mul_val, abs_p1_mul_val = state.ex_mul(GenericOp(Operator.sub,[Const(0), Const(1)]), p2, AbsConst(-1), abs_p2)
    p1_add_val, abs_p1_add_val = state.ex_add(Const(1), p1_mul_val, AbsConst(1), abs_p1_mul_val)
    p1_num_sub, abs_p1_num_sub = state.ex_ite(x, p2, p1_add_val, abs_x, abs_p2, abs_p1_add_val)
    p1_num, abs_p1_num = state.ex_mul(p1, Apply(Identifier("Prob","subst_rv"),[p1_num_sub,rv_par,Const(True)]),abs_p1, abs_p1_num_sub.subst_rv(abs_rv_par, AbsConst(True)))
    p1_denom_mul_val, abs_p1_denom_mul_val = state.ex_mul(GenericOp(Operator.sub,[Const(0), Const(1)]), p2_new, AbsConst(-1), abs_p2_new)
    p1_denom_add_val, abs_p1_denom_add_val = state.ex_add(Const(1), p1_denom_mul_val, AbsConst(1), abs_p1_denom_mul_val)
    p1_denom, abs_p1_denom = state.ex_ite(x, p2_new, p1_denom_add_val, abs_x, abs_p2_new, abs_p1_denom_add_val)
    p1_new, abs_p1_new = state.ex_div(p1_num, p1_denom, abs_p1_num, abs_p1_denom)

    return Bernoulli(p1_new), AbsBernoulli(abs_p1_new)


def abs_bernoulli_posterior(state: SymState, prior: AbsBernoulli, likelihood: AbsBernoulli,
                        rv_par: AbsRandomVar, rv_child: AbsRandomVar, obs: Optional[AbsSymExpr] = None) -> Optional[
    AbsBernoulli]:
    if not abs_bernoulli_conjugate_check(state, prior, likelihood, rv_par, rv_child):
        return None

    x = rv_child if obs is None else obs

    p1, p2 = prior.p, likelihood.p
    p2_new = state.abs_ex_add(state.abs_ex_mul(p1, p2.subst_rv(rv_par, AbsConst(True))),
                          state.abs_ex_mul(state.abs_ex_add(AbsConst(1), state.abs_ex_mul(AbsConst(-1), p1)),
                                       p2.subst_rv(rv_par, AbsConst(False))))

    p1_num_sub = state.abs_ex_ite(x, p2, state.abs_ex_add(AbsConst(1), state.abs_ex_mul(AbsConst(-1), p2)))
    p1_num = state.abs_ex_mul(p1, p1_num_sub.subst_rv(rv_par, AbsConst(True)))
    p1_denom = state.abs_ex_ite(x, p2_new, state.abs_ex_add(AbsConst(1), state.abs_ex_mul(AbsConst(-1), p2_new)))
    p1_new = state.abs_ex_div(p1_num, p1_denom)

    return AbsBernoulli(p1_new)


def bernoulli_conjugate(state: SymState, rv_par: RandomVar, rv_child: RandomVar,abs_rv_par: AbsRandomVar, abs_rv_child: AbsRandomVar) -> tuple[Expr, Expr, AbsSymExpr, AbsSymExpr] | None:
    prior, likelihood = state.distr(rv_par), state.distr(rv_child)
    abs_prior, abs_likelihood = state.Abs_distr(abs_rv_par), state.Abs_distr(abs_rv_child)
    match abs_prior, abs_likelihood:
        case AbsBernoulli(p1), AbsBernoulli(p2):
            marginal, abs_marginal = bernoulli_marginal(state, prior, likelihood, rv_par, rv_child, abs_prior, abs_likelihood, abs_rv_par, abs_rv_child)
            posterior, abs_posterior = bernoulli_posterior(state, prior, likelihood, rv_par, rv_child, abs_prior, abs_likelihood, abs_rv_par, abs_rv_child)
            if marginal is None or posterior is None:
                return None
            return (marginal, posterior, abs_marginal, abs_posterior)
        case _:
            return None

def abs_bernoulli_conjugate(state: SymState, rv_par: AbsRandomVar, rv_child: AbsRandomVar) -> Optional[Tuple[AbsBernoulli, AbsBernoulli]]:
  prior, likelihood = state.Abs_distr(rv_par), state.Abs_distr(rv_child)
  match prior, likelihood:
    case AbsBernoulli(p1), AbsBernoulli(p2):
      marginal = abs_bernoulli_marginal(state, prior, likelihood, rv_par, rv_child)
      posterior = abs_bernoulli_posterior(state, prior, likelihood, rv_par, rv_child)
      if marginal is None or posterior is None:
        return None
      return (marginal, posterior)
    case _:
      return None


### Beta Bernoulli Conjugate ###

def beta_bernoulli_conjugate_check(state: SymState, prior: SymDistr, likelihood: SymDistr,
                                   rv_par: RandomVar, rv_child: RandomVar, abs_prior: AbsSymDistr, abs_likelihood: AbsSymDistr,
                                    abs_rv_par: AbsRandomVar, abs_rv_child: AbsRandomVar) -> bool:
    match abs_prior, abs_likelihood:
        case AbsBeta(abs_a, abs_b), AbsBernoulli(abs_p):
            return abs_rv_par == abs_p \
                and not abs_a.depends_on(abs_rv_child, True) \
                and not abs_b.depends_on(abs_rv_child, True)
        case _:
            return False

def abs_beta_bernoulli_conjugate_check(state: SymState, prior: AbsSymDistr, likelihood: AbsSymDistr,
                                    rv_par: AbsRandomVar, rv_child: AbsRandomVar) -> bool:
  match prior, likelihood:
    case AbsBeta(a, b), AbsBernoulli(p):
      return rv_par == p \
            and not a.depends_on(rv_child, True) \
            and not b.depends_on(rv_child, True)
    case _:
      return False


def beta_bernoulli_marginal(state: SymState, prior: Beta, likelihood: Bernoulli,
                            rv_par: RandomVar, rv_child: RandomVar, abs_prior: AbsBeta, abs_likelihood: AbsBernoulli,
                            abs_rv_par: AbsRandomVar, abs_rv_child: AbsRandomVar) -> tuple[Bernoulli, AbsBernoulli] | None:
    if not beta_bernoulli_conjugate_check(state, prior, likelihood, rv_par, rv_child, abs_prior, abs_likelihood, abs_rv_par, abs_rv_child):
        return None

    a = Apply(Identifier("Prob", "get_a"), [prior])
    b = Apply(Identifier("Prob", "get_b"), [prior])
    abs_a, abs_b = abs_prior.a, abs_prior.b
    val_add, abs_val_add = state.ex_add(a, b, abs_a, abs_b)
    p_new, abs_p_new = state.ex_div(a,val_add, abs_a, abs_val_add)

    return Bernoulli(p_new), AbsBernoulli(abs_p_new)


def abs_beta_bernoulli_marginal(state: SymState, prior: AbsBeta, likelihood: AbsBernoulli,
                            rv_par: AbsRandomVar, rv_child: AbsRandomVar) -> Optional[AbsBernoulli]:
    if not abs_beta_bernoulli_conjugate_check(state, prior, likelihood, rv_par, rv_child):
        return None

    a, b = prior.a, prior.b
    p_new = state.abs_ex_div(a, state.abs_ex_add(a, b))

    return AbsBernoulli(p_new)


def beta_bernoulli_posterior(state: SymState, prior: Beta, likelihood: Bernoulli,
                             rv_par: RandomVar, rv_child: RandomVar, abs_prior: AbsBeta, abs_likelihood: AbsBernoulli,
                             abs_rv_par: AbsRandomVar, abs_rv_child: AbsRandomVar, obs: Optional[SymExpr] = None, abs_obs: Optional[AbsSymExpr] = None) -> tuple[Beta, AbsBeta] | None:
    if not beta_bernoulli_conjugate_check(state, prior, likelihood, rv_par, rv_child, abs_prior, abs_likelihood, abs_rv_par, abs_rv_child):
        return None

    a = Apply(Identifier("Prob", "get_a"), [prior])
    b = Apply(Identifier("Prob", "get_b"), [prior])
    abs_a, abs_b = abs_prior.a, abs_prior.b

    x = rv_child if obs is None else obs
    abs_x = abs_rv_child if abs_obs is None else abs_obs

    a_val_ite, abs_a_val_ite = state.ex_ite(x, Const(1), Const(0), abs_x, AbsConst(1), AbsConst(0))
    b_val_ite, abs_b_val_ite = state.ex_ite(x, Const(0), Const(1), abs_x, AbsConst(0), AbsConst(1))
    a_new, abs_a_new = state.ex_add(a, a_val_ite, abs_a, abs_a_val_ite)
    b_new, abs_b_new = state.ex_add(b, b_val_ite, abs_b, abs_b_val_ite)

    return Beta(a_new, b_new), AbsBeta(abs_a_new, abs_b_new)


def abs_beta_bernoulli_posterior(state: SymState, prior: AbsBeta, likelihood: AbsBernoulli,
                             rv_par: AbsRandomVar, rv_child: AbsRandomVar, obs: Optional[AbsSymExpr] = None) -> \
Optional[AbsBeta]:
    if not abs_beta_bernoulli_conjugate_check(state, prior, likelihood, rv_par, rv_child):
        return None

    a, b = prior.a, prior.b

    x = rv_child if obs is None else obs

    a_new = state.abs_ex_add(a, state.abs_ex_ite(x, AbsConst(1), AbsConst(0)))
    b_new = state.abs_ex_add(b, state.abs_ex_ite(x, AbsConst(0), AbsConst(1)))

    return AbsBeta(a_new, b_new)


def beta_bernoulli_conjugate(state: SymState, rv_par: RandomVar, rv_child: RandomVar, abs_rv_par: AbsRandomVar, abs_rv_child: AbsRandomVar) -> tuple[Bernoulli, Beta, AbsBernoulli, AbsBeta] | None:
    prior, likelihood = state.distr(rv_par), state.distr(rv_child)
    abs_prior, abs_likelihood = state.Abs_distr(abs_rv_par), state.Abs_distr(abs_rv_child)
    match abs_prior, abs_likelihood:
        case AbsBeta(a, b), AbsBernoulli(p):
            marginal, abs_marginal = beta_bernoulli_marginal(state, prior, likelihood, rv_par, rv_child, abs_prior, abs_likelihood, abs_rv_par, abs_rv_child)
            posterior, abs_posterior = beta_bernoulli_posterior(state, prior, likelihood, rv_par, rv_child, abs_prior, abs_likelihood, abs_rv_par, abs_rv_child)
            if marginal is None or posterior is None:
                return None
            return marginal, posterior, abs_marginal, abs_posterior
        case _:
            return None

def abs_beta_bernoulli_conjugate(state: SymState, rv_par: AbsRandomVar, rv_child: AbsRandomVar) -> Optional[Tuple[AbsBernoulli, AbsBeta]]:
  prior, likelihood = state.Abs_distr(rv_par), state.Abs_distr(rv_child)
  match prior, likelihood:
    case AbsBeta(a, b), AbsBernoulli(p):
      marginal = abs_beta_bernoulli_marginal(state, prior, likelihood, rv_par, rv_child)
      posterior = abs_beta_bernoulli_posterior(state, prior, likelihood, rv_par, rv_child)
      if marginal is None or posterior is None:
        return None
      return (marginal, posterior)
    case _:
      return None


### Beta Binomial Conjugate ###

def beta_binomial_conjugate_check(state: SymState, prior: SymDistr, likelihood: SymDistr,
                                  rv_par: RandomVar, rv_child: RandomVar, abs_prior: AbsSymDistr, abs_likelihood: AbsSymDistr,
                                    abs_rv_par: AbsRandomVar, abs_rv_child: AbsRandomVar) -> bool:
    match abs_prior, abs_likelihood:
        case AbsBeta(a, b), AbsBinomial(n, p):
            return isinstance(n, Const) \
                and isinstance(a, Const) \
                and abs_rv_par == p \
                and not a.depends_on(abs_rv_child, True) \
                and not b.depends_on(abs_rv_child, True)
        case _:
            return False

def abs_beta_binomial_conjugate_check(state: SymState, prior: AbsSymDistr, likelihood: AbsSymDistr,
                                    rv_par: AbsRandomVar, rv_child: AbsRandomVar) -> bool:
  match prior, likelihood:
    case AbsBeta(a, b), AbsBinomial(n, p):
      return isinstance(n, AbsConst) \
            and isinstance(a, AbsConst)\
            and rv_par == p \
            and not a.depends_on(rv_child, True) \
            and not b.depends_on(rv_child, True)
    case _:
      return False


def beta_binomial_marginal(state: SymState, prior: Beta, likelihood: Binomial,
                           rv_par: RandomVar, rv_child: RandomVar, abs_prior: AbsBeta, abs_likelihood: AbsBinomial,
                           abs_rv_par: AbsRandomVar, abs_rv_child: AbsRandomVar) -> tuple[BetaBinomial, AbsBetaBinomial] | None:
    if not beta_binomial_conjugate_check(state, prior, likelihood, rv_par, rv_child, abs_prior, abs_likelihood, abs_rv_par, abs_rv_child):
        return None

    a, b = prior.a, prior.b
    abs_a, abs_b = abs_prior.a, abs_prior.b
    n, p = likelihood.n, likelihood.p
    abs_n, abs_p = abs_likelihood.n, abs_likelihood.p

    return BetaBinomial(n, a, b), AbsBetaBinomial(abs_n, abs_a, abs_b)


def abs_beta_binomial_marginal(state: SymState, prior: AbsBeta, likelihood: AbsBinomial,
                           rv_par: AbsRandomVar, rv_child: AbsRandomVar) -> Optional[AbsBetaBinomial]:
    if not abs_beta_binomial_conjugate_check(state, prior, likelihood, rv_par, rv_child):
        return None

    a, b = prior.a, prior.b
    n, p = likelihood.n, likelihood.p

    return AbsBetaBinomial(n, a, b)


def beta_binomial_posterior(state: SymState, prior: Beta, likelihood: Binomial,
                            rv_par: RandomVar, rv_child: RandomVar, abs_prior: AbsBeta, abs_likelihood: AbsBinomial,
                            abs_rv_par: AbsRandomVar, abs_rv_child: AbsRandomVar, obs: Optional[SymExpr] = None, abs_obs: Optional[AbsSymExpr] = None) -> tuple[Beta, AbsBeta] | None:
    if not beta_binomial_conjugate_check(state, prior, likelihood, rv_par, rv_child, abs_prior, abs_likelihood, abs_rv_par, abs_rv_child):
        return None

    a, b = prior.a, prior.b
    abs_a, abs_b = abs_prior.a, abs_prior.b
    n, p = likelihood.n, likelihood.p
    abs_n, abs_p = abs_likelihood.n, abs_likelihood.p

    assert isinstance(n, Const)
    assert isinstance(abs_n, AbsConst)

    a_new, abs_a_new = state.ex_add(a, rv_child, abs_a, abs_rv_child)
    abs_n_new = UnkC() if isinstance(abs_n.v, UnkC) else float(abs_n.v)
    val_mul, abs_val_mul = state.ex_mul(Const(-1), rv_child, AbsConst(-1), abs_rv_child)
    val_add, abs_val_add = state.ex_add(Const(float(n.v)), val_mul, AbsConst(abs_n_new), abs_val_mul)
    b_new, abs_b_new = state.ex_add(b, val_add, abs_b, abs_val_add)

    return Beta(a_new, b_new), AbsBeta(abs_a_new, abs_b_new)


def abs_beta_binomial_posterior(state: SymState, prior: AbsBeta, likelihood: AbsBinomial,
                            rv_par: AbsRandomVar, rv_child: AbsRandomVar, obs: Optional[AbsSymExpr] = None) -> Optional[
    AbsBeta]:
    if not abs_beta_binomial_conjugate_check(state, prior, likelihood, rv_par, rv_child):
        return None

    a, b = prior.a, prior.b
    n, p = likelihood.n, likelihood.p

    assert isinstance(n, AbsConst)

    a_new = state.abs_ex_add(a, rv_child)
    n_new = UnkC() if isinstance(n.v, UnkC) else float(n.v)
    b_new = state.abs_ex_add(b, state.abs_ex_add(AbsConst(n_new), state.abs_ex_mul(AbsConst(-1), rv_child)))

    return AbsBeta(a_new, b_new)


def beta_binomial_conjugate(state: SymState, rv_par: RandomVar, rv_child: RandomVar, abs_rv_par: AbsRandomVar, abs_rv_child: AbsRandomVar) -> tuple[BetaBinomial, Beta, AbsBetaBinomial, AbsBeta] | None:
    prior, likelihood = state.distr(rv_par), state.distr(rv_child)
    abs_prior, abs_likelihood = state.Abs_distr(rv_par), state.Abs_distr(rv_child)
    match abs_prior, abs_likelihood:
        case AbsBeta(a, b), AbsBinomial(n, p):
            marginal, abs_marginal = beta_binomial_marginal(state, prior, likelihood, rv_par, rv_child, abs_prior, abs_likelihood, abs_rv_par, abs_rv_child)
            posterior, abs_posterior = beta_binomial_posterior(state, prior, likelihood, rv_par, rv_child, abs_prior, abs_likelihood, abs_rv_par, abs_rv_child)
            if marginal is None or posterior is None:
                return None
            return marginal, posterior, abs_marginal, abs_posterior
        case _:
            return None

def abs_beta_binomial_conjugate(state: SymState, rv_par: AbsRandomVar, rv_child: AbsRandomVar) -> Optional[Tuple[AbsBetaBinomial, AbsBeta]]:
  prior, likelihood = state.Abs_distr(rv_par), state.Abs_distr(rv_child)
  match prior, likelihood:
    case AbsBeta(a, b), AbsBinomial(n, p):
      marginal = abs_beta_binomial_marginal(state, prior, likelihood, rv_par, rv_child)
      posterior = abs_beta_binomial_posterior(state, prior, likelihood, rv_par, rv_child)
      if marginal is None or posterior is None:
        return None
      return (marginal, posterior)
    case _:
      return None


### Gamma Poisson Conjugate ###

def gamma_poisson_conjugate_check(state: SymState, prior: SymDistr, likelihood: SymDistr,
                                  rv_par: RandomVar, rv_child: RandomVar, abs_prior: AbsSymDistr, abs_likelihood: AbsSymDistr,
                                    abs_rv_par: AbsRandomVar, abs_rv_child: AbsRandomVar) -> bool:
    match abs_prior, abs_likelihood:
        case AbsGamma(a, b), AbsPoisson(l):
            return isinstance(a, AbsConst) \
                and (isinstance(a.v, UnkC) or bool(np.isclose(round(a.v), a.v))) \
                and abs_rv_par == l \
                and not b.depends_on(abs_rv_child, True)
        case _:
            return False

def abs_gamma_poisson_conjugate_check(state: SymState, prior: AbsSymDistr, likelihood: AbsSymDistr,
                                    rv_par: AbsRandomVar, rv_child: AbsRandomVar) -> bool:
  match prior, likelihood:
    case AbsGamma(a, b), AbsPoisson(l):
      return isinstance(a, AbsConst) \
            and (isinstance(a.v, UnkC) or bool(np.isclose(round(a.v), a.v))) \
            and rv_par == l \
            and not b.depends_on(rv_child, True)
    case _:
      return False


def gamma_poisson_marginal(state: SymState, prior: Gamma, likelihood: Poisson,
                           rv_par: RandomVar, rv_child: RandomVar, abs_prior: AbsGamma, abs_likelihood: AbsPoisson,
                           abs_rv_par: AbsRandomVar, abs_rv_child: AbsRandomVar) -> tuple[NegativeBinomial, AbsNegativeBinomial] | None:
    if not gamma_poisson_conjugate_check(state, prior, likelihood, rv_par, rv_child, abs_prior, abs_likelihood, abs_rv_par, abs_rv_child):
        return None

    a, b = prior.a, prior.b
    abs_a, abs_b = abs_prior.a, abs_prior.b
    l = likelihood.l
    abs_l = abs_likelihood.l

    assert isinstance(a, Const)
    assert isinstance(abs_a, AbsConst)

    n_new = Const(int(a.v))
    abs_n_new = AbsConst(UnkC()) if isinstance(abs_a.v, UnkC) else AbsConst(int(abs_a.v))
    val_add, abs_val_add = state.ex_add(Const(1), b, AbsConst(1), abs_b)
    p_new, abs_p_new = state.ex_div(b, val_add, abs_b, abs_val_add)

    return NegativeBinomial(n_new, p_new), AbsNegativeBinomial(abs_n_new, abs_p_new)


def abs_gamma_poisson_marginal(state: SymState, prior: AbsGamma, likelihood: AbsPoisson,
                           rv_par: AbsRandomVar, rv_child: AbsRandomVar) -> Optional[AbsNegativeBinomial]:
    if not abs_gamma_poisson_conjugate_check(state, prior, likelihood, rv_par, rv_child):
        return None

    a, b = prior.a, prior.b
    l = likelihood.l

    assert isinstance(a, AbsConst)

    n_new = AbsConst(UnkC()) if isinstance(a.v, UnkC) else AbsConst(int(a.v))
    p_new = state.abs_ex_div(b, state.abs_ex_add(AbsConst(1), b))

    return AbsNegativeBinomial(n_new, p_new)


def gamma_poisson_posterior(state: SymState, prior: Gamma, likelihood: Poisson,
                            rv_par: RandomVar, rv_child: RandomVar,  abs_prior: AbsGamma, abs_likelihood: AbsPoisson,
                            abs_rv_par: AbsRandomVar, abs_rv_child: AbsRandomVar, obs: Optional[SymExpr] = None, abs_obs: Optional[AbsSymExpr] = None) -> tuple[Gamma, AbsGamma] | None:
    if not gamma_poisson_conjugate_check(state, prior, likelihood, rv_par, rv_child, abs_prior, abs_likelihood, abs_rv_par, abs_rv_child):
        return None

    a, b = prior.a, prior.b
    abs_a, abs_b = abs_prior.a, abs_prior.b

    x = rv_child if obs is None else obs
    abs_x = abs_rv_child if abs_obs is None else abs_obs

    a_new, abs_a_new = state.ex_add(a, x, abs_a, abs_x)
    b_new, abs_b_new = state.ex_add(b, Const(1), abs_b, AbsConst(1))

    return Gamma(a_new, b_new), AbsGamma(abs_a_new, abs_b_new)


def abs_gamma_poisson_posterior(state: SymState, prior: AbsGamma, likelihood: AbsPoisson,
                            rv_par: AbsRandomVar, rv_child: AbsRandomVar, obs: Optional[AbsSymExpr] = None) -> Optional[
    AbsGamma]:
    if not abs_gamma_poisson_conjugate_check(state, prior, likelihood, rv_par, rv_child):
        return None

    a, b = prior.a, prior.b

    x = rv_child if obs is None else obs

    a_new = state.abs_ex_add(a, x)
    b_new = state.abs_ex_add(b, AbsConst(1))

    return AbsGamma(a_new, b_new)


def gamma_poisson_conjugate(state: SymState, rv_par: RandomVar, rv_child: RandomVar, abs_rv_par: AbsRandomVar, abs_rv_child: AbsRandomVar) -> Tuple[NegativeBinomial, Gamma,AbsNegativeBinomial, AbsGamma] | None:
    prior, likelihood = state.distr(rv_par), state.distr(rv_child)
    abs_prior, abs_likelihood = state.Abs_distr(abs_rv_par), state.Abs_distr(abs_rv_child)
    match abs_prior, abs_likelihood:
        case AbsGamma(a, b), AbsPoisson(l):
            marginal, abs_marginal = gamma_poisson_marginal(state, prior, likelihood, rv_par, rv_child, abs_prior, abs_likelihood, abs_rv_par, abs_rv_child)
            posterior, abs_posterior = gamma_poisson_posterior(state, prior, likelihood, rv_par, rv_child, abs_prior, abs_likelihood, abs_rv_par, abs_rv_child)
            if marginal is None or posterior is None:
                return None
            return marginal, posterior, abs_marginal, abs_posterior
        case _:
            return None

def abs_gamma_poisson_conjugate(state: SymState, rv_par: AbsRandomVar, rv_child: AbsRandomVar) -> Optional[Tuple[AbsNegativeBinomial, AbsGamma]]:
  prior, likelihood = state.Abs_distr(rv_par), state.Abs_distr(rv_child)
  match prior, likelihood:
    case AbsGamma(a, b), AbsPoisson(l):
      marginal = abs_gamma_poisson_marginal(state, prior, likelihood, rv_par, rv_child)
      posterior = abs_gamma_poisson_posterior(state, prior, likelihood, rv_par, rv_child)
      if marginal is None or posterior is None:
        return None
      return (marginal, posterior)
    case _:
      return None


### Gamma Normal Conjugate ###
# TODO: case of var being scaled invgamma

def gamma_normal_conjugate_check(state: SymState, prior: Gamma, likelihood: Normal,
                                 rv_par: RandomVar, rv_child: RandomVar, abs_prior: AbsSymDistr, abs_likelihood: AbsSymDistr,
                                 abs_rv_par: AbsRandomVar, abs_rv_child: AbsRandomVar) -> bool:
    a, b = prior, prior
    abs_a, abs_b = abs_prior.a, abs_prior.b
    mu, var = likelihood, likelihood
    abs_mu, abs_var = abs_likelihood.mu, abs_likelihood.var

    return isinstance(abs_mu, AbsConst) \
        and var, abs_var == state.ex_div(Const(1), rv_par,AbsConst(1), abs_rv_par) \
        and not a.depends_on(abs_rv_child, True) \
        and not b.depends_on(abs_rv_child, True)


def abs_gamma_normal_conjugate_check(state: SymState, prior: AbsSymDistr, likelihood: AbsSymDistr,
                                 rv_par: AbsRandomVar, rv_child: AbsRandomVar) -> bool:
    match prior, likelihood:
        case AbsGamma(a, b), AbsNormal(mu, var):
            return isinstance(mu, AbsConst) \
                and var == state.abs_ex_div(AbsConst(1), rv_par) \
                and not a.depends_on(rv_child, True) \
                and not b.depends_on(rv_child, True)
        case _:
            return False


def gamma_normal_marginal(state: SymState, prior: Gamma, likelihood: Normal,
                          rv_par: RandomVar, rv_child: RandomVar, abs_prior: AbsGamma, abs_likelihood: AbsNormal,
                          abs_rv_par: AbsRandomVar, abs_rv_child: AbsRandomVar) -> tuple[StudentT, AbsStudentT] | None:
    if not gamma_normal_conjugate_check(state, prior, likelihood, rv_par, rv_child, abs_prior, abs_likelihood, abs_rv_par, abs_rv_child):
        return None

    a, b = prior, prior
    abs_a, abs_b = abs_prior.a, abs_prior.b
    mu, var = likelihood, likelihood
    abs_mu, abs_var = abs_likelihood.mu, abs_likelihood.var

    tau2, abs_tau2 = state.ex_div(b, a, abs_b, abs_a)
    nu, abs_nu = state.ex_mul(Const(2), a, AbsConst(2), abs_a)

    return StudentT(mu, tau2, nu), AbsStudentT(abs_mu, abs_tau2, abs_nu)


def abs_gamma_normal_marginal(state: SymState, prior: AbsGamma, likelihood: AbsNormal,
                          rv_par: AbsRandomVar, rv_child: AbsRandomVar) -> Optional[AbsStudentT]:
    if not abs_gamma_normal_conjugate_check(state, prior, likelihood, rv_par, rv_child):
        return None

    a, b = prior.a, prior.b
    mu, var = likelihood.mu, likelihood.var

    tau2 = state.abs_ex_div(b, a)
    nu = state.abs_ex_mul(AbsConst(2), a)

    return AbsStudentT(mu, tau2, nu)


def gamma_normal_posterior(state: SymState, prior: Gamma, likelihood: Normal,
                           rv_par: RandomVar, rv_child: RandomVar, abs_prior: AbsGamma, abs_likelihood: AbsNormal,
                           abs_rv_par: AbsRandomVar, abs_rv_child: AbsRandomVar, obs: Optional[SymExpr] = None, abs_obs: Optional[AbsSymExpr] = None) -> tuple[Gamma, AbsGamma] | None:
    if not gamma_normal_conjugate_check(state, prior, likelihood, rv_par, rv_child, abs_prior, abs_likelihood, abs_rv_par, abs_rv_child):
        return None

    a, b = prior, prior
    abs_a, abs_b = abs_prior.a, abs_prior.b
    mu, var = likelihood, likelihood
    abs_mu, abs_var = abs_likelihood.mu, abs_likelihood.var

    x = rv_child if obs is None else obs
    abs_x = abs_rv_child if abs_obs is None else abs_obs
    print("abs_mu in the conjugate", abs_mu)
    print("mu in the conjugate", mu)

    # assert isinstance(mu, Const)
    assert isinstance(abs_mu, AbsConst)
    

    a_new, abs_a_new = state.ex_add(a, Const(0.5), abs_a, AbsConst(0.5))
    b_inner, abs_b_inner = state.ex_add(x, GenericOp(Operator.sub,[0,mu]), abs_x, AbsConst(-abs_mu.v))
    val_mul, abs_val_mul = state.ex_mul(b_inner, b_inner, abs_b_inner, abs_b_inner)
    val_mul1, abs_val_mul1 = state.ex_mul(Const(0.5), val_mul, AbsConst(0.5), abs_val_mul)
    b_new, abs_b_new = state.ex_add(b, val_mul1, abs_b, abs_val_mul1)

    return Gamma(a_new, b_new), AbsGamma(abs_a_new, abs_b_new)


def abs_gamma_normal_posterior(state: SymState, prior: AbsGamma, likelihood: AbsNormal,
                           rv_par: AbsRandomVar, rv_child: AbsRandomVar, obs: Optional[AbsSymExpr] = None) -> Optional[
    AbsGamma]:
    if not abs_gamma_normal_conjugate_check(state, prior, likelihood, rv_par, rv_child):
        return None

    a, b = prior.a, prior.b
    mu, var = likelihood.mu, likelihood.var

    x = rv_child if obs is None else obs

    assert isinstance(mu, AbsConst)

    a_new = state.abs_ex_add(a, AbsConst(0.5))
    b_inner = state.abs_ex_add(x, AbsConst(-mu.v))
    b_new = state.abs_ex_add(b, state.abs_ex_mul(AbsConst(0.5), state.abs_ex_mul(b_inner, b_inner)))

    return AbsGamma(a_new, b_new)


def gamma_normal_conjugate(state: SymState, rv_par: RandomVar, rv_child: RandomVar, abs_rv_par: AbsRandomVar, abs_rv_child: AbsRandomVar) -> tuple[StudentT, Gamma, AbsStudentT, AbsGamma] | None:
    prior, likelihood = state.distr(rv_par), state.distr(rv_child)
    abs_prior, abs_likelihood = state.Abs_distr(abs_rv_par), state.Abs_distr(abs_rv_child)
    match abs_prior, abs_likelihood:
        case AbsGamma(a, b), AbsNormal(mu, var):
            marginal, abs_marginal = gamma_normal_marginal(state, prior, likelihood, rv_par, rv_child, abs_prior, abs_likelihood, abs_rv_par, abs_rv_child)
            posterior, abs_posterior = gamma_normal_posterior(state, prior, likelihood, rv_par, rv_child, abs_prior, abs_likelihood, abs_rv_par, abs_rv_child)
            if marginal is None or posterior is None:
                return None
            return marginal, posterior, abs_marginal, abs_posterior
        case _:
            return None

def abs_gamma_normal_conjugate(state: SymState, rv_par: AbsRandomVar, rv_child: AbsRandomVar) -> Optional[Tuple[AbsStudentT, AbsGamma]]:
  prior, likelihood = state.Abs_distr(rv_par), state.Abs_distr(rv_child)
  match prior, likelihood:
    case AbsGamma(a, b), AbsNormal(mu, var):
      marginal = abs_gamma_normal_marginal(state, prior, likelihood, rv_par, rv_child)
      posterior = abs_gamma_normal_posterior(state, prior, likelihood, rv_par, rv_child)
      if marginal is None or posterior is None:
        return None
      return (marginal, posterior)
    case _:
      return None


### Normal-Inverse-Gamma Normal Conjugate ###

def normal_inverse_gamma_normal_conjugate_check(state: SymState, prior: SymDistr, likelihood: SymDistr,
                                                rv_par: RandomVar, rv_child: RandomVar, abs_prior: AbsSymDistr, abs_likelihood: AbsSymDistr,
                                                abs_rv_par: AbsRandomVar, abs_rv_child: AbsRandomVar) -> bool:
    match abs_prior, abs_likelihood:
        case AbsNormal(mu0, var1), AbsNormal(mu, var2):
            match var2:
                case Div(AbsConst(1), var2_inner):
                    if isinstance(var2_inner, AbsRandomVar):
                        # TODO: should this be passed in
                        match state.Abs_distr(var2_inner):
                        # match state.get_entry(var2_inner, 'distribution'):
                            case AbsGamma(a, b):
                                # var1 should be the invgamma scaled by 1/lambda
                                # abs_k = abs_is_scaled(state, var1, var2)
                                k, abs_k = is_scaled(state, var1, var2, var1, var2)
                                if k is None and abs_k is None:
                                    return False
                                else:
                                    match state.eval(k,abs_k):
                                        case AbsConst(0):
                                            return False
                                        case AbsConst(_):
                                            return isinstance(mu0, AbsConst) \
                                                and mu == abs_rv_par \
                                                and not mu0.depends_on(abs_rv_child, True) \
                                                and not var1.depends_on(abs_rv_child, True)
                                        case _:
                                            return False
                            case _:
                                return False
                    else:
                        return False
                case _:
                    return False
        case _:
            return False


def abs_normal_inverse_gamma_normal_conjugate_check(state: SymState, prior: AbsSymDistr, likelihood: AbsSymDistr,
                                                rv_par: AbsRandomVar, rv_child: AbsRandomVar) -> bool:
    match prior, likelihood:
        case AbsNormal(mu0, var1), AbsNormal(mu, var2):
            match var2:
                case Div(AbsConst(1), var2_inner):
                    if isinstance(var2_inner, AbsRandomVar):
                        # TODO: should this be passed in
                        match state.Abs_distr(var2_inner):
                            case AbsGamma(a, b):
                                # var1 should be the invgamma scaled by 1/lambda
                                k = abs_is_scaled(state, var1, var2)
                                if k is None:
                                    return False
                                else:
                                    match state.eval(k):
                                        case AbsConst(0):
                                            return False
                                        case AbsConst(_):
                                            return isinstance(mu0, AbsConst) \
                                                and mu == rv_par \
                                                and not mu0.depends_on(rv_child, True) \
                                                and not var1.depends_on(rv_child, True)
                                        case _:
                                            return False
                            case _:
                                return False
                    else:
                        return False
                case _:
                    return False
        case _:
            return False


def normal_inverse_gamma_normal_marginal(state: SymState, prior: Normal, likelihood: Normal,
                                         rv_par: RandomVar, rv_child: RandomVar, abs_prior: AbsNormal, abs_likelihood: AbsNormal,
                                         abs_rv_par: AbsRandomVar, abs_rv_child: AbsRandomVar) -> tuple[StudentT, AbsStudentT] | None:
    if not normal_inverse_gamma_normal_conjugate_check(state, prior, likelihood, rv_par, rv_child, abs_prior, abs_likelihood, abs_rv_par, abs_rv_child):
        return None

    mu0, var1 = prior.mu, prior.var
    mu, var2 = likelihood.mu, likelihood.var
    abs_mu0, abs_var1 = abs_prior.mu, abs_prior.var
    abs_mu, abs_var2 = abs_likelihood.mu, abs_likelihood.var

    match abs_var2:
        case Div(AbsConst(1), var2_inner):
            if isinstance(var2_inner, AbsRandomVar):
                match state.Abs_distr(var2_inner):
                # match state.get_entry(var2_inner, 'distribution'):
                    case AbsGamma(a, b):
                        k, abs_k = is_scaled(state, var1, var2,abs_var1, abs_var2 )
                        # abs_k = abs_is_scaled(state, abs_var1, abs_var2)
                        assert k is not None
                        assert abs_k is not None
                        assert isinstance(mu0, Const)
                        assert isinstance(mu0, AbsConst)

                        lam, abs_lam = state.ex_div(Const(1), k, AbsConst(1), abs_k)

                        a_new, abs_a_new = state.ex_add(a, Const(0.5), abs_a, AbsConst(0.5))
                        b_inner, abs_b_inner = state.ex_add(rv_child, Const(-mu0.v), abs_rv_child, AbsConst(-abs_mu0.v))
                        mul_val, abs_mul_val = state.ex_mul(b_inner, b_inner, abs_b_inner, abs_b_inner)
                        div_val, abs_div_val = state.ex_div(mul_val, Const(2), abs_mul_val, AbsConst(2))
                        div_val_lam, abs_div_val_lam = state.ex_div(lam, Const(1), abs_lam, AbsConst(1))
                        div_val1, abs_div_val1 = state.ex_div(lam,div_val_lam, abs_lam, abs_div_val_lam)
                        mul_val1, abs_mul_val1 = state.ex_mul(div_val1,div_val,abs_div_val1,abs_div_val)
                        
                        b_new,abs_b_new = state.ex_add(b, mul_val1,abs_b,abs_mul_val1)

                        state.set_entry(var2_inner, distribution=Gamma(a_new, b_new))
                        state.Abs_set_distr(var2_inner, AbsGamma(abs_a_new, abs_b_new))

                        mu_new = mu0
                        abs_mu_new = abs_mu0
                        a_val_lam, abs_a_val_lam = state.ex_mul(a, lam, abs_a, abs_lam)
                        lam_const_val, abs_lam_const_val = state.ex_add(lam,Const(1), abs_lam, AbsConst(1))
                        b_mul_val, abs_b_mul_val = state.ex_mul(b,lam_const_val,abs_b,abs_lam_const_val)
                        tau2_new, abs_tau2_new = state.ex_div(b_mul_val, a_val_lam, abs_b_mul_val, abs_a_val_lam)
                        nu_new,abs_nu_new = state.ex_mul(Const(2), a, AbsConst(2),abs_a)

                        return StudentT(mu_new, tau2_new, nu_new), AbsStudentT(abs_mu_new, abs_tau2_new, abs_nu_new)
                    case _:
                        return None
            else:
                return None
        case _:
            return None


def abs_normal_inverse_gamma_normal_marginal(state: SymState, prior: AbsNormal, likelihood: AbsNormal,
                                         rv_par: AbsRandomVar, rv_child: AbsRandomVar) -> Optional[AbsStudentT]:
    if not abs_normal_inverse_gamma_normal_conjugate_check(state, prior, likelihood, rv_par, rv_child):
        return None

    mu0, var1 = prior.mu, prior.var
    mu, var2 = likelihood.mu, likelihood.var

    match var2:
        case Div(AbsConst(1), var2_inner):
            if isinstance(var2_inner, AbsRandomVar):
                match state.Abs_distr(var2_inner):
                    case AbsGamma(a, b):
                        k = abs_is_scaled(state, var1, var2)
                        assert k is not None
                        assert isinstance(mu0, AbsConst)

                        lam = state.abs_ex_div(AbsConst(1), k)

                        a_new = state.abs_ex_add(a, AbsConst(0.5))
                        b_inner = state.abs_ex_add(rv_child, AbsConst(-mu0.v))
                        b_new = state.abs_ex_add(b, state.abs_ex_mul(state.abs_ex_div(lam, state.abs_ex_div(lam, AbsConst(1))),
                                                             state.abs_ex_div(state.abs_ex_mul(b_inner, b_inner), AbsConst(2))))

                        state.Abs_set_distr(var2_inner, AbsGamma(a_new, b_new))

                        mu_new = mu0
                        tau2_new = state.abs_ex_div(state.abs_ex_mul(b, state.abs_ex_add(lam, AbsConst(1))), state.abs_ex_mul(a, lam))
                        nu_new = state.abs_ex_mul(AbsConst(2), a)

                        return AbsStudentT(mu_new, tau2_new, nu_new)
                    case _:
                        return None
            else:
                return None
        case _:
            return None


def normal_inverse_gamma_normal_posterior(state: SymState, prior: Normal, likelihood: Normal,
                                          rv_par: RandomVar, rv_child: RandomVar, abs_prior: AbsNormal, abs_likelihood: AbsNormal,
                                          abs_rv_par: AbsRandomVar, abs_rv_child: AbsRandomVar,obs: Optional[SymExpr] = None, 
                                          abs_obs: Optional[AbsSymExpr] = None) -> tuple[Normal, AbsNormal] | None:
    if not normal_inverse_gamma_normal_conjugate_check(state, prior, likelihood, rv_par, rv_child,abs_prior,abs_likelihood,abs_rv_par,abs_rv_child):
        return None

    mu0, var1 = prior.mu, prior.var
    mu, var2 = likelihood.mu, likelihood.var
    abs_mu0, abs_var1 = abs_prior.mu, abs_prior.var
    abs_mu, abs_var2 = abs_likelihood.mu, abs_likelihood.var

    x = rv_child if obs is None else obs
    abs_x = abs_rv_child if abs_obs is None else abs_obs

    match abs_var2:
        case Div(AbsConst(1), var2_inner):
            # var2 must be a random variable of invgamma
            if isinstance(var2_inner, AbsRandomVar):
                # match state.get_entry(var2_inner, 'distribution'):
                match state.Abs_distr(var2_inner):
                    case AbsGamma(a, b):
                        k, abs_k = is_scaled(state, var1, var2, abs_var1, abs_var2)
                        # abs_k = abs_is_scaled(state, abs_var1, abs_var2)
                        assert k is not None
                        assert isinstance(mu0, Const)
                        assert abs_k is not None
                        assert isinstance(abs_mu0, AbsConst)

                        lam, abs_lam = state.ex_div(Const(1), k, AbsConst(1), abs_k)

                        lam_add, abs_lam_add = state.ex_add(lam, Const(1), abs_lam, AbsConst(1))
                        lam_mul, abs_lam_mul = state.ex_mul(lam,mu0,abs_lam,abs_mu0)
                        x_add, abs_x_add = state.ex_add(lam_mul,x,abs_lam_mul,abs_x)
                        
                        mu0_new, abs_mu0_new = state.ex_div(x_add,lam_add,abs_x_add,abs_lam_add)
                        lam_new, abs_lam_new = state.ex_add(lam, Const(1), abs_lam, AbsConst(1))

                        a_new, abs_a_new = state.ex_add(a, Const(0.5), abs_a, AbsConst(0.5))
                        b_inner, abs_b_inner = state.ex_add(x, Const(-mu0.v), abs_x, AbsConst(-abs_mu0.v))
                        mul_b_inner, abs_mul_b_inner = state.ex_mul(b_inner, b_inner, abs_b_inner, abs_b_inner)
                        div_const, abs_div_const = state.ex_div(mul_b_inner,Const(2), abs_mul_b_inner,AbsConst(2))
                        div_const1, abs_div_const1 = state.ex_div(lam,Const(1), abs_lam, AbsConst(1))
                        lam_div, abs_lam_div = state.ex_div(lam,div_const1,abs_lam,abs_div_const1)
                        mul_val, abs_mul_val = state.ex_mul(lam_div,div_const, abs_lam_div, abs_div_const)
                        b_new, abs_b_new = state.ex_add(b, mul_val, abs_b, abs_mul_val)

                        state.set_entry(var2_inner, distribution=Gamma(a_new, b_new))
                        state.Abs_set_distr(var2_inner, AbsGamma(abs_a_new, abs_b_new))

                        lam_mul_val, abs_lam_mul_val = state.ex_mul(lam_new,var2_inner, abs_lam_new, var2_inner)
                        var_new, abs_var_new = state.ex_div(Const(1), lam_mul_val, AbsConst(1), abs_lam_mul_val)

                        return Normal(mu0_new, var_new), AbsNormal(abs_mu0_new, abs_var_new)
                    case _:
                        return None
            else:
                return None
        case _:
            return None


def abs_normal_inverse_gamma_normal_posterior(state: SymState, prior: AbsNormal, likelihood: AbsNormal,
                                          rv_par: AbsRandomVar, rv_child: AbsRandomVar,
                                          obs: Optional[AbsSymExpr] = None) -> Optional[AbsNormal]:
    if not abs_normal_inverse_gamma_normal_conjugate_check(state, prior, likelihood, rv_par, rv_child):
        return None

    mu0, var1 = prior.mu, prior.var
    mu, var2 = likelihood.mu, likelihood.var

    x = rv_child if obs is None else obs

    match var2:
        case Div(AbsConst(1), var2_inner):
            # var2 must be a random variable of invgamma
            if isinstance(var2_inner, AbsRandomVar):
                match state.Abs_distr(var2_inner):
                    case AbsGamma(a, b):
                        k = abs_is_scaled(state, var1, var2)
                        assert k is not None
                        assert isinstance(mu0, AbsConst)

                        lam = state.abs_ex_div(AbsConst(1), k)

                        mu0_new = state.abs_ex_div(state.abs_ex_add(state.abs_ex_mul(lam, mu0), x), state.abs_ex_add(lam, AbsConst(1)))
                        lam_new = state.abs_ex_add(lam, AbsConst(1))

                        a_new = state.abs_ex_add(a, AbsConst(0.5))
                        b_inner = state.abs_ex_add(x, AbsConst(-mu0.v))
                        b_new = state.abs_ex_add(b, state.abs_ex_mul(state.abs_ex_div(lam, state.abs_ex_div(lam, AbsConst(1))),
                                                             state.abs_ex_div(state.abs_ex_mul(b_inner, b_inner), AbsConst(2))))

                        state.Abs_set_distr(var2_inner, AbsGamma(a_new, b_new))

                        var_new = state.abs_ex_div(AbsConst(1), state.abs_ex_mul(lam_new, var2_inner))

                        return AbsNormal(mu0_new, var_new)
                    case _:
                        return None
            else:
                return None
        case _:
            return None


def normal_inverse_gamma_normal_conjugate(state: SymState, rv_par: RandomVar, rv_child: RandomVar,abs_rv_par: AbsRandomVar, abs_rv_child: AbsRandomVar) -> tuple[StudentT, Normal, AbsStudentT, AbsNormal] | None:
    # print('normal_inverse_gamma_normal_conjugate', rv_par, rv_child)
    prior, likelihood = state.distr(rv_par), state.distr(rv_child)
    abs_prior, abs_likelihood = state.Abs_distr(abs_rv_par), state.Abs_distr(abs_rv_child)
    match abs_prior, abs_likelihood:
        case AbsNormal(mu0, var1), AbsNormal(mu, var2):
            marginal, abs_marginal = normal_inverse_gamma_normal_marginal(state, prior, likelihood, rv_par, rv_child,abs_prior,abs_likelihood,abs_rv_par,abs_rv_child)
            posterior, abs_posterior = normal_inverse_gamma_normal_posterior(state, prior, likelihood, rv_par, rv_child,abs_prior,abs_likelihood,abs_rv_par,abs_rv_child)
            if abs_marginal is None or abs_posterior is None:
                return None
            return marginal, posterior,abs_marginal,abs_posterior
        case _:
            return None

def abs_normal_inverse_gamma_normal_conjugate(state: SymState, rv_par: AbsRandomVar, rv_child: AbsRandomVar) -> Optional[Tuple[AbsStudentT, AbsNormal]]:
  # print('normal_inverse_gamma_normal_conjugate', rv_par, rv_child)
  prior, likelihood = state.Abs_distr(rv_par), state.Abs_distr(rv_child)
  match prior, likelihood:
    case AbsNormal(mu0, var1), AbsNormal(mu, var2):
      marginal = abs_normal_inverse_gamma_normal_marginal(state, prior, likelihood, rv_par, rv_child)
      posterior = abs_normal_inverse_gamma_normal_posterior(state, prior, likelihood, rv_par, rv_child)
      if marginal is None or posterior is None:
        return None
      return (marginal, posterior)
    case _:
      return None
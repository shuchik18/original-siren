from importlib.metadata import distribution
from typing import Dict, Tuple, Optional, Set ,Any
from copy import copy, deepcopy
import warnings
from collections.abc import Callable

from numpy.ma.core import identity
from pandas.core.computation.ops import Constant
from typing import Literal

from siren.analysis.interface import *
from siren.grammar import *
from siren.grammar import Const, AbsConst
from siren.utils import is_pair, is_lst, get_pair, get_lst, fast_copy, match_rvs
from siren.inference_plan import *

# Shared code between different inference algorithms
# Contains the SymState interface algorithms must subclass and implement


ED = TypeVar('ED', bound=AbsSymExpr | AbsSymDistr)
OTHERSTATE = TypeVar('OTHERSTATE', bound='AbsSymState')
value_dict = []

# name_to_rv = {}
rv_to_name = {}
# rv_to_distr={}
# get_rv_to_distr={}
# set_distr_forrv ={}

# Exception for when an annotation is violated
class RuntimeViolatedAnnotationError(Exception):
    pass

class AnalysisViolatedAnnotationError(Exception):
  pass



# Symbolic state used for the hybrid inference interface
class SymState(object):
    def __init__(self, seed=None, max_rvs: int=4) -> None:
        super().__init__()
        # State has to have distribution and pv
        # State entries are maintained as a dictionary

        self.Abs_state: Dict[AbsRandomVar, Dict[str, Any]] = {}

        self.state: Dict[RandomVar, Dict[str, Any]] = {}
        self.name_to_rv: Dict[AbsRandomVar,Dict[str, Any]] = {}
        # self.rv_to_distr: Dict[AbsRandomVar,Dict[str, Any]] = {}
        # self.set_distr_forrv: Dict[RandomVar] = {}
        # self.get_rv_to_distr: Dict[RandomVar,Dict[str, Any]] = {}
        self.ctx: Context = Context()
        self.ctx1: AbsContext = AbsContext() #made the changes here
        self.Abs_counter: int = 0
        self.counter: int = 0
        # self.plan: InferencePlan = InferencePlan()
        self.rng = np.random.default_rng(seed=seed)
        self.max_rvs = max_rvs

    # Needs to be overridden if the state contains mutable objects
    def __copy__(self):
        new_state = type(self)()
        new_state.Abs_state = fast_copy(self.Abs_state)
        new_state.state = fast_copy(self.state)
        new_state.ctx = copy(self.ctx)
        new_state.ctx1 = copy(self.ctx1)
        new_state.counter = self.counter
        new_state.rng = self.rng
        return new_state

    def __eq__(self, other: 'SymState') -> bool:
        self_vars = self.vars()
        other_vars = other.vars()

        for rv1 in self_vars:
            if rv1 not in other_vars:
                return False
            for key, item in self.Abs_state[rv1].items():
                if key not in other.Abs_state[rv1] or \
                        item != other.Abs_state[rv1][key]:
                    return False

        for rv2 in other_vars:
            if rv2 not in self_vars:
                return False
            for key, item in other.Abs_state[rv2].items():
                if key not in self.Abs_state[rv2] or \
                        item != self.Abs_state[rv2][key]:
                    return False

        return True

    def new_var(self) -> Expr:
        self.counter += 1
        return Identifier(None,f"rv{self.counter}")  # change to identifier

    def new_observe(self) -> Expr:
        self.counter += 1
        return Identifier(None,f"observe{self.counter}")

    def Abs_new_var(self, min_counter: Optional[int] = None) -> AbsRandomVar:
        if min_counter is not None:
            self.counter = max(self.counter, min_counter)
        self.counter += 1

        return AbsRandomVar(f"rv{self.counter}")

    def vars(self) -> Set[RandomVar]:
        return set(self.state.keys())

    # Use this to get the value of a variable, which handles checking if the variable is in the state
    def get_entry(self, rv: RandomVar, key: str) -> Any:
        print("rv in get entry", rv)
        if rv not in self.state:
            raise ValueError(f"{rv} not in state")
        if key not in self.state[rv]:
            raise ValueError(f"{key} not in {rv}")

        return self.state[rv][key]


    def Abs_get_entry(self,rv:AbsRandomVar, key: str) -> Any:
        if rv not in self.Abs_state:
            raise ValueError(f"{rv} not in state")
        if key not in self.Abs_state[rv]:
            raise ValueError(f"{key} not in {rv}")
        # print("absget_entry",self.Abs_state[rv][key])
        return self.Abs_state[rv][key]


    def abs_set_entry(self, variable: AbsRandomVar, **kwargs) -> None:
        if variable not in self.Abs_state:
            self.Abs_state[variable] = {}
        for key, value in kwargs.items():
            self.Abs_state[variable][key] = value

            # Check if annotations violated

            # if 'distribution' in kwargs:
            #     distribution = kwargs['distribution']

            #     if isinstance(distribution, AbsDelta):
                
            #         pv = self.Abs_get_entry(variable, 'pv')
                
            #         if self.Abs_get_entry(variable, 'annotation') == Annotation.symbolic \
            #                 and distribution.sampled:
            #             raise AnalysisViolatedAnnotationError(
            #                 f"Some of {pv} has symbolic annotation but might be sampled")


    # Use this to set the value of a variable, which handles checking if the variable is in the state and if the annotation is violated
    # By trying to update a symbolic variable with a sampled distribution
    def set_entry(self, variable: RandomVar, **kwargs) -> None:
        if variable not in self.state:
            self.state[variable] = {}

        for key, value in kwargs.items():
            self.state[variable][key] = value

        # Check if annotations violated
        if 'distribution' in kwargs:
            distribution = kwargs['distribution']
            if isinstance(distribution, Delta):
                if self.annotation(variable) == Annotation.symbolic \
                        and distribution.sampled:
                    raise RuntimeViolatedAnnotationError(
                        f"{self.get_entry(variable, 'pv')} is annotated as symbolic but will be sampled")

    def pv(self, rv: RandomVar) -> Optional[Identifier]:
        return self.get_entry(rv, 'pv')

    def abs_pv(self, rv: AbsRandomVar) -> Set[Identifier]:
        return self.Abs_get_entry(rv, 'pv')

    def abs_set_annotation(self, rv: AbsRandomVar, annotation: Annotation) -> None:
        self.abs_set_entry(rv, annotation=annotation)

    def abs_set_pv(self, rv: AbsRandomVar, pv: Set[Identifier]) -> None:
        self.abs_set_entry(rv, pv=pv)


    def distr(self, rv: Expr) -> Callable[[Expr], Expr]:

        # distribution = self.get_entry(rv, 'distribution')
        # distr = self.rv_to_distr[rv]
        # return distribution
        # print("what is the value of the rv in the distr functiondfgsfgszdfgsdfgsdfg ", distr)

        return  Apply(Identifier("Prob", "get_distr"), [rv])
        # return distribution




    # def new_lookup(self,name:Identifier) -> Expr:
    #
    #     return Apply(Identifier(None,"lookup"),(name))


    def Abs_distr(self,rv: AbsRandomVar) -> AbsSymDistr:
        # print("the value of the abs_state",self.Abs_state)
        # print("the value of the rv is ",rv)

        distribution = self.Abs_get_entry(rv, 'distribution')
        
        # print("abs distribution", distribution)
        return distribution


    # def get_par(self, rv : RandomVar, num: Const) ->Expr:
    #     return rv,num

    def annotation(self, rv: RandomVar) -> Optional[Annotation]:
        # print("what is the value of the annotation?::::",Annotation)
        return self.get_entry(rv, 'annotation')

    def abs_annotation(self, rv: AbsRandomVar) -> Optional[Annotation]:
        return self.Abs_get_entry(rv, 'annotation')



    def Abs_set_distr(self, rv: AbsRandomVar, distribution: AbsSymDistr) -> None:
        print("abs rv in the set distr function ", rv)
        print("abs distr in the set distr function ", distribution)
        # if not isinstance(distribution, AbsSymDistr):
        #     raise TypeError(f"Expected abs_distr to be AbsSymDistr, got {type(distribution).__name__}")
        self.abs_set_entry(rv, distribution=distribution)


    def lookup(self,rv:AbsRandomVar) -> Expr:
        val = self.get_name_by_rv(rv)

        return Apply(Identifier("Prob","lookup"), f'"{val}"')

    def get_rv_by_name(self, name: Identifier) ->Expr:
        return self.name_to_rv.get(name)

    def get_name_by_rv(self, rv: AbsRandomVar) -> Expr:
        return rv_to_name.get(rv)

    # def dist_lookup(self,rv:Expr)->Expr:
    #     val = self.get_rv_by_name(rv)
    #     return val

    # def get_rv_to_distr(self,rv:Expr)->Expr:
    #     distr = self.rv_to_distr.get(rv)
    #     if distr is None:
    #         print(f"No distribution found for RV {rv}")
    #     else:
    #         print(f"Distribution for {rv}: {distr} (type={type(distr).__name__})")
    #     return distr

    # def set_distr_forrv(self,rv:Expr,distr:SymDistr):
    #     # if not isinstance(distr, SymDistr):
    #     #     raise TypeError(f"Tried to set distribution for {rv} that is not a SymDistr: {type(distr).__name__}")
    #     self.rv_to_distr[rv] = distr

    # def set_dist_mappings(self, abs_rv: AbsRandomVar, rv: Expr, dist: Expr)-> None:
    #     self.set_dist_for_abs_rv(abs_rv, dist)
    #     self.set_dist_for_rv(rv, dist)
    #
    # def distr_lookup(self,rv:Expr)->Expr:
    #     val = self.get_dist_by_rv(rv)
    #     return val
    #
    # def get_dist_by_abs_rv(self, abs_rv: AbsRandomVar) -> Expr:
    #     return abs_rv_to_dist.get(abs_rv)
    #
    # def get_dist_by_rv(self, rv: Expr) -> Expr:
    #     return rv_to_dist.get(rv)
    #
    # def set_dist_for_abs_rv(self, abs_rv: AbsRandomVar, dist: Expr):
    #     abs_rv_to_dist[abs_rv] = dist
    #
    # def set_dist_for_rv(self, rv: Expr, dist: Expr):
    #     rv_to_dist[rv] = dist



    def set_lookup(self,name:Identifier,rv:Expr,abs_rv: AbsRandomVar) -> Expr:
        if name in self.name_to_rv:
            existing_abs_rv = self.name_to_rv[name]
            if existing_abs_rv == abs_rv:
                rv = existing_abs_rv
            elif existing_abs_rv is None:
                self.name_to_rv[name] = abs_rv
                rv_to_name[abs_rv] = name
            else :
                rv_to_name[existing_abs_rv] = None

        else:
            self.name_to_rv[name] = abs_rv
            rv_to_name[abs_rv] = name

        # if dist is not None:
        #     self.get_dist_by_rv[rv]=dist


        def k(rest:Expr):

            return Let([],Apply(Identifier("Prob","set_lookup"),[f'"{name}"',rv]),rest)

        return k

    def set_distr(self, rv: Expr, distribution: Expr) -> Callable[[Expr], Expr]:
        # self.set_entry(rv, distribution=distribution)
        print("what is the value of the rv in the set_distr",rv)
        print("what is the value of the distribution in the set_distr",distribution)
        # self.set_distr_forrv(rv, distribution)
        def k(rest: Expr):

                # print("the value of the distribution in the set distr func is12343453 ", distribution)
                # print("the value of the rest in the set distr func is......... ", rest)
                # print("distr",distribution)
                # print("rv",rv)
                return Let([], Apply(Identifier("Prob","set_distr"), [rv,distribution]), rest)
        return k




    def set_eval(self, rv: Expr) -> Callable[[Expr], Expr]:
        # self.set_entry(rv, distribution=distribution)
        # print("what is the value of the rv in the set_distr",rv)
        # print("what is the value of the distribution in the set_distr",distribution)
        def k(rest: Expr):

                # print("the value of the distribution in the set distr func is ", distribution)
                # print("the value of the rest in the set distr func is ", rest)
                return Let([], Apply(Identifier(None,"hoist"), [rv]), rest)
        return k



    # def abs_set_pv(self, rv: AbsRandomVar, pv: Set[Identifier]) -> None:
    #     self.set_entry(rv, pv=pv)

    # def set_pv(self, rv: RandomVar, pv: Optional[Identifier]) -> None:
    #     self.set_entry(rv, pv=pv)

    # def set_annotation(self, rv: RandomVar, annotation: Optional[Annotation]) -> None:
    #     self.set_entry(rv, annotation=annotation)

    def is_sampled(self, variable: RandomVar) -> bool:
        match self.get_entry(variable, 'distribution'):
            case Delta(_, sampled):
                return sampled
            case _:
                return False

    def __len__(self) -> int:
        return len(self.state)

    def __iter__(self):
        return iter(self.state)

    def __str__(self):
        return f"SymState({', '.join(map(str, self.state.items()))})"

    # Removes unreachable variables from the state
    def clean(self) -> None:
        used_vars = set().union(*(expr.rvs() for expr in self.ctx.context.values()))
        # get referenced vars in the distribution of each ctx variable
        while True:
            new_used_vars = used_vars | set().union(*(self.distr(rv).rvs() for rv in used_vars))
            if new_used_vars == used_vars:
                used_vars = new_used_vars
                break
            else:
                used_vars = new_used_vars

        # remove unused variables
        for rv in self.vars():
            if rv not in used_vars:
                del self.state[rv]

        # Removes unused variables from the state
        # Depends on referenced_vars and subst_rv, which may need to be overridden

    def abs_clean(self, expr: Optional[AbsSymExpr] = None) -> None:
        expr_vars = set() if expr is None else set(expr.rvs())
        # other_vars = other.vars()
        used_vars = expr_vars | set().union(*(expr.rvs() for expr in self.ctx1.context.values()))
        # get referenced vars in the distribution of each ctx variable
        while True:
            new_used_vars = used_vars | self.entry_referenced_rvs(used_vars)
            if new_used_vars == used_vars:
                used_vars = new_used_vars
                break
            else:
                used_vars = new_used_vars

        # remove unused variables
        for rv in self.vars():
            if rv not in used_vars:
                del self.Abs_state[rv]

    def str_distrs(self, rv: RandomVar) -> str:
        distr = self.eval(rv)
        match self.get_entry(rv, 'distribution'):
            case Normal(mu, var):
                return f"gaussian({self.str_expr(mu)}, {self.str_expr(var)})"
            case Bernoulli(p):
                return f"Bernoulli({self.str_expr(p)})"
            case Beta(a, b):
                return f"Beta({self.str_expr(a)}, {self.str_expr(b)})"
            case Binomial(n, p):
                return f"Binomial({self.str_expr(n)}, {self.str_expr(p)})"
            case BetaBinomial(n, a, b):
                return f"BetaBinomial({self.str_expr(n)}, {self.str_expr(a)}, {self.str_expr(b)})"
            case NegativeBinomial(n, p):
                return f"NegativeBinomial({self.str_expr(n)}, {self.str_expr(p)})"
            case Gamma(a, b):
                return f"Gamma({self.str_expr(a)}, {self.str_expr(b)})"
            case Poisson(l):
                return f"Poisson({self.str_expr(l)})"
            case StudentT(mu, tau2, nu):
                return f"StudentT({self.str_expr(mu)}, {self.str_expr(tau2)}, {self.str_expr(nu)})"
            case Categorical(lower, upper, probs):
                return f"Categorical({self.str_expr(lower)}, {self.str_expr(upper)}, {self.str_expr(probs)})"
            case Delta(v, sampled):

                return f"Delta({self.str_expr(v)}, {sampled})"
            case _:
                raise ValueError(self.get_entry(rv, 'distribution'))

    def str_expr(self, expr: SymExpr) -> str:
        # expr = self.eval(expr)
        match expr:
            case Const(value):
                return str(value)
            case RandomVar(_):
                return self.str_distrs(expr)
            case Add(left, right):
                return f"({self.str_expr(left)} + {self.str_expr(right)})"
            case Mul(left, right):
                return f"({self.str_expr(left)} * {self.str_expr(right)})"
            case Div(left, right):
                return f"({self.str_expr(left)} / {self.str_expr(right)})"
            case Ite(cond, true, false):
                return f"ite({self.str_expr(cond)}, {self.str_expr(true)}, {self.str_expr(false)})"
            case Lst(es):
                return f"[{', '.join(map(self.str_expr, es))}]"
            case Pair(e1, e2):
                return f"({self.str_expr(e1)}, {self.str_expr(e2)})"
            case Eq(left, right):
                return f"({self.str_expr(left)} = {self.str_expr(right)})"
            case Lt(left, right):
                return f"({self.str_expr(left)} < {self.str_expr(right)})"
            case _:
                raise ValueError(expr)

    # Simplifies expressions as they are created

    def ex_add(self, e1:Expr, e2:Expr, abs_e1:AbsSymExpr[Number],abs_e2:AbsSymExpr[Number]) -> [Expr,[AbsSymExpr[Expr]]]:
        if getattr(self, "_fold_runtime_parameters", False):
            return (
                self.conc_ex_add(e1, e2),
                self.abs_ex_add(abs_e1, abs_e2),
            )
        match abs_e1, abs_e2:
            case AbsConst(abs_v1), AbsConst(abs_v2):
                if isinstance(abs_v1, UnkC) or isinstance(abs_v2, UnkC):
                    return Add(e1, e2), AbsConst(UnkC())
                return Const(abs_v1 + abs_v2), AbsConst(abs_v1 + abs_v2)
 
            case AbsConst(abs_v1), AbsAdd(AbsConst(abs_v2), abs_e3):
                if isinstance(abs_v1, UnkC) or isinstance(abs_v2, UnkC):
                    abs_s = AbsConst(UnkC())
                else:
                    abs_s = AbsConst(abs_v1 + abs_v2)
                match e1, e2:
                    case Const(c1), Add(Const(c2), e3):
                        return self.ex_add(Const(c1 + c2), e3, abs_s, abs_e3)
                return Add(e1, e2), self.abs_ex_add(abs_e1, abs_e2)
 
            case AbsConst(abs_v1), AbsAdd(abs_e2_in, AbsConst(abs_v3)):
                if isinstance(abs_v1, UnkC) or isinstance(abs_v3, UnkC):
                    abs_s = AbsConst(UnkC())
                else:
                    abs_s = AbsConst(abs_v1 + abs_v3)
                match e1, e2:
                    case Const(c1), Add(e2_in, Const(c3)):
                        return self.ex_add(Const(c1 + c3), e2_in, abs_s, abs_e2_in)
                return Add(e1, e2), self.abs_ex_add(abs_e1, abs_e2)
 
            case AbsAdd(AbsConst(abs_v1), abs_e1_in), abs_e3:
                match e1:
                    case Add(Const(c1), e1_in):
                        inner, abs_inner = self.ex_add(e1_in, e2, abs_e1_in, abs_e3)
                        return self.ex_add(Const(c1), inner, AbsConst(abs_v1), abs_inner)
                return Add(e1, e2), self.abs_ex_add(abs_e1, abs_e2)
 
            case AbsConst(0), abs_e2:
                return e2, abs_e2
 
            case abs_e1, AbsConst(0):
                return e1, abs_e1
 
            case _:
                return Add(e1, e2), AbsAdd(abs_e1, abs_e2)
 
    def ex_mul(self, e1: Expr, e2: Expr, abs_e1:AbsSymExpr[Number],abs_e2:AbsSymExpr[Number]) -> [Expr,AbsSymExpr[Number]]:
        
        def concrete_number(value):
            if value < 0:
                return GenericOp(
                    Operator.sub,
                    [Const(0), Const(abs(value))],
                )
            return Const(value)
 
        if isinstance(e1, Const) and type(e1.v) in (int, float):
            e1 = concrete_number(e1.v)
        if isinstance(e2, Const) and type(e2.v) in (int, float):
            e2 = concrete_number(e2.v)

        if getattr(self, "_fold_runtime_parameters", False):
            return (
                self.conc_ex_mul(e1, e2),
                self.abs_ex_mul(abs_e1, abs_e2),
            )
 
        match abs_e1, abs_e2:
            case AbsConst(abs_v1), AbsConst(abs_v2):
                if isinstance(abs_v1, UnkC) or isinstance(abs_v2, UnkC):
                    return Mul(e1, e2), AbsConst(UnkC())
                product = abs_v1 * abs_v2
                return concrete_number(product), AbsConst(product)
 
            case AbsConst(abs_v1), AbsMul(AbsConst(abs_v2), abs_e3):
                if isinstance(abs_v1, UnkC) or isinstance(abs_v2, UnkC):
                    abs_s = AbsConst(UnkC())
                else:
                    abs_s = AbsConst(abs_v1 * abs_v2)
                match e1, e2:
                    case Const(c1), Mul(Const(c2), e3):
                        return self.ex_mul(concrete_number(c1 * c2), e3, abs_s, abs_e3)
                return Mul(e1, e2), self.abs_ex_mul(abs_e1, abs_e2)
 
            case AbsConst(abs_v1), AbsMul(abs_e2_in, AbsConst(abs_v3)):
                if isinstance(abs_v1, UnkC) or isinstance(abs_v3, UnkC):
                    abs_s = AbsConst(UnkC())
                else:
                    abs_s = AbsConst(abs_v1 * abs_v3)
                match e1, e2:
                    case Const(c1), Mul(e2_in, Const(c3)):
                        return self.ex_mul(concrete_number(c1 * c3), e2_in, abs_s, abs_e2_in)
                return Mul(e1, e2), self.abs_ex_mul(abs_e1, abs_e2)
 
            case AbsConst(abs_v1), AbsAdd(AbsConst(abs_v2), abs_e3):
                if isinstance(abs_v1, UnkC) or isinstance(abs_v2, UnkC):
                    abs_s = AbsConst(UnkC())
                else:
                    abs_s = AbsConst(abs_v1 * abs_v2)
                match e1, e2:
                    case Const(c1), Add(Const(c2), e3):
                        inner, abs_inner = self.ex_mul(e1, e3, abs_e1, abs_e3)
                        return self.ex_add(
                            concrete_number(c1 * c2), inner, abs_s, abs_inner
                        )
                return Mul(e1, e2), self.abs_ex_mul(abs_e1, abs_e2)
 
            case AbsConst(0), _:
                return Const(0), AbsConst(0)
 
            case _, AbsConst(0):
                return Const(0), AbsConst(0)
 
            case _:
                return Mul(e1, e2), AbsMul(abs_e1, abs_e2)
 
 


    # def ex_add(self, e1:Expr, e2:Expr, abs_e1:AbsSymExpr[Number],abs_e2:AbsSymExpr[Number]) -> [Expr,[AbsSymExpr[Expr]]]:
    #     match abs_e1, abs_e2:
    #         # case Const(v1), Const(v2):
    #         #     return Const(v1 + v2)
    #         case AbsConst(abs_v1),AbsConst(abs_v2):
    #             if isinstance(abs_v1, UnkC) or isinstance(abs_v2, UnkC):
    #                 # return (Const(Add(abs_v1 , abs_v2)),AbsConst(AbsAdd(abs_v1,abs_v2)))
    #                 return Add(e1,e2),AbsConst(UnkC())

    #             return Const(abs_v1 + abs_v2),AbsConst(abs_v1 + abs_v2)
    #         # case Const(v1), Add(Const(v2), e3):
    #         #     return self.ex_add(Const(v1 + v2), e3)
    #         case AbsConst(abs_v1), AbsAdd(AbsConst(abs_v2), abs_e3):
    #             if isinstance(abs_v1, UnkC) or isinstance(abs_v2, UnkC):
    #                 # s = AbsConst(Add(abs_v1,abs_v2))
    #                 abs_s = AbsConst(UnkC())
    #             else:
    #                 abs_s = AbsConst(abs_v1 + abs_v2)
    #                 s = Const(abs_v1+abs_v2)
    #             return Apply(Identifier(None,"ex_add"),(e1,e2)),self.abs_ex_add(abs_e1,abs_e2)
    #         # case Const(v1), Add(e2, Const(v3)):
    #         #     return self.ex_add(Const(v1 + v3), e2)
    #         case AbsConst(abs_v1), AbsAdd(abs_e2, AbsConst(abs_v3)):
    #             if isinstance(abs_v1, UnkC) or isinstance(abs_v3, UnkC):
    #                 # s = AbsConst(Add(abs_v1,abs_v3))
    #                 abs_s = AbsConst(UnkC())
    #             else:
    #                 abs_s = AbsConst(abs_v1 + abs_v3)
    #                 s = Const(abs_v1 + abs_v3)
    #             return Apply(Identifier(None,"ex_add"),(s,abs_e2)), self.abs_ex_add(abs_s, abs_e2)
    #         # case Add(Const(v1), e2), e3:
    #         #     return self.ex_add(Const(v1), self.ex_add(e2, e3))
    #         case AbsAdd(AbsConst(abs_v1), abs_e2), abs_e3:
    #             return Apply(Identifier(None,"ex_add"),(Const(abs_v1),Apply(Identifier(None,"ex_add"),(abs_e2,abs_e3)))), self.abs_ex_add(AbsConst(abs_v1), self.abs_ex_add(abs_e2, abs_e3))
    #         # case Const(0), e2:
    #         #     return e2
    #         case AbsConst(0), abs_e2:
    #             return e2,abs_e2
    #         # case e1, Const(0):
    #         #     return e1
    #         case abs_e1, AbsConst(0):
    #             return e1,abs_e1
    #         case _:
    #             return Add(e1,e2), AbsAdd(abs_e1, abs_e2)



    # def ex_mul(self, e1: Expr, e2: Expr, abs_e1:AbsSymExpr[Number],abs_e2:AbsSymExpr[Number]) -> [Expr,AbsSymExpr[Number]]:
    #     def concrete_number(value):
    #         if value < 0:
    #             return GenericOp(
    #                 Operator.sub,
    #                 [Const(0), Const(abs(value))],
    #             )
    #         return Const(value)

    #     if isinstance(e1, Const) and type(e1.v) in (int, float):
    #         e1 = concrete_number(e1.v)
    #     if isinstance(e2, Const) and type(e2.v) in (int, float):
    #         e2 = concrete_number(e2.v)
        
    #     match abs_e1, abs_e2:
    #         # case Const(v1), Const(v2):
    #         #     return Const(v1 * v2)
    #         case AbsConst(abs_v1), AbsConst(abs_v2):
    #             if isinstance(abs_v1, UnkC) or isinstance(abs_v2, UnkC):
    #                 return Mul(e1, e2), AbsConst(UnkC())
    #             product = abs_v1 * abs_v2
    #             return concrete_number(product), AbsConst(product)
    #         # case Const(v1), Mul(Const(v2), e3):
    #         #     return self.ex_mul(Const(v1 * v2), e3)

    #         case AbsConst(abs_v1), AbsMul(AbsConst(abs_v2), abs_e3):
    #             if isinstance(abs_v1, UnkC) or isinstance(abs_v2, UnkC):
    #                 abs_s = AbsConst(UnkC())
    #             else:
    #                 abs_s = AbsConst(abs_v1 * abs_v2)
    #                 s = concrete_number(abs_v1 * abs_v2)
    #             return Apply(Identifier("Prob","ex_mul"),[s,e2.right]), self.abs_ex_mul(abs_s, abs_e3)
    #         # case Const(v1), Mul(e2, Const(v3)):
    #         #     return self.ex_mul(Const(v1 * v3), e2)
    #         case AbsConst(abs_v1), AbsMul(abs_e2, AbsConst(abs_v3)):
    #             if isinstance(abs_v1, UnkC) or isinstance(abs_v3, UnkC):
    #                 abs_s = AbsConst(UnkC())
    #             else:
    #                 abs_s = AbsConst(abs_v1 * abs_v3)
    #                 s = concrete_number(abs_v1 * abs_v3)
    #             return Apply(Identifier("Prob","ex_mul"),(s,abs_e2)), self.abs_ex_mul(abs_s, abs_e2)
    #         # case Const(v1), Add(Const(v2), e3):
    #         #     return self.ex_add(Const(v1 * v2), self.ex_mul(Const(v1), e3))
    #         case AbsConst(abs_v1), AbsAdd(AbsConst(abs_v2), abs_e3):
    #             if isinstance(abs_v1, UnkC) or isinstance(abs_v2, UnkC):
    #                 abs_s = AbsConst(UnkC())
    #             else:
    #                 abs_s = AbsConst(abs_v1 * abs_v2)
    #                 s = Const(abs_v1 * abs_v2)
    #             return Apply(Identifier("Prob","ex_add"),(s,Apply(Identifier("Prob","ex_mul"),(Const(abs_v1),abs_e3)))),self.abs_ex_add(abs_s, self.abs_ex_mul(AbsConst(abs_v1), abs_e3))
    #         # case Const(0), _:
    #         #     return Const(0)
    #         case AbsConst(0), _:
    #             return Const(0), AbsConst(0)
    #         # case _, Const(0):
    #         #     return Const(0)
    #         case _, AbsConst(0):
    #             return Const(0), AbsConst(0)
    #         case _:
    #             return Mul(e1,e2), AbsMul(abs_e1, abs_e2)

    def ex_div(self, e1: Expr, e2: Expr,abs_e1:AbsSymExpr[Number],abs_e2:AbsSymExpr[Number]) -> [Expr,AbsSymExpr[Number]]:
        
        def concrete_number(value):
            if value < 0:
                return GenericOp(
                    Operator.sub,
                    [Const(0), Const(abs(value))],
                )
            return Const(value)

        if isinstance(e1, Const) and type(e1.v) in (int, float):
            e1 = concrete_number(e1.v)
        if isinstance(e2, Const) and type(e2.v) in (int, float):
            e2 = concrete_number(e2.v)

        if getattr(self, "_fold_runtime_parameters", False):
            return (
                self.conc_ex_div(e1, e2),
                self.abs_ex_div(abs_e1, abs_e2),
            )
        
        match abs_e1, abs_e2:
            # case Const(v1), Const(v2):
            #     return Const(v1 / v2)
            case AbsConst(abs_v1), AbsConst(abs_v2):
                if isinstance(abs_v1, UnkC) or isinstance(abs_v2, UnkC):
                    return Div(e1, e2), AbsConst(UnkC())
                div1 = abs_v1 / abs_v2
                return concrete_number(div1), AbsConst(abs_v1 / abs_v2)
            # case e1, Const(1):
            #     return e1
            case abs_e1, AbsConst(1):
                return e1, abs_e1
            case _:
                return Div(e1,e2), AbsDiv(abs_e1, abs_e2)

    def ex_ite(self, cond: SymExpr[bool], true: SymExpr[T], false: SymExpr[T],abs_cond:AbsSymExpr[bool], abs_true:  AbsSymExpr[T], abs_false:AbsSymExpr[T]) -> [SymExpr[T],AbsSymExpr[T]]:
        # match cond:
        #     case Const(v):
        #         return true if v else false  #need to merge this cond
        

        def _is_const(abs_expr: AbsSymExpr) -> bool:
            match abs_expr:
                case AbsConst(_):
                    return True
                case AbsRandomVar(_):
                    match self.distr(abs_expr):
                        case AbsDelta(v, _):
                            return _is_const(v)
                        case _:
                            return False
                case AbsAdd(e1, e2):
                    return _is_const(e1) and _is_const(e2)
                case AbsMul(e1, e2):
                    return _is_const(e1) and _is_const(e2)
                case AbsDiv(e1, e2):
                    return _is_const(e1) and _is_const(e2)
                case AbsIte(abs_cond, abs_true, abs_false):
                    return _is_const(abs_cond) and _is_const(abs_true) and _is_const(abs_false)
                case AbsEq(e1, e2):
                    return _is_const(e1) and _is_const(e2)
                case AbsLt(e1, e2):
                    return _is_const(e1) and _is_const(e2)
                case AbsLst(es):
                    return all(_is_const(e) for e in es)
                case AbsPair(e1, e2):
                    return _is_const(e1) and _is_const(e2)
                case UnkE(parents):
                    return len(parents) == 0
                case TopE():
                    return False
                case _:
                    raise ValueError(abs_expr)

        if isinstance(abs_cond, AbsConst) and not isinstance(abs_cond.v, UnkC):
            return (true if abs_cond.v else false),(abs_true if abs_cond.v else abs_false)

        if not _is_const(abs_cond):
            return IfElse(cond, true, false), AbsIte(abs_cond, abs_true, abs_false)
        else:
            e = self.narrow_join_expr(abs_true, abs_false, copy(self))
            return IfElse(cond, true, false),e



    def ex_eq(self, e1: Expr[T], e2: Expr[T],abs_e1:AbsSymExpr[T],abs_e2:AbsSymExpr[T] ) -> [Expr,AbsSymExpr[bool]]:
        match abs_e1, abs_e2:
            # case Const(v1), Const(v2):
            #     return Const(v1 == v2)
            case AbsConst(abs_v1), AbsConst(abs_v2):
                if isinstance(abs_v1, UnkC) or isinstance(abs_v2, UnkC):
                    return Eq(e1, e2), AbsConst(UnkC())
                return Const(abs_v1 == abs_v2), AbsConst(abs_v1 == abs_v2)
            case _:
                return Eq(e1,e2),AbsEq(abs_e1,abs_e2)

    def ex_lt(self, e1: Expr, e2: Expr,abs_e1:AbsSymExpr[Number],abs_e2:AbsSymExpr[Number]) ->[Expr,AbsSymExpr[bool]]:
        match abs_e1, abs_e2:
            # case Const(v1), Const(v2):
            #     return Const(v1 < v2)
            case AbsConst(abs_v1), AbsConst(abs_v2):
                if isinstance(abs_v1, UnkC) or isinstance(abs_v2, UnkC):
                  return Lt(e1, e2), AbsConst(UnkC())
                return Const(abs_v1 < abs_v2), AbsConst(abs_v1 < abs_v2)
            case _:
                return Lt(e1,e2),AbsLt(abs_e1, abs_e2)

    def conc_ex_add(self, e1: SymExpr[Number], e2: SymExpr[Number]) -> SymExpr[Number]:
        match e1, e2:
            case Const(v1), Const(v2):
                return Const(v1 + v2)
            case Const(v1), Add(Const(v2), e3):
                return self.conc_ex_add(Const(v1 + v2), e3)
            case Const(v1), Add(e2, Const(v3)):
                return self.conc_ex_add(Const(v1 + v3), e2)
            case Add(Const(v1), e2), e3:
                return self.conc_ex_add(Const(v1), self.conc_ex_add(e2, e3))
            case Const(0), e2:
                return e2
            case e1, Const(0):
                return e1
            case _:
                return Add(e1, e2)

    def conc_ex_mul(self, e1: SymExpr[Number], e2: SymExpr[Number]) -> SymExpr[Number]:
        match e1, e2:
            case Const(v1), Const(v2):
                return Const(v1 * v2)
            case Const(v1), Mul(Const(v2), e3):
                return self.conc_ex_mul(Const(v1 * v2), e3)
            case Const(v1), Mul(e2, Const(v3)):
                return self.conc_ex_mul(Const(v1 * v3), e2)
            case Const(v1), Add(Const(v2), e3):
                return self.conc_ex_add(Const(v1 * v2), self.conc_ex_mul(Const(v1), e3))
            case Const(0), _:
                return Const(0)
            case _, Const(0):
                return Const(0)
            case _:
                return Mul(e1, e2)

    def conc_ex_div(self, e1: SymExpr[Number], e2: SymExpr[Number]) -> SymExpr[Number]:
        match e1, e2:
            case Const(v1), Const(v2):
                return Const(v1 / v2)
            case e1, Const(1):
                return e1
            case _:
                return Div(e1, e2)

    def conc_ex_ite(self, cond: SymExpr[bool], true: SymExpr[T], false: SymExpr[T]) -> SymExpr[T]:
        match cond:
            case Const(v):
                return true if v else false
            case _:
                return Ite(cond, true, false)

    def conc_ex_eq(self, e1: SymExpr[T], e2: SymExpr[T]) -> SymExpr[bool]:
        match e1, e2:
            case Const(v1), Const(v2):
                return Const(v1 == v2)
            case _:
                return Eq(e1, e2)

    def conc_ex_lt(self, e1: SymExpr[Number], e2: SymExpr[Number]) -> SymExpr[bool]:
        match e1, e2:
            case Const(v1), Const(v2):
                return Const(v1 < v2)
            case _:
                return Lt(e1, e2)

    def rename(self, expr: ED, old: AbsRandomVar, new: AbsRandomVar) -> ED:
        if new == old:
            return expr
        # rename each distribution
        for rv in self.vars():
            # rename references in parameters
            self.entry_rename_rv(rv, old, new)

            # use new name as entry key
            if rv == old:
                self.Abs_state[new] = self.Abs_state[rv]
                del self.Abs_state[rv]

        # rename entries in context
        self.ctx1.rename(old, new)

        return expr.rename(old, new)

    def set_dynamic(self, variable: AbsRandomVar) -> None:
        if self.annotation(variable) == Annotation.symbolic:
            raise AnalysisViolatedAnnotationError(
                f"{self.Abs_get_entry(variable, 'pv')} is annotated as symbolic but cannot be determined")


        for pv in self.abs_pv(variable):
            self.plan[pv] = DistrEnc.dynamic #changes made here

    def entry_rename_rv(self, rv: AbsRandomVar, old: AbsRandomVar, new: AbsRandomVar) -> None:
        self.Abs_set_distr(rv, self.Abs_distr(rv).rename(old, new))

    def entry_join(self, rv: AbsRandomVar, other: 'AbsSymState') -> None:
        self.Abs_set_distr(rv, self.join_distr(self.Abs_distr(rv), other.Abs_distr(rv), other))
        self.abs_set_pv(rv, self.abs_pv(rv) | other.abs_pv(rv))
        annotations = [self.abs_annotation(rv), other.abs_annotation(rv)]
        if Annotation.symbolic in annotations:
            self.abs_set_annotation(rv, Annotation.symbolic)
        elif Annotation.sample in annotations:
            self.abs_set_annotation(rv, Annotation.sample)
        else:
            self.abs_set_annotation(rv, None)



    def unify(self, e1: AbsSymExpr, e2: AbsSymExpr, other: OTHERSTATE) -> Tuple[AbsSymExpr, AbsSymExpr, OTHERSTATE]:
        vars2 = set(e2.rvs())

        capture_avoiding_mapping: Dict[AbsRandomVar, AbsRandomVar] = {}

        unify_mapping = match_rvs({}, e1, e2)

        # Get canonical variable for for each e2_var from set of e1_vars
        map2temp = {}
        # use temp vars to avoid collision
        map2canonical = {}
        for e2_var, e1_vars in unify_mapping.items():
            canon_var = min(e1_vars, key=lambda v: v.rv)

            # capture avoiding substitution to avoid mapping a variable
            # in e2 to an irrelevant variable in e2 or a variable referenced
            # by that old variable
            # note: this doesn't happen for e1 vars because they will be
            # mapped to one of the vars in the mapped-to set, which means
            # they will never be mapped to something that isn't used in e1
            if (canon_var in other.vars() and canon_var not in vars2) \
                    or canon_var in other.entry_referenced_rvs({e2_var}):
                # capture avoiding substitution
                Abs_new_var = self.Abs_new_var(other.Abs_counter)

                capture_avoiding_mapping[canon_var] = Abs_new_var

            temp_var = self.Abs_new_var(other.Abs_counter)
            map2temp[e2_var] = temp_var
            map2canonical[temp_var] = canon_var

        for old_var, Abs_new_var in capture_avoiding_mapping.items():
            e2 = other.rename(e2, old_var, Abs_new_var)

        # rename other vars in e1_vars to temp_var in e1 and self
        for e2_var, e1_vars in unify_mapping.items():
            temp_var = map2temp[e2_var]

            for e1_var in e1_vars:
                e1 = self.rename(e1, e1_var, temp_var)

        # rename e2_var to temp_var in e2 and other
        for e2_var, e1_vars in unify_mapping.items():
            temp_var = map2temp[e2_var]

            e2 = other.rename(e2, e2_var, temp_var)

        # rename temp vars to canon vars
        for temp_var, canon_var in map2canonical.items():
            e2 = other.rename(e2, temp_var, canon_var)
            e1 = self.rename(e1, temp_var, canon_var)

        return (e1, e2, other)

    def entry_referenced_rvs(self, rvs: Set[AbsRandomVar]) -> Set[AbsRandomVar]:

        return set().union(*(self.Abs_distr(rv).rvs() for rv in rvs))

    # def new_join(self,other: 'AbsSymState') -> None:
    #
    #     for name in self.name_to_rv:
    #         if name in self.name_to_rv and name in other.name_to_rv(name):
    #             self.join_expr(name,rv_to_name.get(name),other)
    #
    #         elif name in self.name_to_rv:
    #             pass
    #
    #         elif name in other.name_to_rv:
    #             self.set_pv(name,copy(other.name_to_rv(name)))
    #             for key,value in other.state[name].items():
    #                 self.set_entry(name,**{key:copy(value)})
    #
    #         else:
    #             raise ValueError(name)

            # if name not in name_to_rv and self.get_name_by_rv(name) not in rv_to_name:
            #     self.join_expr(name,rv_to_name.get(name),other)


    def join(self, other: 'AbsSymState') -> None:
        self_vars = self.vars()
        other_vars = other.vars()

        for rv in self_vars | other_vars:
            if rv in self_vars and rv in other_vars:

                self.entry_join(rv, other)
            elif rv in self_vars:
                # keep distribution
                pass
            elif rv in other_vars:
                # copy entry
                # copy pv first
                self.abs_set_pv(rv, copy(other.pv(rv)))

                for key, value in other.state[rv].items():
                    self.abs_set_entry(rv, **{key: copy(value)})
            else:
                raise ValueError(rv)

        # set counter to max of both states
        self.counter = max(self.counter, other.counter)

    def join_distr(self, d1: AbsSymDistr, d2: AbsSymDistr, other: 'AbsSymState') -> AbsSymDistr:
        match d1, d2:
            case AbsNormal(mu1, var1), AbsNormal(mu2, var2):
                return AbsNormal(self.join_expr(mu1, mu2, other), self.join_expr(var1, var2, other))
            case AbsBernoulli(p1), AbsBernoulli(p2):
                return AbsBernoulli(self.join_expr(p1, p2, other))
            case AbsBeta(a1, b1), AbsBeta(a2, b2):
                return AbsBeta(self.join_expr(a1, a2, other), self.join_expr(b1, b2, other))
            case AbsBinomial(n1, p1), AbsBinomial(n2, p2):
                return AbsBinomial(self.join_expr(n1, n2, other), self.join_expr(p1, p2, other))
            case AbsBetaBinomial(n1, a1, b1), AbsBetaBinomial(n2, a2, b2):
                return AbsBetaBinomial(self.join_expr(n1, n2, other), self.join_expr(a1, a2, other),
                                       self.join_expr(b1, b2, other))
            case AbsNegativeBinomial(n1, p1), AbsNegativeBinomial(n2, p2):
                return AbsNegativeBinomial(self.join_expr(n1, n2, other), self.join_expr(p1, p2, other))
            case AbsGamma(a1, b1), AbsGamma(a2, b2):
                return AbsGamma(self.join_expr(a1, a2, other), self.join_expr(b1, b2, other))
            case AbsPoisson(l1), AbsPoisson(l2):
                return AbsPoisson(self.join_expr(l1, l2, other))
            case AbsStudentT(mu1, tau21, nu1), AbsStudentT(mu2, tau22, nu2):
                return AbsStudentT(self.join_expr(mu1, mu2, other), self.join_expr(tau21, tau22, other),
                                   self.join_expr(nu1, nu2, other))
            case AbsCategorical(lower1, upper1, probs1), AbsCategorical(lower2, upper2, probs2):
                return AbsCategorical(self.join_expr(lower1, lower2, other), self.join_expr(upper1, upper2, other),
                                      self.join_expr(probs1, probs2, other))
            case AbsDelta(v1, sampled1), AbsDelta(v2, sampled2):
                if sampled1 == sampled2:
                    return AbsDelta(self.join_expr(v1, v2, other), sampled1)
                else:
                    # Once a delta is sampled, it is always sampled
                    return AbsDelta(self.join_expr(v1, v2, other), sampled=True)
            case TopD(), _:
                parents = []
                for p in d2.rvs():
                    if p in parents:
                        continue
                    parents.append(p)
                if len(parents) > self.max_rvs:
                    for rv_par in parents:
                        if rv_par in other.vars():
                            other.set_dynamic(rv_par)
                return TopD()
            case _, TopD():
                parents = []
                for p in d1.rvs():
                    if p in parents:
                        continue
                    parents.append(p)
                if len(parents) > self.max_rvs:
                    for rv_par in parents:
                        if rv_par in self.vars():
                            self.set_dynamic(rv_par)
                return TopD()
            case _, _:
                parents = []
                for p in d1.rvs() + d2.rvs():
                    if p in parents:
                        continue
                    parents.append(p)
                if len(parents) > self.max_rvs:
                    for rv_par in parents:
                        if rv_par in self.vars():
                            self.set_dynamic(rv_par)
                        if rv_par in other.vars():
                            other.set_dynamic(rv_par)
                    return TopD()
                return UnkD(parents)

    def narrow_join_expr(self, e1: AbsSymExpr, e2: AbsSymExpr, other: 'AbsSymState') -> AbsSymExpr:
        e1, e2, other = self.unify(e1, e2, other)
        # print(v)
        e = self.join_expr(e1, e2, other)

        self.join(other)

        return e

    def join_expr(self, e1: AbsSymExpr, e2: AbsSymExpr, other: 'AbsSymState') -> AbsSymExpr:

        match e1, e2:
            case AbsConst(v1), AbsConst(v2):
                eq = v1 == v2

                if isinstance(eq, UnkC):
                    return AbsConst(UnkC())
                elif eq:
                    return AbsConst(v1)
                else:

                    # value_dict.append(v.store_val)
                    # print(f"Appending value: {value_dict}")
                    # return AbsConst(UnkC(IfElse(v.val, v1, v2), v.store_val))
                    return AbsConst(UnkC())

            case AbsConst(_), AbsRandomVar(_):
                return e2
            case AbsRandomVar(_), AbsConst(_):
                return e1
            case AbsRandomVar(_), AbsRandomVar(_):
                return e1 if e1 == e2 else UnkE([e1, e2])
            case AbsAdd(e11, e12), AbsAdd(e21, e22):
                return AbsAdd(self.join_expr(e11, e21, other), self.join_expr(e12, e22, other))
            case AbsMul(e11, e12), AbsMul(e21, e22):
                return AbsMul(self.join_expr(e11, e21, other), self.join_expr(e12, e22, other))
            case AbsDiv(e11, e12), AbsDiv(e21, e22):
                return AbsDiv(self.join_expr(e11, e21, other), self.join_expr(e12, e22, other))
            case AbsIte(cond1, true1, false1), AbsIte(cond2, true2, false2):
                return AbsIte(self.join_expr(cond1, cond2, other),
                              self.join_expr(true1, true2, other),
                              self.join_expr(false1, false2, other))
            case AbsEq(e11, e12), AbsEq(e21, e22):
                return AbsEq(self.join_expr(e11, e21, other), self.join_expr(e12, e22, other))
            case AbsLt(e11, e12), AbsLt(e21, e22):
                return AbsLt(self.join_expr(e11, e21, other), self.join_expr(e12, e22, other))
            case AbsConst(c), AbsLst(es):
                if isinstance(c, UnkC) or isinstance(c, list):
                    return self.join_expr(AbsLst([]), e2, other)
                else:
                    raise ValueError(c)
            case AbsLst(es), AbsConst(c):
                if isinstance(c, UnkC) or isinstance(c, list):
                    return self.join_expr(e1, AbsLst([]), other)
                else:
                    raise ValueError(c)
            case AbsLst(es1), AbsLst(es2):
                es = []
                if len(es1) == 0:
                    return e2
                if len(es2) == 0:
                    return e1
                max_len = max(len(es1), len(es2))
                for i in range(max_len):
                    if i < len(es1) and i < len(es2):
                        es.append(self.join_expr(es1[i], es2[i], other))
                    else:
                        if i < len(es1):
                            rest_parents = AbsLst(es1[i:]).rvs()
                        else:  # i < len(es2)
                            rest_parents = AbsLst(es2[i:]).rvs()

                        # Collapse with the last element if it's just UnkE
                        if len(es) > 0:
                            match es[-1]:
                                case UnkE(parents):
                                    new_parents = []
                                    for p in parents + rest_parents:
                                        if p in new_parents:
                                            continue
                                        new_parents.append(p)
                                    es[-1] = UnkE(new_parents)
                                case TopE():
                                    new_parents = []
                                    for p in rest_parents:
                                        if p in new_parents:
                                            continue
                                        new_parents.append(p)
                                        if p in self.vars():
                                            self.set_dynamic(p)
                                        if p in other.vars():
                                            other.set_dynamic(p)
                                    es[-1] = TopE()
                                case _:
                                    es.append(UnkE(rest_parents))
                            break
                return AbsLst(es)
            case AbsConst(c), AbsPair(e21, e22):
                if isinstance(c, UnkC):
                    return self.join_expr(AbsPair(AbsConst(UnkC()), AbsConst(UnkC())), e2, other)
                elif isinstance(c, tuple):
                    return self.join_expr(AbsPair(AbsConst(c[0]), AbsConst(c[1])), e2, other)
                else:
                    raise ValueError(c)
            case AbsPair(e11, e12), AbsConst(c):
                if isinstance(c, UnkC):
                    return self.join_expr(e1, AbsPair(AbsConst(UnkC()), AbsConst(UnkC())), other)
                elif isinstance(c, tuple):
                    return self.join_expr(e1, AbsPair(AbsConst(c[0]), AbsConst(c[1])), other)
                else:
                    raise ValueError(c)
            case AbsPair(e11, e12), AbsPair(e21, e22):
                return AbsPair(self.join_expr(e11, e21, other), self.join_expr(e12, e22, other))
            case TopE(), _:
                parents = []
                for p in e2.rvs():
                    if p in parents:
                        continue
                    parents.append(p)

                if len(parents) > self.max_rvs:
                    for rv_par in parents:
                        if rv_par in other.vars():
                            other.set_dynamic(rv_par)
                return TopE()
            case _, TopE():
                parents = []
                for p in e1.rvs():
                    if p in parents:
                        continue
                    parents.append(p)
                if len(parents) > self.max_rvs:
                    for rv_par in parents:
                        if rv_par in self.vars():
                            self.set_dynamic(rv_par)
                return TopE()
            case _, _:
                parents = []
                for p in e1.rvs() + e2.rvs():
                    if p in parents:
                        continue
                    parents.append(p)
                if len(parents) > self.max_rvs:
                    # print(e1, e2)
                    for rv_par in parents:
                        if rv_par in self.vars():
                            self.set_dynamic(rv_par)
                        if rv_par in other.vars():
                            other.set_dynamic(rv_par)
                    return TopE()
                return UnkE(parents)

    def abs_ex_add(self, e1: AbsSymExpr[Number], e2: AbsSymExpr[Number]) -> AbsSymExpr[Number]:
        match e1, e2:
            case AbsConst(v1), AbsConst(v2):
                if isinstance(v1, UnkC) or isinstance(v2, UnkC):
                    return AbsConst(UnkC())
                return AbsConst(v1 + v2)
            case AbsConst(v1), AbsAdd(AbsConst(v2), e3):
                if isinstance(v1, UnkC) or isinstance(v2, UnkC):
                    s = AbsConst(UnkC())
                else:
                    s = AbsConst(v1 + v2)
                return self.abs_ex_add(s, e3)
            case AbsConst(v1), AbsAdd(e2, AbsConst(v3)):
                if isinstance(v1, UnkC) or isinstance(v3, UnkC):
                    s = AbsConst(UnkC())
                else:
                    s = AbsConst(v1 + v3)
                return self.abs_ex_add(s, e2)
            case AbsAdd(AbsConst(v1), e2), e3:
                return self.abs_ex_add(AbsConst(v1), self.abs_ex_add(e2, e3))
            case AbsConst(0), e2:
                return e2
            case e1, AbsConst(0):
                return e1
            case _:
                return AbsAdd(e1, e2)

    def abs_ex_mul(self, e1: AbsSymExpr[Number], e2: AbsSymExpr[Number]) -> AbsSymExpr[Number]:
        match e1, e2:
            case AbsConst(v1), AbsConst(v2):
                if isinstance(v1, UnkC) or isinstance(v2, UnkC):
                    return AbsConst(UnkC())
                return AbsConst(v1 * v2)
            case AbsConst(v1), AbsMul(AbsConst(v2), e3):
                if isinstance(v1, UnkC) or isinstance(v2, UnkC):
                    s = AbsConst(UnkC())
                else:
                    s = AbsConst(v1 * v2)
                return self.abs_ex_mul(s, e3)
            case AbsConst(v1), AbsMul(e2, AbsConst(v3)):
                if isinstance(v1, UnkC) or isinstance(v3, UnkC):
                    s = AbsConst(UnkC())
                else:
                    s = AbsConst(v1 * v3)
                return self.abs_ex_mul(s, e2)
            case AbsConst(v1), AbsAdd(AbsConst(v2), e3):
                if isinstance(v1, UnkC) or isinstance(v2, UnkC):
                    s = AbsConst(UnkC())
                else:
                    s = AbsConst(v1 * v2)
                return self.abs_ex_add(s, self.abs_ex_mul(AbsConst(v1), e3))
            case AbsConst(0), _:
                return AbsConst(0)
            case _, AbsConst(0):
                return AbsConst(0)
            case _:
                return AbsMul(e1, e2)

    def abs_ex_div(self, e1: AbsSymExpr[Number], e2: AbsSymExpr[Number]) -> AbsSymExpr[Number]:
        match e1, e2:
            case AbsConst(v1), AbsConst(v2):
                if isinstance(v1, UnkC) or isinstance(v2, UnkC):
                    return AbsConst(UnkC())
                return AbsConst(v1 / v2)
            case e1, AbsConst(1):
                return e1
            case _:
                return AbsDiv(e1, e2)

    def abs_ex_ite(self, cond: AbsSymExpr[bool], true: AbsSymExpr[T], false: AbsSymExpr[T]) -> AbsSymExpr[T]:
        def _is_const(expr: AbsSymExpr) -> bool:
            match expr:
                case AbsConst(_):
                    return True
                case AbsRandomVar(_):
                    match self.Abs_distr(expr):
                        case AbsDelta(v, _):
                            return _is_const(v)
                        case _:
                            return False
                case AbsAdd(e1, e2):
                    return _is_const(e1) and _is_const(e2)
                case AbsMul(e1, e2):
                    return _is_const(e1) and _is_const(e2)
                case AbsDiv(e1, e2):
                    return _is_const(e1) and _is_const(e2)
                case AbsIte(cond, true, false):
                    return _is_const(cond) and _is_const(true) and _is_const(false)
                case AbsEq(e1, e2):
                    return _is_const(e1) and _is_const(e2)
                case AbsLt(e1, e2):
                    return _is_const(e1) and _is_const(e2)
                case AbsLst(es):
                    return all(_is_const(e) for e in es)
                case AbsPair(e1, e2):
                    return _is_const(e1) and _is_const(e2)
                case UnkE(parents):
                    return len(parents) == 0
                case TopE():
                    return False
                case _:
                    raise ValueError(expr)

        if isinstance(cond, AbsConst) and not isinstance(cond.v, UnkC):
            return true if cond.v else false

        if not _is_const(cond):
            return AbsIte(cond, true, false)
        else:
            e = self.narrow_join_expr(true, false, copy(self), cond)
            return e

    def abs_ex_eq(self, e1: AbsSymExpr[T], e2: AbsSymExpr[T]) -> AbsSymExpr[bool]:
        match e1, e2:
            case AbsConst(v1), AbsConst(v2):
                if isinstance(v1, UnkC) or isinstance(v2, UnkC):
                    return AbsConst(UnkC())
                return AbsConst(v1 == v2)
            case _:
                return AbsEq(e1, e2)

    def abs_ex_lt(self, e1: AbsSymExpr[Number], e2: AbsSymExpr[Number]) -> AbsSymExpr[bool]:
        match e1, e2:
            case AbsConst(v1), AbsConst(v2):
                if isinstance(v1, UnkC) or isinstance(v2, UnkC):
                    return AbsConst(UnkC())
                return AbsConst(v1 < v2)
            case _:
                return AbsLt(e1, e2)

    def ex_delta(self,expr: Expr)->Expr:

        return Apply(Identifier(None, "extract_delta"), (expr))




    # Simplify expressions
    def eval(self, expr: Expr,abs_expr:AbsSymExpr ) -> [Expr,AbsSymExpr]:
        # print("expr1................", expr)
        # print("absexpr1................", abs_expr)
        if not isinstance(expr, Expr):
            raise TypeError(f"Expected expr to be Expr, got {type(expr).__name__}")
        if not isinstance(abs_expr, AbsSymExpr):
            raise TypeError(f"Expected abs_expr to be AbsSymExpr, got {type(abs_expr).__name__}")

        def _const_list(abs_es:List[AbsSymExpr] ) -> [Const, AbsConst]:
            if not isinstance(abs_es, list):
                raise TypeError(f"_const_list expected a List[AbsSymExpr], got {type(abs_es).__name__}")

            consts = []
            for e in abs_es:
                if not isinstance(e, Const):
                    return None
                consts.append(e.v)

            return Const(consts),AbsConst(consts)

        def _eval(expr:Expr, abs_expr: AbsSymExpr) -> (Expr,AbsSymExpr):
            # if isinstance(abs_expr, (int, float)):
            #     abs_expr = AbsConst(abs_expr)
            # if not isinstance(expr, Expr):
            #     raise TypeError(f"_eval expected Expr, got {type(expr).__name__}")
            # if not isinstance(abs_expr, AbsSymExpr):
            #     raise TypeError(f"_eval expected AbsSymExpr, got {type(abs_expr).__name__}")
            # print("expr12334422................",expr)
            # print("absexpr1234412................",abs_expr)
            match abs_expr:
                case AbsConst(v):
                    if (
                        isinstance(v, UnkC)
                        or getattr(self, "_fold_runtime_parameters", False)
                    ):
                        return expr, abs_expr

                    return Const(v), abs_expr
                    # return (expr if isinstance(v, UnkC) else Const(v)), abs_expr
                case AbsRandomVar(_):
                    distribution = self.Abs_distr(abs_expr)

                    if isinstance(distribution, AbsDelta):
                        return _eval(expr, distribution.v)

                    if isinstance(distribution, AbsNormal):
                        return expr, abs_expr

                    name = self.get_name_by_rv(abs_expr)
                    return (
                        name if name is not None else expr
                    ), abs_expr
                case AbsAdd(abs_e1, abs_e2):
                    if not isinstance(expr, Add):
                        return expr, self.Abs_eval(abs_expr)
                    # print("5")
                    # (e1,abs_expr_e1) = _eval(self.distr(expr),abs_e1)
                    (e1,abs_expr_e1) = _eval(expr.left if isinstance(expr, Add) else expr, abs_e1)
                    (e2,abs_expr_e2) = _eval(expr.right if isinstance(expr, Add) else expr, abs_e2)
                    # print("=================================inside eval==================================")
                    # print("e1",e1)
                    # print("e2",e2)
                    # print("abs_e1",abs_expr_e1)
                    # print("abs_e2",abs_expr_e2)
                    # print("===================================================================")
                    (e1_expr,abs_e1_expr) = self.ex_add(e1,e2,abs_expr_e1,abs_expr_e2)
                    return e1_expr, abs_e1_expr
                    # return self.ex_add(e1_expr,e2_expr,abs_expr_e1,abs_expr_e2)
                    # return Apply(Identifier(None,"add"),[e1_expr,e2_expr]),self.abs_ex_add(abs_expr_e1,abs_expr_e2)

                case AbsMul(abs_e1, abs_e2):
                    if not isinstance(expr, Mul):
                        return expr, self.Abs_eval(abs_expr)
                    # print("6")

                    (e1,abs_expr_e1) = _eval(expr.left if isinstance(expr, Mul) else expr, abs_e1)
                    (e2,abs_expr_e2) = _eval(expr.right if isinstance(expr, Mul) else expr, abs_e2)
                    (e1_expr,abs_e1_expr) = self.ex_mul(e1,e2,abs_expr_e1,abs_expr_e2)
                    return e1_expr, abs_e1_expr
                    # return self.ex_mul(e1_expr,e2_expr,abs_expr_e1,abs_expr_e2)
                    # return Apply(Identifier(None,"mul"),[e1_expr,e2_expr]),self.abs_ex_mul(abs_expr_e1,abs_expr_e2)

                case AbsDiv(abs_e1, abs_e2):
                    if not isinstance(expr, Div):
                        return expr, self.Abs_eval(abs_expr)
                    # print("7")
                    (e1,abs_expr_e1) = _eval(expr.left if isinstance(expr, Div) else expr, abs_e1)
                    (e2,abs_expr_e2) = _eval(expr.right if isinstance(expr, Div) else expr, abs_e2)
                    # print("=================================inside eval==================================")
                    # print("e1",e1)
                    # print("e2",e2)
                    # print("abs_e1",abs_expr_e1)
                    # print("abs_e2",abs_expr_e2)
                    # print("===================================================================")
                    (e1_expr,abs_e1_expr) = self.ex_div(e1,e2,abs_expr_e1,abs_expr_e2)
                    return e1_expr, abs_e1_expr
                    # return self.ex_div(e1_expr,e2_expr,abs_expr_e1,abs_expr_e2)
                    # return Apply(Identifier(None,"div"),[e1_expr,e2_expr]),self.abs_ex_div(abs_expr_e1,abs_expr_e2)

                case AbsIte(abs_cond, abs_true, abs_false):
                    match expr:
                        case IfElse(cond_node, true_node, false_node):
                            pass
                        case Ite(cond_node, true_node, false_node):
                            pass
                        case _:
                            return expr, self.Abs_eval(abs_expr)

                    cond, abs_expr_cond = _eval(
                        cond_node, abs_cond
                    )
                    true, abs_expr_true = _eval(
                        true_node, abs_true
                    )
                    false, abs_expr_false = _eval(
                        false_node, abs_false
                    )
                    return self.ex_ite(
                        cond, true, false,
                        abs_expr_cond, abs_expr_true, abs_expr_false
                    )
                case AbsEq(abs_e1, abs_e2):
                    match expr:
                        case Eq(left, right):
                            pass
                        case GenericOp(
                            op=Operator.eq, args=[left, right]
                        ):
                            pass
                        case _:
                            # No accessible concrete operands.
                            # Preserve the expression rather than
                            # incorrectly using it as both operands.
                            return expr, self.Abs_eval(abs_expr)

                    e1, abs_expr_e1 = _eval(left, abs_e1)
                    e2, abs_expr_e2 = _eval(right, abs_e2)
                    return self.ex_eq(
                        e1, e2, abs_expr_e1, abs_expr_e2
                    )

                case AbsLt(abs_e1, abs_e2):
                    match expr:
                        case Lt(left, right):
                            pass
                        case GenericOp(
                            op=Operator.lt, args=[left, right]
                        ):
                            pass
                        case _:
                            return expr, self.Abs_eval(abs_expr)

                    e1, abs_expr_e1 = _eval(left, abs_e1)
                    e2, abs_expr_e2 = _eval(right, abs_e2)
                    return self.ex_lt(
                        e1, e2, abs_expr_e1, abs_expr_e2
                    )
                case AbsLst(abs_es):
                    es1,abs_es1 = [self.eval(self.distr(expr),e) for e in abs_es]
                    const_list = _const_list(abs_es)
                    return const_list if const_list is not None else AbsLst(abs_es)
                case AbsPair(abs_e1, abs_e2):
                    (e1_expr,abs_expr_e1) = self.eval(self.distr(expr),abs_e1)
                    (e2_expr,abs_expr_e2) = self.eval(self.distr(expr),abs_e2)
                    # abs_e1, abs_e2 = self.eval(e1,abs_e1), self.eval(e2,abs_e2)
                    match abs_expr_e1, abs_expr_e2:
                        case Const(_), Const(_):
                            return Const(e1_expr,e2_expr),Const((abs_expr_e1.v, abs_expr_e2.v))
                        case _:
                            return Pair(e1_expr,e2_expr),AbsPair(abs_expr_e1, abs_expr_e2)
                case UnkE(s):
                    parents = []
                    for var in s:
                        match self.Abs_distr(var):
                            case AbsDelta(v, _):
                                continue
                            case _:
                                if var not in parents:
                                    parents.append(var)
                        # If no parents, then it's a constant
                        if len(parents) == 0:
                            return expr,AbsConst(UnkC())
                    return expr,UnkE(parents)
                case TopE():
                    return expr,TopE()
                case _:
                    raise TypeError(f"Unsupported AbsSymExpr type: {type(abs_expr).__name__}")
                            # raise ValueError(expr)

        return _eval(expr,abs_expr)


    def eval_distr(self, distr: Expr,abs_distr: AbsSymDistr) -> [SymDistr,AbsSymDistr]:
        if not isinstance(distr, Expr):
            raise TypeError(f"Expected distr to be SymDistr, got {type(distr).__name__}")
        if not isinstance(abs_distr, AbsSymDistr):
            print("abs_distr in eval distr", abs_distr)
            raise TypeError(f"Expected abs_distr to be AbsSymDistr, got {type(abs_distr).__name__}")
        match abs_distr:
            case AbsNormal(abs_mu, abs_var):
                print("=======inside eval distr=======")
                print("mu values ",abs_mu)
                print("var val",abs_var)

                (mu_expr,abs_mu_expr) = self.eval(Apply(Identifier("Prob", "get_mu"), [distr]), abs_mu)
                (var_expr,abs_var_expr)= self.eval(Apply(Identifier("Prob", "get_var"), [distr]), abs_var)
                print("mu_expr",mu_expr)
                print("var_expr",var_expr)
                print("================")
                # return Normal(mu_expr,var_expr), AbsNormal(abs_mu_expr,abs_var_expr)
                return Apply(Identifier(None,"gaussian"),[mu_expr,var_expr]),AbsNormal(abs_mu_expr,abs_var_expr)
            case AbsBernoulli(abs_p):
                print("distr", distr)
                (p_expr,abs_p_expr) = self.eval(Apply(Identifier("Prob", "get_p"), [distr]),abs_p)
                # return Bernoulli(p_expr), AbsBernoulli(abs_p_expr)
                return Apply(Identifier(None,"bernoulli"),[p_expr]),AbsBernoulli(abs_p_expr)
            case AbsBeta(abs_a, abs_b):
                
                a, abs_expr_a = self.eval(Apply(Identifier("Prob", "get_a"), [distr]),abs_a)
                b, abs_expr_b = self.eval(Apply(Identifier("Prob", "get_b"), [distr]),abs_b)


                return Apply(Identifier(None,"beta"),[a,b]),AbsBeta(abs_expr_a,abs_expr_b)
            case AbsBinomial(abs_n, abs_p):
                (n,abs_expr_n) = self.eval(distr,abs_n)
                (p,abs_expr_p) = self.eval(distr,abs_p)
                return Apply(Identifier(None,"binomial"),[n,p]),AbsBinomial(abs_expr_n,abs_expr_p)
            case AbsBetaBinomial(abs_n, abs_a, abs_b):
                (n,abs_expr_n) = self.eval(distr,abs_n)
                (a,abs_expr_a) = self.eval(distr,abs_a)
                (b,abs_expr_b) = self.eval(distr,abs_b)
                return Apply(Identifier(None,"betabinomial"),[n,a,b]),AbsBetaBinomial(abs_expr_n,abs_expr_a,abs_expr_b)
            case AbsNegativeBinomial(abs_n, abs_p):
                (n,abs_expr_n) = self.eval(Apply(Identifier(None,"get_n"),([distr])),abs_n)
                (p,abs_expr_p) = self.eval(Apply(Identifier(None,"get_p"),([distr])),abs_p)
                return Apply(Identifier(None,"negativebinomial"),[n,p]),AbsNegativeBinomial(abs_expr_n,abs_expr_p)
            case AbsGamma(abs_a, abs_b):
                (a,abs_expr_a) = self.eval(distr,abs_a)
                (b,abs_expr_b) = self.eval(distr,abs_b)
                return Apply(Identifier(None,"gamma"),[a,b]),AbsGamma(abs_expr_a,abs_expr_b)
            case AbsPoisson(abs_l):
                (l,abs_expr_l) = self.eval(Apply(Identifier(None,"get_l"),([distr])),abs_l)
                return Apply(Identifier(None,"poisson"),(l)),AbsPoisson(abs_l)
            case AbsStudentT(abs_mu, abs_tau2, abs_nu):
                (mu,abs_expr_mu) = self.eval(Apply(Identifier(None,"get_mu"),([distr])),abs_mu)
                (tau2,abs_expr_tau2) = self.eval(Apply(Identifier(None,"get_tau2"),([distr])),abs_tau2)
                (nu,abs_expr_nu) = self.eval(Apply(Identifier(None,"get_nu"),([distr])),abs_nu)
                return Apply(Identifier(None,"studentT"),[mu,tau2,nu]),AbsStudentT(abs_expr_mu,abs_expr_tau2,abs_expr_nu)
            case AbsCategorical(abs_lower, abs_upper, abs_probs):
                (lower,abs_expr_lower) = self.eval(Apply(Identifier(None,"get_lower"),([distr])),abs_lower)
                (upper,abs_expr_upper) = self.eval(Apply(Identifier(None,"get_upper"),([distr])),abs_upper)
                (probs,abs_expr_probs) = self.eval(Apply(Identifier(None,"get_probs"),([distr])),abs_probs)
                return Apply(Identifier(None,"categorical"),[lower,upper,probs]),AbsCategorical(abs_expr_lower, abs_expr_upper, abs_expr_probs)
            case AbsDelta(abs_v, abs_sampled):
                (v,abs_expr_v) = self.eval(Apply(Identifier(None,"get_v"),([distr])),abs_v)
                (sampled,abs_expr_sampled) = self.eval(Apply(Identifier(None,"get_sampled"),([distr])),abs_sampled)
                return Apply(Identifier(None,"delta"),[v,sampled]),AbsDelta(abs_expr_v,abs_expr_sampled)
            case UnkD(s):
                parents = []
                for var in s:
                    match self.Abs_distr(var):
                        case AbsDelta(v, _):
                            continue
                        case _:
                            if var not in parents:
                                parents.append(var)

                return Apply(Identifier(None,"UnkD"),([distr])), UnkD(parents)
            case TopD():
                return Apply(Identifier(None,"TopD"),([distr])),TopD()
            case _:
                # print("errorrr...............",distr)
                raise TypeError(f"Unsupported AbsSymDistr type: {type(abs_distr).__name__}")
                # raise ValueError(distr)

    def conc_eval(self, expr: SymExpr) -> SymExpr:
        def _const_list(es: List[SymExpr]) -> Optional[Const[List[SymExpr]]]:
            consts = []
            for e in es:
                if not isinstance(e, Const):
                    return None
                consts.append(e.v)

            return Const(consts)

        def _eval_conc(expr: SymExpr) -> SymExpr:
            match expr:
                case Const(_):
                    return expr
                case RandomVar(_):
                    match self.get_entry(expr, 'distribution'):
                        case Delta(v, _):
                            return _eval_conc(v)
                        case _:
                            self.set_entry(expr, distribution=self.conc_eval_distr(self.get_entry(expr, 'distribution')))
                            return expr
                case Add(e1, e2):
                    return self.conc_ex_add(_eval_conc(e1), _eval_conc(e2))
                case Mul(e1, e2):
                    return self.conc_ex_mul(_eval_conc(e1), _eval_conc(e2))
                case Div(e1, e2):
                    return self.conc_ex_div(_eval_conc(e1), _eval_conc(e2))
                case Ite(cond, true, false):
                    return self.conc_ex_ite(_eval_conc(cond), _eval_conc(true), _eval_conc(false))
                case Eq(e1, e2):
                    return self.conc_ex_eq(_eval_conc(e1), _eval_conc(e2))
                case Lt(e1, e2):
                    return self.conc_ex_lt(_eval_conc(e1), _eval_conc(e2))
                case Lst(es):
                    es = [self.conc_eval(e) for e in es]
                    const_list = _const_list(es)
                    return const_list if const_list is not None else Lst(es)
                case Pair(e1, e2):
                    e1, e2 = self.conc_eval(e1), self.conc_eval(e2)
                    match e1, e2:
                        case Const(_), Const(_):
                            return Const((e1.v, e2.v))
                        case _:
                            return Pair(e1, e2)
                case _:
                    raise ValueError(expr)

        return _eval_conc(expr)

    def conc_eval_distr(self, distr: SymDistr) -> SymDistr:
        match distr:
            case Normal(mu, var):
                return Normal(self.conc_eval(mu), self.conc_eval(var))
            case Bernoulli(p):
                return Bernoulli(self.conc_eval(p))
            case Beta(a, b):
                return Beta(self.conc_eval(a), self.conc_eval(b))
            case Binomial(n, p):
                return Binomial(self.conc_eval(n), self.conc_eval(p))
            case BetaBinomial(n, a, b):
                return BetaBinomial(self.conc_eval(n), self.conc_eval(a), self.conc_eval(b))
            case NegativeBinomial(n, p):
                return NegativeBinomial(self.conc_eval(n), self.conc_eval(p))
            case Gamma(a, b):
                return Gamma(self.conc_eval(a), self.conc_eval(b))
            case Poisson(l):
                return Poisson(self.conc_eval(l))
            case StudentT(mu, tau2, nu):
                return StudentT(self.conc_eval(mu), self.conc_eval(tau2), self.conc_eval(nu))
            case Categorical(lower, upper, probs):
                return Categorical(self.conc_eval(lower), self.conc_eval(upper), self.conc_eval(probs))
            case Delta(v, sampled):
                return Delta(self.conc_eval(v), sampled)
            case _:
                raise ValueError(distr)





    def Abs_eval(self, expr: AbsSymExpr) -> AbsSymExpr:
        def _const_list(es: List[AbsSymExpr]) -> Optional[AbsConst[List[AbsSymExpr]]]:
            consts = []
            for e in es:
                if not isinstance(e, AbsConst):
                    return None
                consts.append(e.v)

            return AbsConst(consts)

        def _eval(expr: AbsSymExpr) -> AbsSymExpr:
            # print("the value of the expr",expr)
            match expr:
                case AbsConst(_):
                    return expr
                case AbsRandomVar(_):
                    match self.Abs_distr(expr):
                        case AbsDelta(v, _):
                            return _eval(v)
                        case _:

                            self.Abs_set_distr(expr, self.abs_eval_distr(self.Abs_distr(expr)))

                            # self.Abs_set_distr(expr, Apply(Identifier(None,"abs_eval_distr")))
                            return expr
                case AbsAdd(e1, e2):
                    return self.abs_ex_add(_eval(e1), _eval(e2))
                case AbsMul(e1, e2):
                    return self.abs_ex_mul(_eval(e1), _eval(e2))
                case AbsDiv(e1, e2):
                    return self.abs_ex_div(_eval(e1), _eval(e2))
                case AbsIte(cond, true, false):
                    return self.abs_ex_ite(_eval(cond), _eval(true), _eval(false))
                case AbsEq(e1, e2):
                    return self.abs_ex_eq(_eval(e1), _eval(e2))
                case AbsLt(e1, e2):
                    return self.abs_ex_lt(_eval(e1), _eval(e2))
                case AbsLst(es):
                    es = [self.Abs_eval(e) for e in es]
                    const_list = _const_list(es)
                    return const_list if const_list is not None else AbsLst(es)
                case AbsPair(e1, e2):
                    e1, e2 = self.Abs_eval(e1), self.Abs_eval(e2)
                    match e1, e2:
                        case AbsConst(_), AbsConst(_):
                            return AbsConst((e1.v, e2.v))
                        case _:
                            return AbsPair(e1, e2)
                case UnkE(s):
                    parents = []
                    for var in s:
                        match self.Abs_distr(var):
                            case AbsDelta(v, _):
                                continue
                            case _:
                                if var not in parents:
                                    parents.append(var)
                    # If no parents, then it's a constant
                    if len(parents) == 0:
                        return AbsConst(UnkC())
                    return UnkE(parents)
                case TopE():
                    return TopE()
                case _:
                    raise ValueError(expr)

        return _eval(expr)


    def abs_eval_distr(self, distr: AbsSymDistr) -> AbsSymDistr:
        match distr:
            case AbsNormal(mu, var):
                return AbsNormal(self.Abs_eval(mu), self.Abs_eval(var))
            case AbsBernoulli(p):
                return AbsBernoulli(self.Abs_eval(p))
            case AbsBeta(a, b):
                return AbsBeta(self.Abs_eval(a), self.Abs_eval(b))
            case AbsBinomial(n, p):
                return AbsBinomial(self.Abs_eval(n), self.Abs_eval(p))
            case AbsBetaBinomial(n, a, b):
                return AbsBetaBinomial(self.Abs_eval(n), self.Abs_eval(a), self.Abs_eval(b))
            case AbsNegativeBinomial(n, p):
                return AbsNegativeBinomial(self.Abs_eval(n), self.Abs_eval(p))
            case AbsGamma(a, b):
                return AbsGamma(self.Abs_eval(a), self.Abs_eval(b))
            case AbsPoisson(l):
                return AbsPoisson(self.Abs_eval(l))
            case AbsStudentT(mu, tau2, nu):
                return AbsStudentT(self.Abs_eval(mu), self.Abs_eval(tau2), self.Abs_eval(nu))
            case AbsCategorical(lower, upper, probs):
                return AbsCategorical(self.Abs_eval(lower), self.Abs_eval(upper), self.Abs_eval(probs))
            case AbsDelta(v, sampled):
                return AbsDelta(self.Abs_eval(v), sampled)
            case UnkD(s):
                parents = []
                for var in s:
                    match self.Abs_distr(var):
                        case AbsDelta(v, _):
                            continue
                        case _:
                            if var not in parents:
                                parents.append(var)
                return UnkD(parents)
            case TopD():
                return TopD()
            case _:
                raise ValueError(distr)

    # Computes the expectation of the given symbolic expression
    def mean(self, expr: SymExpr) -> float:
        expr = self.eval(expr)

        match expr:
            case Const(value):
                return value
            case RandomVar(_):
                self.marginalize(expr)
                return self.get_entry(expr, 'distribution').mean()
            case Add(left, right):
                return self.mean(left) + self.mean(right)
            case Mul(left, right):
                return self.mean(left) * self.mean(right)
            case Div(left, right):
                return self.mean(left) / self.mean(right)
            case Ite(cond, true, false):
                cond = self.mean(cond)
                true = self.mean(true)
                false = self.mean(false)
                return cond * true + (1 - cond) * false
            case Eq(left, right):
                return self.mean(left) == self.mean(right)
            case Lt(left, right):
                return self.mean(left) < self.mean(right)
            case _:
                raise ValueError(expr)

    # Needs to be overridden by the implementation
    def marginalize(self, expr: RandomVar) -> None:
        raise NotImplementedError()

    # Samples all the random variables in the expression and simplifies it
    def value_expr(self, expr: Expr,abs_expr: AbsSymExpr) -> Tuple[Expr,Expr]:
        print("inside value expr")
        print("expr in val_expr", expr)
        print("abs_expr in val_expr", abs_expr)
        match abs_expr:
            case AbsConst(_):
                print("enter absconst in value expr")
                return expr,abs_expr
            case AbsRandomVar(_):
                print("enter absrandomvar in value expr")
                (p, val, abs_val) = self.value(expr,abs_expr)
                print("inside the valueexpr of the case absrandom var")
                print(p)
                print(val)
                print(abs_val)
                return val,abs_val
            case AbsAdd(abs_fst,abs_snd):
                print("enter absadd in value expr")
                (fst,abs_expr_fst)  = self.value_expr(Apply(Identifier(None,"add_fst"),(self.distr(expr))),abs_fst)
                (snd,abs_expr_snd) = self.value_expr(Apply(Identifier(None,"add_snd"),(self.distr(expr))),abs_snd)
                
                if isinstance(abs_expr_fst, Const) and isinstance(abs_expr_snd, Const):
                    return Const(abs_expr_fst.v + abs_expr_snd.v), AbsConst(abs_expr_fst.v + abs_expr_snd.v)#return value should be the new one not the old one
                else:
                    return Add(abs_expr_fst.v,abs_expr_snd.v), AbsConst(abs_expr_fst.v + abs_expr_snd.v)
                    # return Apply(Identifier(None,"add"),[abs_expr_fst.v,abs_expr_snd.v]),AbsConst(abs_expr_fst.v + abs_expr_snd.v)
            case AbsMul(abs_fst,abs_snd):
                print("enter absmul in value expr")
                (fst, abs_expr_fst) = self.value_expr(Apply(Identifier(None, "mul_fst"), (self.distr(expr))), abs_fst)
                (snd, abs_expr_snd) = self.value_expr(Apply(Identifier(None, "mul_snd"), (self.distr(expr))), abs_snd)

                if isinstance(abs_expr_fst, Const) and isinstance(abs_expr_snd, Const):
                    return Const(abs_expr_fst.v * abs_expr_snd.v), AbsConst(abs_expr_fst.v * abs_expr_snd.v)
                else:
                    return Mul(abs_expr_fst.v,abs_expr_snd.v),AbsConst(abs_expr_fst.v * abs_expr_snd.v)
                    # return Apply(Identifier(None,"mul"),[abs_expr_fst.v,abs_expr_snd.v]),AbsConst(abs_expr_fst.v * abs_expr_snd.v)
            case AbsDiv(abs_fst,abs_snd):
                print("enter absdiv in value expr")
                (fst, abs_expr_fst) = self.value_expr(Apply(Identifier(None, "div_fst"), (self.distr(expr))), abs_fst)
                (snd, abs_expr_snd) = self.value_expr(Apply(Identifier(None, "div_snd"), (self.distr(expr))), abs_snd)


                if isinstance(abs_expr_fst, Const) and isinstance(abs_expr_snd, Const):
                    return Const(abs_fst.v / abs_snd.v), AbsConst(abs_fst.v / abs_snd.v)
                else:
                    return Div(abs_fst, abs_snd), AbsConst(abs_fst.v / abs_snd.v)
                    # return Apply(Identifier(None, "div"), [abs_fst, abs_snd]), AbsConst(abs_fst.v / abs_snd.v)

            case AbsIte(abs_cond, abs_true, abs_false):
                print("enter absitem in value expr")
                (cond, abs_expr_cond) = self.value_expr(Apply(Identifier(None, "ite_cond"), (self.distr(expr))), abs_cond)
                (true, abs_expr_true) = self.value_expr(Apply(Identifier(None, "ite_true"), (self.distr(expr))), abs_true)
                (false, abs_expr_false) = self.value_expr(Apply(Identifier(None, "ite_false"), (self.distr(expr))), abs_false)
                # cond = self.value_expr(abs_cond).v
                if isinstance(abs_expr_cond, bool):
                    if abs_expr_cond or abs_cond:
                        # return self.value_expr(true,abs_true)
                        return self.value_expr(Apply(Identifier(None, "ite_true"), (self.distr(true))), abs_true)
                    else:
                        # return self.value_expr(false, abs_false)
                        return self.value_expr(Apply(Identifier(None, "ite_false"), (self.distr(false))), abs_false)
                else:
                    raise ValueError(cond,abs_cond)
            case AbsEq(abs_fst, abs_snd):
                print("enter abseq in value expr")
                (fst, abs_expr_fst) = self.value_expr(Apply(Identifier(None, "eq_fst"), (self.distr(expr))), abs_fst)
                (snd, abs_expr_snd) = self.value_expr(Apply(Identifier(None, "eq_snd"), (self.distr(expr))), abs_snd)

                if abs_expr_fst and abs_expr_snd is Const:
                    return Const(abs_fst.v == abs_snd.v), AbsConst(abs_fst.v == abs_snd.v)
                else:
                    return Eq(abs_fst, abs_snd), AbsConst(abs_fst.v == abs_snd.v)
                    # return Apply(Identifier(None, "eq"), [abs_fst, abs_snd]), AbsConst(abs_fst.v == abs_snd.v)
            case AbsLt(abs_fst, abs_snd):
                print("enter abslt in value expr")
                (fst, abs_expr_fst) = Apply(Identifier(None, "value_expr"), (self.distr(expr), abs_fst))
                (snd, abs_expr_snd) = Apply(Identifier(None, "value_expr"), (self.distr(expr), abs_snd))

                if isinstance(abs_fst, UnkC) or isinstance(abs_snd, UnkC):
                    return AbsConst(UnkC())
                if fst and snd is Const:
                    return Const(fst.v < snd.v), AbsConst(abs_fst.v < abs_snd.v)
                else:
                    return Lt(fst, snd), AbsConst(abs_fst.v < abs_snd.v)
                    # return Apply(Identifier(None, "Lt"), (fst, snd)), AbsConst(abs_fst.v < abs_snd.v)
            case AbsLst(abs_es):
                print("enter abslst in value expr")

                (es,abs_expr_es) = Apply(Identifier(None, "value_expr"), (self.distr(expr), abs_es))
                return Lst(es),AbsConst([self.value_expr(e,e).v for e in abs_es])
                # return Apply(Identifier(None, "Lst"), (es)),AbsConst([self.value_expr(e,e).v for e in abs_es])
            case AbsPair(abs_fst, abs_snd):
                print("enter abspair in value expr")
                return Const((self.value_expr(abs_fst), self.value_expr(abs_snd))),AbsConst((self.value_expr(abs_fst).v, self.value_expr(abs_snd).v))
            case UnkE(_):
                return expr, AbsConst(UnkC())
            case TopE():
                return expr, AbsConst(UnkC())
            case (_):
                raise ValueError(expr)

    # Hybrid inference interface
    # Needs to be overridden by the implementation
    def assume(self, name: Optional[Identifier], annotation: Optional[Annotation], distribution: SymDistr[T]) -> \
    RandomVar[T]:
        raise NotImplementedError()

    def observe(self, rv: RandomVar[T], value: Const[T]) -> float:
        raise NotImplementedError()

    def value(self, rv: RandomVar[T]) -> Const[T]:
        raise NotImplementedError()

    def value_impl(self, rv: AbsRandomVar[T]) -> AbsConst[T]:
        raise NotImplementedError()

    def abs_value(self, rv: AbsRandomVar[T]) -> AbsConst[T]:
        # for pv in self.pv(rv):
        #     self.plan[pv] = DistrEnc.sample

        return self.value_impl(rv)


# Context class for tracking program variable assignments
class Context(object):
    def __init__(self, init={}) -> None:
        super().__init__()
        self.context: Dict[Identifier, SymExpr] = init

    def __getitem__(self, identifier: Identifier) -> Expr:
        # print("value in the context class",self.context.keys())
        # print("val of the keys in the context class ",self.context.values())
        return self.context[identifier]

    def __setitem__(self, identifier: Identifier, value: SymExpr) -> None:
        self.context[identifier] = value

    def __len__(self) -> int:
        return len(self.context)

    def __iter__(self):
        return iter(self.context)

    def __or__(self, other: 'Context') -> 'Context':
        new = deepcopy(self)
        for k, v in other.context.items():
            new.context[k] = v
        return new

    def __str__(self) -> str:
        return f"Context({', '.join(map(str, self.context.items()))})"

    def temp_var(self, name: str = "x") -> Identifier:
        i = 0
        while True:
            identifier = Identifier(None, f"{name}_{i}")
            if identifier not in self:
                return identifier
            i += 1

    def rename(self, old: AbsRandomVar, new: AbsRandomVar) -> None:
        for k, v in self.context.items():
            self.context[k] = v.rename(old, new)


# Abstract prob states
# TODO: Basically the same as concrete version, so can refractor somehow
class AbsContext(object):
    def __init__(self, init={}) -> None:
        super().__init__()
        self.context: Dict[Identifier, AbsSymExpr] = init

    def __getitem__(self, identifier: Identifier) -> AbsSymExpr:
        # print("in the abscontext", identifier)
        return self.context[identifier]

    def __setitem__(self, identifier: Identifier, value: AbsSymExpr) -> None:
        self.context[identifier] = value

    def __len__(self) -> int:
        return len(self.context)

    def __iter__(self):
        return iter(self.context)

    def __or__(self, other: 'AbsContext') -> 'AbsContext':
        new = deepcopy(self)
        for k, v in other.context.items():
            new.context[k] = v
        return new

    def __str__(self) -> str:

        return f"AbsContext({', '.join(map(str, self.context.items()))})"

    def temp_var(self, name: str = "x") -> Identifier:
        i = 0
        while True:
            identifier = Identifier(None, f"{name}_{i}")
            if identifier not in self:
                return identifier
            i += 1

    def rename(self, old: AbsRandomVar, new: AbsRandomVar) -> None:
        for k, v in self.context.items():
            self.context[k] = v.rename(old, new)


# Particle object used for the hybrid inference
# Maintains a symbolic state and an expression to simplfy
# It also has a score, and a flag to indicate if it is finished
class Particle(object):
    def __init__(
            self, cont: Expr[SymExpr],
            state: SymState = SymState(),
            score: float = 0.,
            finished: bool = False,
    ) -> None:
        super().__init__()
        self.cont: Expr[SymExpr] = cont
        self.state: SymState = state
        self.score: float = score  # logscale
        self.finished: bool = finished

    # Asserts that the particle is finished and returns the final expression
    # which must be a symbolic expression
    @property
    def final_expr(self) -> SymExpr:
        if self.finished:
            assert isinstance(self.cont, SymExpr)
            return self.cont
        else:
            raise ValueError(f'Particle not finished: {self}')

    # Update the particle with new values (or keep the same if None is passed)
    def update(self, cont: Optional[Expr] = None,
               state: Optional[SymState] = None,
               score: Optional[float] = None,
               finished: Optional[bool] = None) -> 'Particle':
        if cont is not None:
            self.cont = cont
        if state is not None:
            self.state = state
        if score is not None:
            self.score = score
        if finished is not None:
            self.finished = finished
        return self

    def __str__(self):
        return f"Particle({self.cont}, {self.state}, {self.score}, {self.finished})"

    def __copy__(self) -> 'Particle':
        return Particle(
            self.cont,
            copy(self.state),
            self.score,
            self.finished,
        )

    # Only for debugging
    def simplify(self) -> 'Particle':
        for rv in self.state.vars():
            self.state.set_entry(rv, distribution=self.state.eval_distr(self.state.get_entry(rv, 'distribution')))
        return self


# Mixture object is the final result object of the program
# it's a mixture distribution constructed from the particles
# class Mixture(object):
#     def __init__(self, mixture: List[Tuple[SymExpr, SymState, float]]):
#         super().__init__()
#         if len(mixture) == 0:
#             raise ValueError("Empty distribution")
#         self.mixture: List[Tuple[SymExpr, SymState, float]] = mixture
#
#     def __str__(self) -> str:
#         s = '\n'.join([f"{state.str_expr(expr)}: {prob}" for expr, state, prob in self.mixture])
#         return f"Mixture(\n{s}\n)"
#
#     def __repr__(self) -> str:
#         return self.__str__()
#
#     def __len__(self) -> int:
#         return len(self.mixture)
#
#     def __iter__(self):
#         return iter(self.mixture)
#
#     def __getitem__(self, index: int) -> Tuple[SymExpr, SymState, float]:
#         return self.mixture[index]
#
#     def __setitem__(self, index: int, value: Tuple[SymExpr, SymState, float]) -> None:
#         self.mixture[index] = value
#
#     @property
#     def is_pair_mixture(self) -> bool:
#         if len(self.mixture) == 0:
#             raise ValueError("No results")
#         return all(is_pair(expr) for expr, _, _ in self.mixture)
#
#     def get_pair_mixture(self) -> Tuple['Mixture', 'Mixture']:
#         if len(self.mixture) == 0:
#             raise ValueError("No results")
#         fst, snd = [], []
#         for expr, state, weight in self.mixture:
#             f, s = get_pair(expr)
#             fst.append((f, state, weight))
#             snd.append((s, state, weight))
#         return Mixture(fst), Mixture(snd)
#
#     @property
#     def is_lst_mixture(self) -> bool:
#         if len(self.mixture) == 0:
#             raise ValueError("No results")
#         return all(is_lst(expr) for expr, _, _ in self.mixture)
#
#     def get_lst_mixture(self) -> List['Mixture']:
#         if len(self.mixture) == 0:
#             raise ValueError("No results")
#         all_lsts = [(get_lst(expr), s, w) for expr, s, w in self.mixture]
#         max_len = max(len(lst) for lst, _, _ in all_lsts)
#         acc = []
#         for i in range(max_len):
#             acc.append([])
#             for lst, state, weight in all_lsts:
#                 if i < len(lst):
#                     acc[i].append((lst[i], state, weight))
#         return list(map(lambda x: Mixture(x), acc))
#
#     # Computes the expectation of the mixture
#     def mean(self) -> float:
#         if len(self.mixture) == 0:
#             raise ValueError("No results")
#         if len(self.mixture) == 1:
#             expr, state, _ = self.mixture[0]
#             return state.mean(expr)
#
#         unique_values = {}
#         for expr, state, weight in self.mixture:
#             v = state.mean(expr)
#             key = str(v)
#
#             if key not in unique_values:
#                 unique_values[key] = (v, weight)
#             else:
#                 unique_values[key] = (v, unique_values[key][1] + weight)
#
#         acc = 0.0
#         for v, weight in unique_values.values():
#             acc += weight * v
#         return acc


# A set of particles, returns a Mixture distribution
# class ProbState(object):
#     def __init__(self, n_particles: int, cont: Expr, method: type[SymState], seed: Optional[int] = None) -> None:
#         super().__init__()
#         self.seed = seed
#         self.rng = np.random.default_rng(seed=seed)
#         self.particles: List[Particle] = [
#             Particle(cont, method(seed=seed)) for i in range(n_particles)
#         ]
#
#     def __copy__(self) -> 'ProbState':
#         # doesn't really matter what goes in constructor, since it will be overwritten
#         new_state = ProbState(1, self.particles[0].cont, type(self.particles[0].state), seed=self.seed)
#         new_state.particles = [copy(p) for p in self.particles]
#         return new_state
#
#     def __getitem__(self, index: int) -> Particle:
#         return self.particles[index]
#
#     def __setitem__(self, index: int, value: Particle) -> None:
#         self.particles[index] = value
#
#     def __len__(self) -> int:
#         return len(self.particles)
#
#     def __iter__(self):
#         return iter(self.particles)
#
#     # Only for debugging
#     def simplify(self) -> 'ProbState':
#         for p in self.particles:
#             p.simplify()
#         return self
#
#     def __str__(self) -> str:
#         particles_indices = {}
#         for i, p in enumerate(self.particles):
#             key = str(p)
#             if key not in particles_indices:
#                 particles_indices[key] = [i]
#             else:
#                 particles_indices[key].append(i)
#
#         return "\n".join([f"{str(indices)}: {p}" for p, indices in particles_indices.items()])
#
#         # return "\n".join([f"{i}: {p}" for i, p in enumerate(self.particles)])
#
#     # Normalize the probabilities of the particles based on their scores
#     def normalized_probabilities(self) -> List[float]:
#         scores = np.array([p.score for p in self.particles])
#         if np.max(scores) == -np.inf:
#             warnings.warn("All particles have 0 weight")
#             scores = np.zeros(len(scores))
#             # raise RuntimeError("All particles have 0 weight")
#         probabilities = np.exp(scores - np.max(scores))
#         return list(probabilities / probabilities.sum())
#
#     # Create the mixture distribution of the particles
#     def mixture(self) -> Mixture:
#         probabilities = self.normalized_probabilities()
#         values = [p.final_expr for p in self.particles]
#         states = [p.state for p in self.particles]
#
#         unique_values = Mixture(list(zip(values, states, probabilities)))
#
#         return unique_values
#
#     # Compute the expectation of the result of the particles
#     # Pairs and Lists are handled recursively
#     def result(self) -> SymExpr:
#         mixture = self.mixture()
#
#         def _get_mean(res: Mixture) -> Const:
#             if res.is_pair_mixture:
#                 fst, snd = res.get_pair_mixture()
#                 fst = _get_mean(fst)
#                 snd = _get_mean(snd)
#                 return Const((fst.v, snd.v))
#             elif res.is_lst_mixture:
#                 lst = res.get_lst_mixture()
#                 return Const(list(map(lambda x: _get_mean(x).v, lst)))
#             else:
#                 return Const(res.mean())
#
#         return _get_mean(mixture)
#
#     @property
#     def finished(self) -> bool:
#         return all(p.finished for p in self.particles)
#
#     # Resamples its set of particles based on their scores, resetting the scores at the end.
#     def resample(self) -> 'ProbState':
#         particles = self.particles
#         probabilities = self.normalized_probabilities()
#         idxs = self.rng.choice(np.arange(len(particles)), size=len(particles),
#                                replace=True, p=probabilities)
#         used_idxs = set()
#         new_particles = []
#         for idx in idxs:
#             if idx not in used_idxs:
#                 new_particles.append(particles[idx])
#                 used_idxs.add(idx)
#             else:
#                 new_particles.append(copy(particles[idx]))
#         self.particles = new_particles
#         for p in self.particles:
#             p.update(score=0.)
#         return self


from distutils.dep_util import newer_group


from random import Random
from typing import Any, Set, Callable, Tuple, List
from copy import copy
from collections import deque

import siren.compiler.conjugate as conj
from siren.grammar import Expr, Apply
from siren.compiler.conjugate import is_affine
from siren.compiler.interface import *
from siren.analysis.ssi import *


# Implementation of semi-symbolic inference
# Adapted from https://github.com/IBM/probzelus/

# Custom exception for non-conjugate distributions
# Will be caught by the hoist function
class NonConjugate(Exception):
    def __init__(self, message, rv_nc: RandomVar, abs_rv_nc: AbsRandomVar):
        super().__init__(message)
        self.abs_rv_nc = abs_rv_nc
        self.rv_nc = rv_nc


class SSIState(SymState):
    ###
    # State entry:
    #   rv: (pv, distribution)
    ###

    def __str__(self):
        s = '\n\t'.join(map(str, self.state.items()))
        return f"SSIState(\n\t{s}\n)" if s else "SSIState()"

    def assume(self, name: Optional[Identifier], annotation: Optional[Annotation], distribution: Expr, abs_distribution:Optional[AbsSymExpr]) -> \
    tuple[[Callable[[Expr], Expr], Expr],Expr]:

        rv = self.new_var()

        # print("value of rv", rv)
        # if annotation is not None:
        #     if name is None:
        #         raise ValueError('Cannot annotate anonymous variable')
        # self.set_annotation(rv, annotation)

        # print("adcaefe", annotation)

            # return self.state.values(rv)
        # print("the value of the distribution in the assume for the sysmstate is3412341234", distribution)
        # print("distr in assume......................", distribution)
        # print("absdistr in assume....................",abs_distribution)
        # self.set_distr_forrv(rv, distribution)
        k = self.set_distr(rv, distribution)



        # print("distribution of k111111", k)
        # self.set_pv(rv, name)
        # print("name in ssi.......... ", name)
        # print("rv", rv)

        def r(res1:Expr):
            # print("kres val............",k(res1))
            return Let([rv], Apply(Identifier("Prob", "new_var"),[]),k(res1))



        #abstract
        rv1 = self.Abs_new_var()
        # pv1 = {name} if name is not None else set()
        # if annotation is not None:
        #     if name is None:
        #         raise ValueError('Cannot annotate anonymous variable')
        # self.set_annotation(rv1, annotation)
        #
        # self.register_expr_for_abs_rv(rv1, rv)
        # self.set_pv(rv1,pv1)
        print("abs rv1 in the abs_set_distr",rv1)
        print("distribution in the abs_set_distr",abs_distribution)
        self.Abs_set_distr(rv1,abs_distribution)


        return (r,rv,rv1)

    # def Abs_assume(self, name: Optional[Identifier], annotation: Optional[Annotation], distribution: AbsSymDistr[T]) -> \
    # AbsRandomVar[T]:
    #     rv = self.new_var()
    #     pv = {name} if name is not None else set()
    #     if annotation is not None:
    #         if name is None:
    #             raise ValueError('Cannot annotate anonymous variable')
    #     self.set_annotation(rv, annotation)
    #
    #     self.set_pv(rv, pv)
    #     self.set_distr(rv, distribution)
    #     return rv

    def observe(self, rv: RandomVar[T],abs_rv: AbsRandomVar[T], value: Const[T], abs_value: AbsConst[T]) -> Expr:
        # Try to score the random variable until it works
        def _observe() -> Callable[[Expr], Expr]:
            try:

                val = self.hoist_and_eval(rv,abs_rv)
                print("hoist in the observe", val)


                s_val = self.score(rv, abs_rv,value)
                # def new_val(res:Expr)->Expr:
                #     val2 = Let([],(Apply(Identifier("Prob", "observe_inner"), rv)),res)
                #     return val2

                def new_val1(res: Expr) -> Expr:
                    return val(s_val((res)))
                print("------------------------------print in observe---------------------------------")
                print(val)
                print(s_val)
                print(new_val1)
                print("---------------------------------------------------------------------------------")
                return new_val1
            except NonConjugate as e:
                # Sample whichever parent that is not conjugate
                self.value(e.rv_nc,e.abs_rv_nc)
                return _observe()

        s = _observe()
        print("values of the s", s)
        # Update the random variable with the observed value

        # self.intervene(rv, Delta(value, sampled=False))# changing into the apply and lower case of the delta
        print("rv  in observe", rv)
        print("v in observe",value)
        self.intervene(rv,(Apply(Identifier("", "delta"),[value])))
        self.Abs_intervene(abs_rv, AbsDelta(abs_value, sampled=False))
        return s



    def value(self, rv: Expr, abs_rv: AbsRandomVar) -> tuple[Callable[[Expr], Expr], Expr]:
        # Try to draw the random variable until it works
        # def _value() -> T:
            # try:

                # return self.draw(rv)
            # except NonConjugate as e:
            #     # Sample whichever parent that is not conjugate
            #     self.value(e.rv_nc)
            #     return _value()

        # v = _value()
        # print("the value of the rv is ",rv)
        # print("the value of the abs_rv is ",abs_rv)

        val = self.hoist_and_eval(rv, abs_rv)

        print("the value of the val in the value function", val)
        (k1,v,abs_v) = self.draw(rv,abs_rv)
        print("abs_v",abs_v)

        # d = self.Abs_distr(abs_rv)

        # Update the random variable with the sampled value

        # k2 =  self.intervene(rv, Delta(Const(v), sampled=True))
        # k2 = self.intervene(rv,Delta(Const(v), sampled=False))
        print("rv in value", rv)
        print("v in value", v)
        k2 = self.intervene(rv,Apply(Identifier(None, "delta"),[Const(v)]))
        self.Abs_intervene(abs_rv, AbsDelta(AbsConst(UnkC()), sampled=True))
        # print("the value in abs_intervene", self.Abs_intervene(abs_rv, AbsDelta(AbsConst(UnkC()), sampled=True)))

        # return Const(v)
        print("val in value ", val)
        print("k1 in value ", k1)
        print("k2 in value ", k2)
        def k(res1:Expr):
            return val(k1(k2(res1)))

        print("the value of the k  ", k)

        # print("the value absconst ", AbsConst(UnkC()))

        # return (k,Apply(Identifier(None, "mk_const"), (v)))
        return (k,v,AbsConst(UnkC()))

    # def value_impl(self, rv: AbsRandomVar[T]) -> AbsConst[T]:
    #     # fetch_rv_val = []
    #
    #     def _value() -> AbsSymDistr[T]:
    #         try:
    #             self.hoist_and_eval(rv)
    #             d = self.Abs_distr(rv)
    #             return d
    #         except NonConjugate as e:
    #             self.abs_value(e.rv_nc)
    #             return _value()
    #
    #     d = _value()
    #     self.Abs_intervene(rv, d)
    #     # fetch_rv_val.append(rv.rv)
    #     # print("val", fetch_rv_val)
    #
    #     return AbsConst(UnkC)


    def marginalize(self, rv: RandomVar,abs_rv:AbsRandomVar) -> None:
        # Try to hoist the random variable until it works
        try:
            self.hoist_and_eval(rv,abs_rv)
        except NonConjugate as e:
            # Sample whichever parent that is not conjugate
            self.value(e.rv_nc,e.abs_rv_nc)
            self.marginalize(rv,abs_rv)

    # def abs_marginalize(self,rv:AbsRandomVar) -> None:
    #     try:
    #         self.abs_hoist_and_eval(rv)
    #     except NonConjugate as e:
    #         self.abs_value(e.rv_nc)
    #         self.abs_marginalize(rv)
    ########################################################################

    def parents(self, rv: RandomVar) -> List[RandomVar]:
        print("parents val",rv)
        return self.distr(rv).rvs()

    def abs_parents(self,rv:AbsRandomVar)-> List[AbsRandomVar]:
        # print("abs_parents val ",rv)
        return self.Abs_distr(rv).rvs()

    def score(self, rv: RandomVar[T],abs_rv: AbsRandomVar[T], v: T) -> Callable[[Expr], Expr]:
        # RV has to be root
        # print("the value of the rv is.........................1222 ",rv)
        # print("the value of the abs_rv is.......................12222 ",abs_rv)

        # val = self.hoist_and_eval(rv,abs_rv)

        #return self.distr(rv).score(v)
        # def new_val(res:Expr) -> Expr:
        #     return val

        distrs  = self.distr(rv)

        # print("value of the expr in the score",self.Abs_distr(abs_rv))
        # def rest(res1:Expr)-> Expr:
        #     return
        print("---------------values in score---------------------")
        print(v)
        # print(val)
        print(distrs)
        # print(rest)

        def k(res1:Expr)-> Expr:

                if isinstance(v, bool):
                    value_code = f"const({v})"
                else:
                    value_code = v
                return Let([],Apply(Identifier("Prob","observe_inner"),[rv,value_code]),res1)


        print(k)
        print("-----------------------------------------------------")
        return k

    def draw(self, rv: Expr,abs_rv:AbsRandomVar) -> \
    tuple[Callable[[Expr], Expr], Expr]:
        # RV has to be root
        # self.hoist_and_eval(rv)
        # r = self.distr(rv)
        # calling the hoist func in the continuation and then returning the value of random varibale
        # (r,r1) = self.hoist(rv)
        self.counter += 1
        new_id = Identifier(None,f"S{self.counter}")
        new_val = Identifier(None, f"x{self.counter}")
        new_distr = self.lookup(abs_rv)
        #
        # print("what is the value of the rv in the draw func ", rv)
        # def r(res1:Expr):
        #
        #     # if new_distr is None:
        #     val = (Apply(Identifier("Prob", "get_distr"), rv))
        # #     else:
        #
        # #         val = self.lookup(abs_rv)
        #     s1 = Let([new_val], val, res1)
        #     # print("s111111 ",s1)
        #     return s1
        #


        def r1(res: Expr):
            # print("what is the value of the res in the draw function:",res)
            counter1 = self.counter
            val = (Apply(Identifier("Prob", "get_distr"), rv))

            s = LetRV(new_id, Annotation.sample.name, val,
                      res)
            print("s in draw func ", s)

            return s
            # return (Let([],Apply(Identifier(None,"hoist_and_eval"),(rv)),s))

            # return Let([], Apply(Identifier(None,"Sample"),(Apply(Identifier(None, "get_distr"), (rv)))),res)
        def r2(res1:Expr):
                return (r1(res1))
        # (r, r1) = self.hoist(rv)
        # return self.distr(rv).draw(self.rng). # check the draw(self.rng) what does this return
        # print("new id is", new_id)
        # print("r1 is ",r1)
        return r2,new_id,abs_rv

    # def tryhoist(self,rv: Expr) -> Expr:
    #     v = self.hoist(rv)
    #     return Let([], Apply(Identifier(None, "hoist"), (v)),self.draw(rv))

    def intervene(self, rv: Expr, v: Expr) -> Callable[[Expr], Expr]:
        print("intervene is ",v)
        print("rv",rv)
        return self.set_distr(rv,v)

    def Abs_intervene(self,rv:AbsRandomVar[T],v:AbsDelta[T])-> None:
        print("absrv in absintervene",rv)
        print("absintervene",v)
        self.Abs_set_distr(rv,v)

    # Check if the distribution of the given "expr" RV depends on rv, transitively checking
    # distributions of the RVs that the "expr" RV depends on
    # def rv_depends_on_transitive(self, expr: RandomVar, rv: RandomVar) -> bool:
    #     match self.distr(expr):
    #         case Normal(mu, var):
    #             return self.depends_on(mu, rv, True) or self.depends_on(var, rv, True)
    #         case Bernoulli(p):
    #             return self.depends_on(p, rv, True)
    #         case Beta(a, b):
    #             return self.depends_on(a, rv, True) or self.depends_on(b, rv, True)
    #         case Binomial(n, p):
    #             return self.depends_on(n, rv, True) or self.depends_on(p, rv, True)
    #         case BetaBinomial(n, a, b):
    #             return self.depends_on(n, rv, True) or self.depends_on(a, rv, True) or self.depends_on(b, rv, True)
    #         case NegativeBinomial(n, p):
    #             return self.depends_on(n, rv, True) or self.depends_on(p, rv, True)
    #         case Gamma(a, b):
    #             return self.depends_on(a, rv, True) or self.depends_on(b, rv, True)
    #         case Poisson(l):
    #             return self.depends_on(l, rv, True)
    #         case StudentT(mu, tau2, nu):
    #             return self.depends_on(mu, rv, True) or self.depends_on(tau2, rv, True) or self.depends_on(nu, rv, True)
    #         case Categorical(lower, upper, probs):
    #             return self.depends_on(lower, rv, True) or self.depends_on(upper, rv, True) or self.depends_on(probs,
    #                                                                                                            rv, True)
    #         case Delta(v, _):
    #             return self.depends_on(v, rv, True)
    #         case _:
    #             raise ValueError(self.distr(expr))
    #
    # # Check if the given expression depends on the given random variable, can be transitive or not
    # def depends_on(self, expr: SymExpr, rv: RandomVar, transitive: bool) -> bool:
    #     match expr:
    #         case Const(_):
    #             return False
    #         case RandomVar(_):
    #             if expr == rv:
    #                 return True
    #             else:
    #                 # If not transition, we're done
    #                 if not transitive:
    #                     return False
    #                 else:
    #                     # Otherwise, check if the distribution of expr depends on rv
    #                     return self.rv_depends_on_transitive(expr, rv)
    #         case Add(e1, e2):
    #             return self.depends_on(e1, rv, transitive) or self.depends_on(e2, rv, transitive)
    #         case Mul(e1, e2):
    #             return self.depends_on(e1, rv, transitive) or self.depends_on(e2, rv, transitive)
    #         case Div(e1, e2):
    #             return self.depends_on(e1, rv, transitive) or self.depends_on(e2, rv, transitive)
    #         case Ite(cond, true, false):
    #             return self.depends_on(cond, rv, transitive) or self.depends_on(true, rv,
    #                                                                             transitive) or self.depends_on(false,
    #                                                                                                            rv,
    #                                                                                                            transitive)
    #         case Eq(e1, e2):
    #             return self.depends_on(e1, rv, transitive) or self.depends_on(e2, rv, transitive)
    #         case Lt(e1, e2):
    #             return self.depends_on(e1, rv, transitive) or self.depends_on(e2, rv, transitive)
    #         case Lst(es):
    #             return any(self.depends_on(e, rv, transitive) for e in es)
    #         case Pair(e1, e2):
    #             return self.depends_on(e1, rv, transitive) or self.depends_on(e2, rv, transitive)
    #         case _:
    #             raise ValueError(expr)

    def rv_depends_on_transitive(self, expr: AbsRandomVar, rv: AbsRandomVar) -> bool:
        match self.Abs_distr(expr):
            case AbsNormal(mu, var):
                return self.depends_on(mu, rv, True) or self.depends_on(var, rv, True)
            case AbsBernoulli(p):
                return self.depends_on(p, rv, True)
            case AbsBeta(a, b):
                return self.depends_on(a, rv, True) or self.depends_on(b, rv, True)
            case AbsBinomial(n, p):
                return self.depends_on(n, rv, True) or self.depends_on(p, rv, True)
            case AbsBetaBinomial(n, a, b):
                return self.depends_on(n, rv, True) or self.depends_on(a, rv, True) or self.depends_on(b, rv, True)
            case AbsNegativeBinomial(n, p):
                return self.depends_on(n, rv, True) or self.depends_on(p, rv, True)
            case AbsGamma(a, b):
                return self.depends_on(a, rv, True) or self.depends_on(b, rv, True)
            case AbsPoisson(l):
                return self.depends_on(l, rv, True)
            case AbsStudentT(mu, tau2, nu):
                return self.depends_on(mu, rv, True) or self.depends_on(tau2, rv, True) or self.depends_on(nu, rv, True)
            case AbsCategorical(lower, upper, probs):
                return self.depends_on(lower, rv, True) or self.depends_on(upper, rv, True) or self.depends_on(probs,
                                                                                                               rv, True)
            case AbsDelta(v, _):
                return self.depends_on(v, rv, True)
            case UnkD(parents):
                return (rv in parents) or any(self.depends_on(p, rv, True) for p in parents)
            case TopD():
                return True
            case _:
                raise ValueError(self.Abs_distr(expr))

    def depends_on(self, expr: AbsSymExpr, rv: AbsRandomVar, transitive: bool) -> bool:
        match expr:
            case AbsConst(_):
                return False
            case AbsRandomVar(_):
                if expr == rv:
                    return True
                else:
                    if not transitive:
                        return False
                    else:
                        return self.rv_depends_on_transitive(expr, rv)
            case AbsAdd(e1, e2):
                return self.depends_on(e1, rv, transitive) or self.depends_on(e2, rv, transitive)
            case AbsMul(e1, e2):
                return self.depends_on(e1, rv, transitive) or self.depends_on(e2, rv, transitive)
            case AbsDiv(e1, e2):
                return self.depends_on(e1, rv, transitive) or self.depends_on(e2, rv, transitive)
            case AbsIte(cond, true, false):
                return self.depends_on(cond, rv, transitive) or self.depends_on(true, rv,
                                                                                transitive) or self.depends_on(false,
                                                                                                               rv,
                                                                                                               transitive)
            case AbsEq(e1, e2):
                return self.depends_on(e1, rv, transitive) or self.depends_on(e2, rv, transitive)
            case AbsLt(e1, e2):
                return self.depends_on(e1, rv, transitive) or self.depends_on(e2, rv, transitive)
            case AbsLst(es):
                return any(self.depends_on(e, rv, transitive) for e in es)
            case AbsPair(e1, e2):
                return self.depends_on(e1, rv, transitive) or self.depends_on(e2, rv, transitive)
            case UnkE(parents):
                return (rv in parents) or any(self.depends_on(p, rv, transitive) for p in parents)
            case TopE():
                return True
            case _:
                raise ValueError(expr)

    def hoist(self, rv: Expr,abs_rv: AbsRandomVar) -> Callable[[Expr], Expr]:
        # Topological sort of the random variables
        def _topo_sort(abs_rvs: List[AbsRandomVar]) -> List[AbsRandomVar]:
            sorted_nodes = []
            print("----------------hoist inside the topo sort------------------")

            def _visit(rv: AbsRandomVar) -> None:
                try:
                    parents = self.abs_parents(rv)
                    print("parents ", parents)
                except RecursionError:
                    raise RecursionError(rv)

                for parent in parents:
                    _visit(parent)

                if rv not in sorted_nodes:
                    sorted_nodes.append(rv)

            for rv in abs_rvs:
                _visit(rv)

            nodes = []
            print("nodes", nodes)
            for node in sorted_nodes:
                if node in abs_rvs:
                    nodes.append(node)

            return nodes
            # sorted_nodes = []
            # abs_sorted_nodes =[]
            # visited = set()
            # # print("val of nodes")
            # print("================")
            # print(abs_rvs)

            # def _visit(abs_rv:AbsRandomVar) -> None:
            #
            #     for abs_rv in abs_rvs:
            #         _visit(abs_rv)
            #     if abs_rv in visited:
            #         return visited.add(abs_rv)
            #
            #     abs_parents = self.abs_parents(abs_rv)
            #     for abs_parent in abs_parents:
            #         _visit(abs_parent)
            #
            #     abs_sorted_nodes.append(abs_rv)
            #     print("val of nodes in visit func")
            #     print("========================")
            #     print(abs_sorted_nodes)
            # #     # parents = self.parents(rv)
            # #     # try:
            # #     abs_parents = self.abs_parents(abs_rv)
            # #     # except RecursionError:
            # #     #     raise RecursionError(abs_rv)
            # #
            # #
            # #
            # #
            # #     for abs_parent in abs_parents:
            # #             _visit(abs_parent)
            # #
            # #
            # #     # if rv not in sorted_nodes:
            # #     #     sorted_nodes.append(rv)
            # #
            # #     if abs_rv not in abs_sorted_nodes:
            # #         abs_sorted_nodes.append(abs_rv)
            # #
            # #
            # #
            # # for abs_rv in abs_rvs:
            # #     _visit(abs_rv)
            # #
            # # # nodes = []
            # abs_nodes = []
            # # # for node in sorted_nodes:
            # #     # if node in rvs:
            # #     #     nodes.append(node)
            # #
            # for abs_node in abs_sorted_nodes:
            #     if abs_node in abs_rvs:
            #         abs_nodes.append(abs_node)
            # def _visit(abs_rv):  # did the changes in the loops of abs_rv and
            #     if abs_rv not in visited:
            #         return visited.add(abs_rv)
            #
            #     abs_parents = self.abs_parents(abs_rv)
            #     for abs_parent in abs_parents:
            #         _visit(abs_parent)
            #
            #     abs_sorted_nodes.append(abs_rv)
            #
            # # Visit all starting nodes
            # for abs_rv in abs_rvs:
            #     _visit(abs_rv)
            #
            # abs_nodes = []
            # for node in abs_sorted_nodes:
            #     if node in abs_rvs:
            #         abs_nodes.append(node)
            #
            # print("val of nodes in visit func")
            # print("========================")
            # print(abs_sorted_nodes)

            # return abs_nodes

        # Checks for circular dependencies
        def _can_swap(rv_par: AbsRandomVar, rv_child: AbsRandomVar) -> bool:
            def _has_other_deps_on_par(expr: AbsSymExpr) -> bool:
                match expr:
                    case AbsConst(_):
                        return False
                    case AbsRandomVar(_):
                        if expr == rv_par:
                            return False
                        else:
                            return self.rv_depends_on_transitive(expr, rv_par)
                    case AbsAdd(e1, e2):
                        return _has_other_deps_on_par(e1) or _has_other_deps_on_par(e2)
                    case AbsMul(e1, e2):
                        return _has_other_deps_on_par(e1) or _has_other_deps_on_par(e2)
                    case AbsDiv(e1, e2):
                        return _has_other_deps_on_par(e1) or _has_other_deps_on_par(e2)
                    case AbsIte(cond, true, false):
                        return _has_other_deps_on_par(cond) or _has_other_deps_on_par(true) or _has_other_deps_on_par(
                            false)
                    case AbsEq(e1, e2):
                        return _has_other_deps_on_par(e1) or _has_other_deps_on_par(e2)
                    case AbsLt(e1, e2):
                        return _has_other_deps_on_par(e1) or _has_other_deps_on_par(e2)
                    case AbsLst(es):
                        return any(_has_other_deps_on_par(e) for e in es)
                    case AbsPair(e1, e2):
                        return _has_other_deps_on_par(e1) or _has_other_deps_on_par(e2)
                    case UnkE(parents):
                        return any(self.rv_depends_on_transitive(p, rv_par) for p in parents)
                    case TopE():
                        return True
                    case _:
                        raise ValueError(expr)

            match self.Abs_distr(rv_child):
                case AbsNormal(mu, var):
                    return (self.depends_on(mu, rv_par, False) or self.depends_on(var, rv_par, False)) \
                        and (not _has_other_deps_on_par(mu)) and (not _has_other_deps_on_par(var))
                case AbsBernoulli(p):
                    return self.depends_on(p, rv_par, False) and (not _has_other_deps_on_par(p))
                case AbsBeta(a, b):
                    return (self.depends_on(a, rv_par, False) or self.depends_on(b, rv_par, False)) \
                        and (not _has_other_deps_on_par(a)) and (not _has_other_deps_on_par(b))
                case AbsBinomial(n, p):
                    return (self.depends_on(n, rv_par, False) or self.depends_on(p, rv_par, False)) \
                        and (not _has_other_deps_on_par(n)) and (not _has_other_deps_on_par(p))
                case AbsBetaBinomial(n, a, b):
                    return (self.depends_on(n, rv_par, False) or self.depends_on(a, rv_par, False) or self.depends_on(b,
                                                                                                                      rv_par,
                                                                                                                      False)) \
                        and (not _has_other_deps_on_par(n)) and (not _has_other_deps_on_par(a)) and (
                            not _has_other_deps_on_par(b))
                case AbsNegativeBinomial(n, p):
                    return (self.depends_on(n, rv_par, False) or self.depends_on(p, rv_par, False)) \
                        and (not _has_other_deps_on_par(n)) and (not _has_other_deps_on_par(p))
                case AbsGamma(a, b):
                    return (self.depends_on(a, rv_par, False) or self.depends_on(b, rv_par, False)) \
                        and (not _has_other_deps_on_par(a)) and (not _has_other_deps_on_par(b))
                case AbsPoisson(l):
                    return self.depends_on(l, rv_par, False) and (not _has_other_deps_on_par(l))
                case AbsStudentT(mu, tau2, nu):
                    return (self.depends_on(mu, rv_par, False) or self.depends_on(tau2, rv_par,
                                                                                  False) or self.depends_on(nu, rv_par,
                                                                                                            False)) \
                        and (not _has_other_deps_on_par(mu)) and (not _has_other_deps_on_par(tau2)) and (
                            not _has_other_deps_on_par(nu))
                case AbsCategorical(lower, upper, probs):
                    return (self.depends_on(lower, rv_par, False) or self.depends_on(upper, rv_par,
                                                                                     False) or self.depends_on(probs,
                                                                                                               rv_par,
                                                                                                               False)) \
                        and (not _has_other_deps_on_par(lower)) and (not _has_other_deps_on_par(upper)) and (
                            not _has_other_deps_on_par(probs))
                case AbsDelta(v, _):
                    return self.depends_on(v, rv_par, False) and (not _has_other_deps_on_par(v))
                case UnkD(parents):
                    return not (rv_par in parents)
                case TopD():
                    return False
                case _:
                    raise ValueError(self.Abs_distr(rv_child))



        # def _can_swap(rv_par: RandomVar, rv_child: RandomVar) -> bool:
        #     def _has_other_deps_on_par(expr: SymExpr) -> bool:
        #         match expr:
        #             case Const(_):
        #                 return False
        #             case RandomVar(_):
        #                 if expr == rv_par:
        #                     return False
        #                 else:
        #                     return self.rv_depends_on_transitive(expr, rv_par)
        #             case Add(e1, e2):
        #                 return _has_other_deps_on_par(e1) or _has_other_deps_on_par(e2)
        #             case Mul(e1, e2):
        #                 return _has_other_deps_on_par(e1) or _has_other_deps_on_par(e2)
        #             case Div(e1, e2):
        #                 return _has_other_deps_on_par(e1) or _has_other_deps_on_par(e2)
        #             case Ite(cond, true, false):
        #                 return _has_other_deps_on_par(cond) or _has_other_deps_on_par(true) or _has_other_deps_on_par(
        #                     false)
        #             case Eq(e1, e2):
        #                 return _has_other_deps_on_par(e1) or _has_other_deps_on_par(e2)
        #             case Lt(e1, e2):
        #                 return _has_other_deps_on_par(e1) or _has_other_deps_on_par(e2)
        #             case Lst(es):
        #                 return any(_has_other_deps_on_par(e) for e in es)
        #             case Pair(e1, e2):
        #                 return _has_other_deps_on_par(e1) or _has_other_deps_on_par(e2)
        #             case _:
        #                 raise ValueError(expr)
        #
        #     match self.distr(rv_child):
        #         case Normal(mu, var):
        #             return (self.depends_on(mu, rv_par, False) or self.depends_on(var, rv_par, False)) \
        #                 and (not _has_other_deps_on_par(mu)) and (not _has_other_deps_on_par(var))
        #         case Bernoulli(p):
        #             return self.depends_on(p, rv_par, False) and (not _has_other_deps_on_par(p))
        #         case Beta(a, b):
        #             return (self.depends_on(a, rv_par, False) or self.depends_on(b, rv_par, False)) \
        #                 and (not _has_other_deps_on_par(a)) and (not _has_other_deps_on_par(b))
        #         case Binomial(n, p):
        #             return (self.depends_on(n, rv_par, False) or self.depends_on(p, rv_par, False)) \
        #                 and (not _has_other_deps_on_par(n)) and (not _has_other_deps_on_par(p))
        #         case BetaBinomial(n, a, b):
        #             return (self.depends_on(n, rv_par, False) or self.depends_on(a, rv_par, False) or self.depends_on(b,
        #                                                                                                               rv_par,
        #                                                                                                               False)) \
        #                 and (not _has_other_deps_on_par(n)) and (not _has_other_deps_on_par(a)) and (
        #                     not _has_other_deps_on_par(b))
        #         case NegativeBinomial(n, p):
        #             return (self.depends_on(n, rv_par, False) or self.depends_on(p, rv_par, False)) \
        #                 and (not _has_other_deps_on_par(n)) and (not _has_other_deps_on_par(p))
        #         case Gamma(a, b):
        #             return (self.depends_on(a, rv_par, False) or self.depends_on(b, rv_par, False)) \
        #                 and (not _has_other_deps_on_par(a)) and (not _has_other_deps_on_par(b))
        #         case Poisson(l):
        #             return self.depends_on(l, rv_par, False) and (not _has_other_deps_on_par(l))
        #         case StudentT(mu, tau2, nu):
        #             return (self.depends_on(mu, rv_par, False) or self.depends_on(tau2, rv_par,
        #                                                                           False) or self.depends_on(nu, rv_par,
        #                                                                                                     False)) \
        #                 and (not _has_other_deps_on_par(mu)) and (not _has_other_deps_on_par(tau2)) and (
        #                     not _has_other_deps_on_par(nu))
        #         case Categorical(lower, upper, probs):
        #             return (self.depends_on(lower, rv_par, False) or self.depends_on(upper, rv_par,
        #                                                                              False) or self.depends_on(probs,
        #                                                                                                        rv_par,
        #                                                                                                        False)) \
        #                 and (not _has_other_deps_on_par(lower)) and (not _has_other_deps_on_par(upper)) and (
        #                     not _has_other_deps_on_par(probs))
        #         case Delta(v, _):
        #             return self.depends_on(v, rv_par, False) and (not _has_other_deps_on_par(v))
        #         case _:
        #             raise ValueError(self.distr(rv_child))

        # Swaps the distributions of the two random variables
        # def _swap(rv_par: RandomVar, rv_child: RandomVar) -> bool:
        #     def _update(marginal_posterior: Optional[Tuple[SymDistr, SymDistr]]) -> bool:
        #         if marginal_posterior is None:
        #             return False
        #
        #         marginal, posterior = marginal_posterior
        #         self.set_distr(rv_par, posterior)
        #         self.set_distr(rv_child, marginal)
        #         return True
        #
        #     # Check if the distributions are conjugate
        #     match self.distr(rv_par), self.distr(rv_child):
        #         case Normal(_), Normal(_):
        #             if _update(conj.gaussian_conjugate(self, rv_par, rv_child)):
        #                 return True
        #             else:
        #                 return _update(conj.normal_inverse_gamma_normal_conjugate(self, rv_par, rv_child))
        #         case Bernoulli(_), Bernoulli(_):
        #             return _update(conj.bernoulli_conjugate(self, rv_par, rv_child))
        #         case Beta(_), Bernoulli(_):
        #             return _update(conj.beta_bernoulli_conjugate(self, rv_par, rv_child))
        #         case Beta(_), Binomial(_):
        #             return _update(conj.beta_binomial_conjugate(self, rv_par, rv_child))
        #         case Gamma(_), Poisson(_):
        #             return _update(conj.gamma_poisson_conjugate(self, rv_par, rv_child))
        #         case Gamma(_), Normal(_):
        #             return _update(conj.gamma_normal_conjugate(self, rv_par, rv_child))
        #         case _:
        #             return False

        def _swap(rv_par:Expr,rv_child: Expr,abs_rv_par: AbsRandomVar, abs_rv_child: AbsRandomVar) -> tuple[
            Callable[[Expr], Expr], bool]:
            print("------------In swap func ------------------")
            print("rv_par", rv_par)
            print("rv_child", rv_child)
            print("abs_rv_par", abs_rv_par)
            print("abs_rv_child", abs_rv_child)
            
            print("--------------")
            def _update(marginal_posterior:Optional[Tuple[Expr, Expr]] ,abs_marginal_posterior: Optional[Tuple[AbsSymDistr, AbsSymDistr]]) -> \
                    tuple[Callable[[Expr], Expr], bool]:
                print("abs_marginal_posterior", abs_marginal_posterior)
                print("---------inside update swap----------")
                if abs_marginal_posterior is None:
                    return False,False

                abs_marginal, abs_posterior = abs_marginal_posterior
                marginal, posterior = marginal_posterior
                print("-------start swap-----")
                print("marginal", marginal)
                print("posterior", posterior)
                print("marginal posterior", marginal_posterior)
                print("abs_posterior", abs_posterior)
                print("abs_marginal", abs_marginal)
                print(self.set_distr(rv_par, posterior))
                print("----end swap-------")
                self.counter += 1
                posterior_tmp = Identifier(
                    None, f"__siren_update_posterior_{self.counter}"
                )
                marginal_tmp = Identifier(
                    None, f"__siren_update_marginal_{self.counter}"
                )

                par_set_distr = self.set_distr(rv_par, posterior_tmp)
                child_set_distr = self.set_distr(rv_child, marginal_tmp)

                self.Abs_set_distr(abs_rv_par, abs_posterior)
                self.Abs_set_distr(abs_rv_child, abs_marginal)

                def par_distr(res: Expr):
                    return Let(
                        [posterior_tmp], posterior,
                        Let(
                            [marginal_tmp], marginal,
                            par_set_distr(child_set_distr(res)),
                        ),
                    )

                return par_distr, True
                # par_set_distr = self.set_distr(rv_par, posterior)
                # child_set_distr = self.set_distr(rv_child, marginal)
                # self.Abs_set_distr(abs_rv_par, abs_posterior)
                # self.Abs_set_distr(abs_rv_child, abs_marginal)
                # def par_distr(res:Expr):

                #     return par_set_distr(child_set_distr(res))

                # # def child_distr(res1:Expr)->Expr:
                # #     return child_set_distr(res1)


                # return par_distr,True

            # print("=========")
            # print("", self.Abs_distr(abs_rv_par))
            # print("",self.Abs_distr(abs_rv_child))
            #
            # print("==========")
            match self.Abs_distr(abs_rv_par), self.Abs_distr(abs_rv_child):
            

                case AbsNormal(_), AbsNormal(_):
                    print("-------------inside match case in swap after the update function------------------")
                    print("in swap parent", rv_par)
                    print("in swap child",rv_child)

                    print("normal in swap")
                    self.counter += 1
                    new_id_par = Identifier(None, f"x{self.counter}")
                    new_id_child = Identifier(None, f"y{self.counter}")

                    # def r1(res:Expr):
                    #     val = Apply(Identifier("Prob", "gaussian_posterior"), [rv_par, rv_child])
                    #     s1  = Let([new_id_par],val,res)
                    #     return s1

                    # def r2(res1:Expr):
                    #     val1 = Apply(Identifier("Prob","gaussian_marginal"),[rv_par,rv_child])
                    #     s2  = Let([new_id_child],val1,res1)
                    #     return s2


                    marginal,posterior,abs_marginal,abs_posterior = conj.gaussian_conjugate(self,rv_par, rv_child, abs_rv_par, abs_rv_child)

                    # if marginal == "SWAP" or posterior == "SWAP":
                    #     val = (r2,r1)
                    # else:
                    val = (marginal,posterior)


                    # res = .is_affine(rv_par,rv_child,abs_rv_par,abs_rv_child)

                    #seperate the identifier gaussian and the rv_par and the rv_child in two seperable so it would be easy in update
                    [tmp1,abs_tmp] = _update(val, (abs_marginal,abs_posterior))
                    # print("what is tmp1",tmp1)
                    def r(rest:Expr)->Expr:
                        return r1(r2(tmp1(rest)))
                    if abs_tmp:
                        return tmp1,True
                    else:
                        invmarginal,invposterior,abs_invmarginal,abs_invposterior = conj.normal_inverse_gamma_normal_conjugate(self,rv_par, rv_child, abs_rv_par, abs_rv_child)
                        val1 = (invmarginal,invposterior)
                        [temp_val, abs_temp_val] = _update(val1, (abs_invmarginal,abs_invposterior))
                        
                        return temp_val, abs_temp_val
                case AbsBernoulli(_), AbsBernoulli(_):
                    print("bernoulli in swap")
                    print("rv_par", rv_par)
                    print("rv_child", rv_child)
                    print("abs_rv_par", abs_rv_par)
                    print("abs_rv_child", abs_rv_child)
                    marginal, posterior, abs_marginal, abs_posterior = conj.bernoulli_conjugate(self, rv_par, rv_child, abs_rv_par, abs_rv_child)
                    val1 = (marginal,posterior)
                    [temp_val, abs_temp_val] = _update(val1, (abs_marginal,abs_posterior))
                    return temp_val, abs_temp_val
                case AbsBeta(_), AbsBernoulli(_):
                    print("beta bernoulli in swap")
                    print("rv_par", rv_par)
                    print("rv_child", rv_child)
                    print("abs_rv_par", abs_rv_par)
                    print("abs_rv_child", abs_rv_child)
                    marginal, posterior, abs_marginal, abs_posterior = conj.beta_bernoulli_conjugate(self, rv_par, rv_child, abs_rv_par, abs_rv_child)
                    val1 = (marginal,posterior)
                    [temp_val, abs_temp_val] = _update(val1, (abs_marginal,abs_posterior))
                    return temp_val, abs_temp_val
                case AbsBeta(_), AbsBinomial(_):
                    print("beta binomial in swap")
                    marginal, posterior, abs_marginal, abs_posterior = conj.beta_binomial_conjugate(self, rv_par, rv_child, abs_rv_par, abs_rv_child)
                    val1 = (marginal,posterior)
                    [temp_val, abs_temp_val] = _update(val1, (abs_marginal,abs_posterior))
                    return temp_val, abs_temp_val
                case AbsGamma(_), AbsPoisson(_):
                    print("gamma poisson in swap")
                    marginal, posterior, abs_marginal, abs_posterior = conj.gamma_poisson_conjugate(self, rv_par, rv_child, abs_rv_par, abs_rv_child)
                    val1 = (marginal,posterior)
                    [temp_val, abs_temp_val] = _update(val1, (abs_marginal,abs_posterior))
                    return temp_val, abs_temp_val
                case AbsGamma(_), AbsNormal(_):
                    print("gamma gamma in swap")
                    marginal, posterior, abs_marginal, abs_posterior = conj.gamma_normal_conjugate(self, rv_par, rv_child, abs_rv_par, abs_rv_child)
                    val1 = (marginal,posterior)
                    [temp_val, abs_temp_val] = _update(val1, (abs_marginal,abs_posterior))
                    return temp_val, abs_temp_val
                case UnkD(_), _:
                    print("UnkD in swap")
                    self.set_dynamic(abs_rv_par)
                    return (Identifier(None,"UnkD")),False
                case TopD(), _:
                    print("TopD in swap")
                    self.set_dynamic(abs_rv_par)
                    return (Identifier(None,"TopD")),False
                case _:
                    # print("false case in swap")
                    # print("what is parent", self.Abs_distr(abs_rv_par))
                    # print("what is child",self.Abs_distr(abs_rv_child))
                    return (Identifier(None,"false")),False

        # def _swap(rv_par: AbsRandomVar, rv_child: AbsRandomVar) -> bool:
        #     def _update(marginal_posterior: Optional[Tuple[AbsSymDistr, AbsSymDistr]]) -> bool:
        #         if marginal_posterior is None:
        #             return False
        #
        #         marginal, posterior = marginal_posterior
        #         self.Abs_set_distr(rv_par, posterior)
        #         self.Abs_set_distr(rv_child, marginal)
        #         return True
        #
        #     match self.Abs_distr(rv_par), self.Abs_distr(rv_child):
        #         case AbsNormal(_), AbsNormal(_):
        #             if _update(conj.gaussian_conjugate(self, rv_par, rv_child)):
        #                 return True
        #             else:
        #                 return _update(conj.normal_inverse_gamma_normal_conjugate(self, rv_par, rv_child))
        #         case AbsBernoulli(_), AbsBernoulli(_):
        #             return _update(conj.bernoulli_conjugate(self, rv_par, rv_child))
        #         case AbsBeta(_), AbsBernoulli(_):

        #             return _update(conj.beta_bernoulli_conjugate(self, rv_par, rv_child))
        #         case AbsBeta(_), AbsBinomial(_):
        #             return _update(conj.beta_binomial_conjugate(self, rv_par, rv_child))
        #         case AbsGamma(_), AbsPoisson(_):
        #             return _update(conj.gamma_poisson_conjugate(self, rv_par, rv_child))
        #         case AbsGamma(_), AbsNormal(_):
        #             return _update(conj.gamma_normal_conjugate(self, rv_par, rv_child))
        #         case UnkD(_), _:
        #             self.set_dynamic(rv_par)
        #             return False
        #         case TopD(), _:
        #             self.set_dynamic(rv_par)
        #             return False
        #         case _:
        #             return False

        def _hoist_inner(rv:Expr,abs_rv: AbsRandomVar,ghost_roots: Set[AbsRandomVar]) -> Callable[[Expr], Expr]:
            # Hoist parents
            print("--------------------------------------start-hoist inner----------------------------------------------")
            print("rv.....",rv)
            print("abs_rv.....",abs_rv)
            old_abs_parents = self.abs_parents(abs_rv)
            print("old_abs_parents",old_abs_parents)
            abs_parents = _topo_sort(self.abs_parents(abs_rv))
            print("abs_parents",abs_parents)

            k1_update = []
            k1_update.append(lambda x: x)

            k_update =[]
            k_update.append(lambda x: x)
            print("value of abs_parents")
            print("-------------")
            print(abs_parents)
            print("----------")


            # parents = _topo_sort(self.parents(abs_rv))
            ghost_roots1 = copy(ghost_roots)
            # # Hoist parent in order, skipping the ghost roots
            # for idx in parents:
            #     _hoist_inner(Apply(Identifier(None,"get_par"),(abs_rv,idx)),abs_rv.parents(idx))
            for value in abs_parents:
                idx = old_abs_parents.index(value)
                if value in ghost_roots1:
                    continue
                if value not in ghost_roots1:

                    # new_idx = abs_parents.index(value)


                    print("value in hoist_inner",value)
                    print("idx in hoist inner",idx)
                    print("old_abs_parents with idx", old_abs_parents[idx])
                    # _hoist_inner(Apply(Identifier(None,"get_par"),(abs_rv,idx)),abs_rv.abs_parents[idx])

                    k = _hoist_inner(Apply(Identifier("Prob","get_par"),[rv,Const(idx)]), old_abs_parents[idx], ghost_roots1)#did changes here
                    print("k in hoist inner",k)


                    ghost_roots1.add(value)
                # tmp4 = k_update[-1]
                # def make_k_rest(prev, kk):
                #     def k_rest(rest1: Expr) -> Expr:
                #         return prev(kk(rest1))
                #     return k_rest
                # k_update[-1] = make_k_rest(tmp4, k)

                tmp4 = k_update[-1]
                def k_rest(rest1:Expr, previous=tmp4, current=k)->Expr:
                    # print("k_rest in func ",tmp(k(rest1)))
                    return previous(current(rest1))
                k_update[-1] = k_rest
                print("k_update",k_update)
                # print("k_rest value in hoist_inner",k_rest)


            #
            # # Hoist current node
            for parent in abs_parents[::-1]:
                if parent in ghost_roots:
                    continue
                print("abs",abs_parents)
                idx = old_abs_parents.index(parent)
                print("idx",idx)
                print("parent",parent)
                print("old parents ",old_abs_parents[idx])
                if parent not in ghost_roots:

                    print("val of old idx and abs parents")
                    print("---------------------")
                    
                    print(abs_parents)
                    print(parent)
                    print(idx)
                    print(abs_parents[idx])
                    print(old_abs_parents[idx])
                    print(abs_rv)
                    print(ghost_roots)
                    print("----------------------")
                    if not _can_swap(parent, abs_rv): # changes here as list indices must be integers or slices, not AbsRandomVar
                        raise ValueError(f'Cannot swap {parent} and {abs_rv}')

                    print("rv is ",rv)
                    print("what is the idx",idx)

                    val_rv_par = Apply(Identifier("Prob","get_par"),[rv,Const(idx)])
                    print("val rv of the parent",val_rv_par)
                    print("========before swap")
                    print("val_rv_par", val_rv_par)
                    print("rv" , rv)
                    print("rv distr", self.distr(rv))
                    print("parent",parent)
                    print("parent distr", self.Abs_distr(parent))
                    print("abs_rv",abs_rv)
                    print("abs_rv distr", self.Abs_distr(abs_rv))
                    print("=========")

                    [conc_swap,abs_swap] = _swap(val_rv_par,rv, parent,abs_rv)
                    print("========after swap")
                    print("parent", parent)
                    print("conc_swap", conc_swap)
                    print("abs_swap", abs_swap)
                    print("parent distr", self.Abs_distr(parent))
                    print("abs_rv", abs_rv)
                    print("abs_rv distr", self.Abs_distr(abs_rv))
                    print("=========")
                    # print("1111111111111111111",conc_swap)
                    if not abs_swap:
                        raise NonConjugate(f'Nonconjugate {abs_parents[idx]} and {abs_rv}', abs_parents[idx], abs_rv)
                    else:
                        def k1(rest1:Expr, swap=conc_swap)->Expr:
                            # print("----------------in hoist inner func before swap---------")
                            # print("val_rv_par", val_rv_par)
                            # print("rv",rv)
                            # print("parent",parent)
                            # print(conc_swap)
                            # print("abs_rv",abs_rv)
                            # print("abs_distribution", self.Abs_distr(abs_rv))
                            # print("distrbution", self.Abs_distr(parent))
                            # print("------------------")
                            return swap(rest1)
                            # return Let([],conc_swap,rest1)
                            # return Let([],Apply(Identifier("Prob", "swap"), [Apply(Identifier("Prob","get_par"),[rv,Const(idx)]), rv]), rest1)


                # tmp3 = k1_update[-1]
                # def make_k3(prev, swap_k):
                #     def k3(rest: Expr) -> Expr:
                #         return prev(swap_k(rest))
                #     return k3

                # k1_update.append(make_k3(k1_update[-1], k1))

                tmp3 = k1_update[-1]
                def k3(rest : Expr, previous=tmp3, current=k1)->Expr:
                    return previous(current(rest))

                # print("Before k1_update replaced")
                # print("=========================")
                # print(k1_update[-1](None))
                # print(k1_update)
                # print("=========================")
                k1_update.append(k3)
                # print("After k1_update replaced")
                # print("=========================")
                # print(k1_update)
                # print(k1_update[-1](None))
                # print(k1_update)
                # print("=========================")

            tmp1 = k1_update[-1]
            tmp2 = k_update[-1]
            def k4(rest : Expr)->Expr:
                return tmp2(tmp1(rest))

            # print("------------------------------")
            # print("before k4 return value ")
            # print(k4(None))
            # print("------------------------------")
            return k4

        print("--------------------------------------end-hoist inner----------------------------------------------")
        #swap should be in recursive call
        tmp = _hoist_inner(rv, abs_rv, set())

        # _hoist_inner(rv,abs_rv, set())
        print("---------------hoist inner return func----------------")
        print(tmp(None))
        print("----------------------------------")



        return tmp




    def hoist_and_eval(self, rv: Expr, abs_rv: AbsRandomVar) -> tuple[[Callable[[Expr], Expr], Expr],Expr]:
        # k = self.set_distr(rv, self.eval(rv,abs_rv))#apply(identifier(eval(rv))))
        print("bas_rv in the hoist and eval", abs_rv)
        if isinstance(self.Abs_distr(abs_rv), AbsNormal):
            self.Abs_set_distr(
                abs_rv,
                self.abs_eval_distr(self.Abs_distr(abs_rv)),
            )
        eval_val,abs_eval_val = self.eval_distr(self.distr(rv), self.Abs_distr(abs_rv))
        print("eval_val in hoist and eval......",eval_val)
        print("abs_eval_val in hoist and eval......",abs_eval_val)
        k = self.set_distr(rv,  eval_val)#get distr should
        # (k,k_abs) = k
        # self.Abs_set_distr(abs_rv, self.Abs_eval(abs_rv))
        self.Abs_set_distr(abs_rv, abs_eval_val)
        # self.Abs_set_distr(abs_rv, Apply(Identifier(None, "eval"), (rv, abs_rv)))
        # print('1')
        k4 = self.hoist(rv,abs_rv)#apply dynamicall
        # print("-----------------------k4 value in hoist and eval------------------")
        # print(k4(None))
        # print("--------------------------------------------------------")
        eval_val1,abs_eval_val1 = self.eval_distr(self.distr(rv), self.Abs_distr(abs_rv))
        # print("------------------------------------------")
        # print("absrv in hoist and eval", abs_rv)
        # print("abs_eval in hoist and eval", abs_eval_val)
        # print("------------------------------------------")
        # k4 = self.set_eval(rv)
        k2 = self.set_distr(rv, eval_val1)
        self.Abs_set_distr(abs_rv, abs_eval_val1)
        # self.Abs_set_distr(abs_rv, self.abs_eval_distr(self.Abs_distr(rv)))
        # self.Abs_set_distr(abs_rv, Apply(Identifier(None, "eval"), (rv, abs_rv)))
        def kret_val(val1: Expr) -> Expr:
            # print("Inside kret_val")
            # print("===============")
            # print(val1)
            # print(k2(val1))
            # print(k4(k2(val1)))
            # print(k(k4(k2(val1))))
            # print("===============")
            return k(k4(k2(val1)))

        return kret_val

# all the values in the hoist_and_eval of the set_distr gets ignored so store the values of the set_distr and other functions for returning that values return the value in the continuation (k,expr,abs_epxr) expr and abs_expr can be ignored just return the continaution and also look at the assume to work simialrly
# topo_sort should be changed as it should return the indexes as it gets called in the hoist_inner and so it would get the each index from the topo_sort when called for the random var in the hoist_inner
    # def abs_hoist_and_eval(self,rv:AbsRandomVar) -> None:
    #     self.Abs_set_distr(rv, self.abs_eval_distr(self.Abs_distr(rv)))
    #     self.hoist(rv)
    #     self.Abs_set_distr(rv, self.abs_eval_distr(self.Abs_distr(rv)))

    def set_dynamic(self, rv: AbsRandomVar) -> None:
        super().set_dynamic(rv)

        for rv_par in self.parents(rv):
            self.set_dynamic(rv_par)

# from siren.grammar import Normal, Const, Mul
from siren.grammar import *
from typing import List
from siren.evaluate import observe as siren_observe
from siren.inference.interface import SymState
from siren.inference.ssi import SSIState

class Particle(object):
  def __init__(
    self, cont,
    state: SymState = SymState(),
    score: float = 0.,
    finished: bool = False,
  ) -> None:
    super().__init__()
    self.cont = cont
    self.state: SymState = state
    self.score: float = score  # logscale
    self.finished: bool = finished

def run(cont, n_particles):
  particles = [Particle(cont, SSIState(), 0, False) for i in range(n_particles)]
  for i, particle in enumerate(particles):
    res = particles[i].cont(particles[i])
    particles[i].cont = res
    assert(particles[i].finished)
  return particles

# gaussian(Num, Num)
def gaussian(mu, var):
  return Normal(mu, var)

def bernoulli(prob):
  return Bernoulli(prob)

def delta(x):
  return Delta(x)

# const(Num)
def const(n):
  return Const(n)

# times(Num, Num)
def times(n1, n2):
  return Mul.make(n1, n2)


def div(n1, n2):
  return Div.make(n1, n2)


def add(n1, n2):
  return Add.make(n1, n2)


def sub(left, right):
  return Add.make(left, Mul.make(Const(-1), right))
  
# Compatibility name emitted by compiled programs.
def ex_mul(n1, n2):
  return times(n1, n2)

# def div(n1,n2):
#   return Div(n1,n2)
#
# def add(n1,n2):
#   return Add(n1,n2)

def beta(val1,val2):
  return Beta(val1,val2)

def cons(val1, val2):
  # val2 is a list built by lst()/cons()
  return [val1] + list(val2)

def eq(left, right):
    return Eq.make(left, right)

def lt(left, right):
    return Lt.make(left, right)

def ifelse(condition, true_value, false_value):
    return Ite.make(condition, true_value, false_value)


def subst_rv(expr,rv,value):
  return expr.subst_rv(rv,value)

def pair(a, b):
  # Generated tuple patterns are flat: pair(a, pair(b, c)) is (a, b, c).
  # Lists remain individual fields, and left-hand tuple fields stay grouped.
  if isinstance(b, tuple):
    return (a, *b)
  return (a, b)

# assume(Particle, Distr)
def assume(particle, distr):
  return particle.state.assume(None, None, distr)

#sample(particle,distr)
def sample(particle, distr):
  # Unlike assume(), a sample site must realize its random variable.
  distr = particle.state.eval_distr(distr)
  if not distr.rvs():
    return const(distr.draw(particle.state.rng))
  rv = particle.state.assume(None, None, distr)
  return particle.state.value_impl(rv)

# observe(Particle, Distr, Val)
def observe(particle, distr, val):
  # print("particle:",particle)
  # print("distr:",distr)
  # print("val:",val)
  s = siren_observe(particle.score, distr, val, particle.state)
  particle.score = s

#list.range(const, const)
def list_range(val1, val2):
  def bound(value):
    if isinstance(value, SymExpr):
      value = SSIState().eval(value)
      if not isinstance(value, Const):
        raise TypeError("list_range bounds must evaluate to constants")
      value = value.v
    if isinstance(value, float) and value.is_integer():
      value = int(value)
    if not isinstance(value, int):
      raise TypeError("list_range bounds must be integers")
    return value
  return [Const(i) for i in range(bound(val1), bound(val2))]

def list_len(values):
  return Const(len(values))

def list_hd(val):
  return val[0]

def list_tl(val):
  return val[1:]

def list_rev(val):
  return val[::-1]

def get_a(distr):
    return distr.a

def get_b(distr):
  return distr.b

def list_map(particle, func, values):
  return [func(particle, item) for item in values]

def lst(*args):
  # lst() -> empty list; lst(x) -> [x]
  return list(args)

#fold(func, lst, acc)
def fold(particle, func, lst, acc):
  if not isinstance(lst, (list, tuple)):
    lst = [lst]
  for item in lst:
    acc = func(particle, item, acc)
  return acc

# finalize(Particle, RV)
def finalize(particle, rv):
  return rv

# new_var(Particle)
def new_rv(particle):
  return particle.state.new_var()

# get_par(Particle, RV)
def get_par(particle, rv, idx):
  parents = particle.state.val_parents(rv)
  assert parents, f"{rv} has no parents; distr = {particle.state.distr(rv)}"
  return parents[idx.v]

# def sub(left, right):
#     return Add(left, Mul(Const(-1), right))


# set_distr(Particle, RV, Distr)
def set_distr(particle, rv, distr):
  particle.state.set_distr(rv, distr)

def get_distr(particle, rv):
  return particle.state.distr(rv)

# get_mu(Distr)
def get_mu(distr):
  return distr.mu

# get_var(Distr)
def get_var(distr):
  return distr.var

#get_p(Distr):
def get_p(distr):
  return distr.p

# observe_inner(Particle, RV, Num)
def observe_inner(particle, rv, num):
  v = particle.state.value_expr(num)
  s = particle.state.observe(rv, v)
  particle.score += s



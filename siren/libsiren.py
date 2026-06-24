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

# const(Num)
def const(n):
  return Const(n)

# times(Num, Num)
def times(n1,n2):
  Mul(n1,n2)

# assume(Particle, Distr)
def assume(particle, distr):
  return particle.state.assume(None, None, distr)

#sample(particle,distr)
def sample(particle, distr):
  # print("Available methods in SSIState:", dir(particle.state))
  # print("particel",particle)
  # print("distr", distr)
  return particle.state.assume(None, None, distr)

# observe(Particle, Distr, Val)
def observe(particle, distr, val):
  print("particle:",particle)
  print("distr:",distr)
  print("val:",val)
  s = siren_observe(particle.score, distr, val, particle.state)
  particle.score = s

#list.range(const, const)
def list_range(val1, val2):
  return [Const(i) for i in range(val1, val2)]


#fold(func, lst, acc)
def fold(particle, func, lst, acc):
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
  return parents[idx.v]


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



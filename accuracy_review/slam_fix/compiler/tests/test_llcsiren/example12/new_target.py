from siren.libsiren import *

def make_obs1(obs, x_pre):
  rv3 = new_rv(particle)
  set_distr(particle, rv3, gaussian(x_pre, const(1.0)))
  set_lookup(x,rv3)
  x = rv3
  rv5 = new_rv(particle)
  set_distr(particle, rv5, gaussian(x, const(1.0)))
  set_distr(particle, rv5, gaussian(x, const(1.0)))
  set_distr(particle, rv5, gaussian(x, const(1.0)))
  set_distr(particle, get_par(get_lookup("x_pre")), gaussian(times(get_lookup(x), const(0.9900990099009901)), const(0.9900990099009901)))
  set_distr(particle, get_lookup("x"), gaussian(const(0.0), const(101.0)))
  set_distr(particle, get_lookup("x"), gaussian(times(add(times(const(1.0), get_distr(particle, get_lookup(rv3))), rv5), const(0.5)), const(0.5)))
  set_distr(particle, rv5, gaussian(times(const(1.0), get_distr(particle, get_lookup(rv3))), const(2.0)))
  observe_inner(particle, rv5, obs)
  return x

def program(particle):
  data = cons(const(1.0), cons(const(3.0), lst()))
  rv1 = new_rv(particle)
  set_distr(particle, rv1, gaussian(const(0.0), const(100.0)))
  set_lookup(x_init,rv1)
  x_init = rv1
  xs = fold(particle,make_obs1, data, x_init)
  particle.finished = True
  return xs

run(program, 1000)

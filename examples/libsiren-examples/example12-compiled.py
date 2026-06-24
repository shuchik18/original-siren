from siren.libsiren import *

def make_obs1(particle, obs, x_pre):
  rv3 = new_rv(particle)
  set_distr(particle, rv3, gaussian(x_pre, const(1.0)))
  x = rv3
  rv5 = new_rv(particle)
  set_distr(particle, rv5, gaussian(x, const(1.0)))
  set_distr(particle, rv5, gaussian(get_mu(get_distr(particle, rv5)), get_var(get_distr(particle, rv5))))
  set_distr(particle, rv5, gaussian(get_mu(get_distr(particle, rv5)), get_var(get_distr(particle, rv5))))
  set_distr(particle, get_par(particle, get_par(particle, rv5, const(0)), const(0)), gaussian(times(get_par(particle, rv5, const(0)), const(0.9900990099009901)), const(0.9900990099009901)))
  set_distr(particle, get_par(particle, rv5, const(0)), gaussian(const(0.0), const(101.0)))
  set_distr(particle, get_par(particle, rv5, const(0)), gaussian(times(rv5, const(0.9901960784313726)), const(0.9901960784313726)))
  set_distr(particle, rv5, gaussian(const(0.0), const(102.0)))
  observe_inner(particle, rv5, obs)
  return x

def program(particle):
  data = list_range(1, 3)
  rv1 = new_rv(particle)
  set_distr(particle, rv1, gaussian(const(0.0), const(100.0)))
  x_init = rv1
  xs = fold(particle, make_obs1, data, x_init)
  particle.finished = True
  return xs

run(program, 1000)

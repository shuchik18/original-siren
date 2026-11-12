from siren.libsiren import *

def new_make_obs1(particle, obs, x_pre):
  rv3 = new_rv(particle)
  set_distr(particle, rv3, gaussian(get_mu(get_distr(particle, x_pre)), add(get_var(get_distr(particle, x_pre)), const(1.0))))
  x = rv3
  rv5 = new_rv(particle)
  set_distr(particle, rv5, gaussian(x, const(1.0)))
  set_distr(particle, rv5, gaussian(get_mu(get_distr(particle, rv5)), get_var(get_distr(particle, rv5))))
  __siren_update_marginal_8 = gaussian(get_mu(get_distr(particle, get_par(particle, rv5, const(0)))), add(get_var(get_distr(particle, get_par(particle, rv5, const(0)))), const(1.0)))
  set_distr(particle, get_par(particle, rv5, const(0)), gaussian(times(add(div(get_mu(get_distr(particle, get_par(particle, rv5, const(0)))), get_var(get_distr(particle, get_par(particle, rv5, const(0))))), rv5), div(const(1.0), add(div(const(1.0), get_var(get_distr(particle, get_par(particle, rv5, const(0))))), const(1.0)))), div(const(1.0), add(div(const(1.0), get_var(get_distr(particle, get_par(particle, rv5, const(0))))), const(1.0)))))
  set_distr(particle, rv5, __siren_update_marginal_8)
  set_distr(particle, rv5, gaussian(get_mu(get_distr(particle, rv5)), get_var(get_distr(particle, rv5))))
  observe_inner(particle, rv5, obs)
  return x

def program(particle):
  data = list_range(1, 10)
  rv1 = new_rv(particle)
  set_distr(particle, rv1, gaussian(const(0.0), const(100.0)))
  x_init = rv1
  xs = fold(particle, new_make_obs1, data, x_init)
  particle.finished = True
  return xs

run(program, 1000)

from siren.libsiren import *

def make_obs1(particle, obs, x_pre):
  rv3 = new_rv(particle)
  set_distr(particle, rv3, bernoulli(const(0.9)))
  set_distr(particle, rv3, bernoulli(get_p(get_distr(particle, rv3))))
  set_distr(particle, rv3, bernoulli(get_p(get_distr(particle, rv3))))
  S5 = sample(particle, get_distr(particle, rv3))
  set_distr(particle, rv3, delta(S5))
  out = S5
  if out:
    rv6 = new_rv(particle)
    set_distr(particle, rv6, gaussian(x_pre, const(1.0)))
    x = rv6
    rv8 = new_rv(particle)
    set_distr(particle, rv8, gaussian(x, const(1.0)))
    set_distr(particle, rv8, gaussian(get_mu(get_distr(particle, rv8)), get_var(get_distr(particle, rv8))))
    set_distr(particle, rv8, gaussian(get_mu(get_distr(particle, rv8)), get_var(get_distr(particle, rv8))))
    set_distr(particle, get_par(particle, get_par(particle, rv8, const(0)), const(0)), gaussian(times(get_par(particle, rv8, const(0)), const(0.9900990099009901)), const(0.9900990099009901)))
    set_distr(particle, get_par(particle, rv8, const(0)), gaussian(const(0.0), const(101.0)))
    set_distr(particle, get_par(particle, rv8, const(0)), gaussian(times(rv8, const(0.9901960784313726)), const(0.9901960784313726)))
    set_distr(particle, rv8, gaussian(const(0.0), const(102.0)))
    observe_inner(particle, rv8, obs)
    __siren_tmp_1 = x
  else:
    rv6 = new_rv(particle)
    set_distr(particle, rv6, gaussian(x_pre, const(1.0)))
    x = rv6
    rv8 = new_rv(particle)
    set_distr(particle, rv8, gaussian(const(0.0), const(100.0)))
    set_distr(particle, rv8, gaussian(get_mu(get_distr(particle, rv8)), get_var(get_distr(particle, rv8))))
    set_distr(particle, rv8, gaussian(get_mu(get_distr(particle, rv8)), get_var(get_distr(particle, rv8))))
    observe_inner(particle, rv8, obs)
    __siren_tmp_1 = x
  return __siren_tmp_1

def program(particle):
  data = list_range(1, 10)
  rv1 = new_rv(particle)
  set_distr(particle, rv1, gaussian(const(0.0), const(100.0)))
  x_init = rv1
  xs = fold(particle, make_obs1, data, x_init)
  particle.finished = True
  return xs

run(program, 1000)

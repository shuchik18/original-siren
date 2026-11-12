from siren.libsiren import *

def new_make_obs1(particle, obs, x_pre):
  rv3 = new_rv(particle)
  set_distr(particle, rv3, bernoulli(const(0.9)))
  set_distr(particle, rv3, bernoulli(const(0.9)))
  set_distr(particle, rv3, bernoulli(const(0.9)))
  S5 = sample(particle, get_distr(particle, rv3))
  set_distr(particle, rv3, delta(S5))
  out = S5
  if particle.state.value_expr(out).v:
    rv6 = new_rv(particle)
    set_distr(particle, rv6, gaussian(get_mu(get_distr(particle, x_pre)), add(get_var(get_distr(particle, x_pre)), const(1.0))))
    x = rv6
    rv8 = new_rv(particle)
    set_distr(particle, rv8, gaussian(x, const(1.0)))
    set_distr(particle, rv8, gaussian(get_mu(get_distr(particle, rv8)), const(1.0)))
    __siren_update_marginal_11 = gaussian(add(times(const(1.0), get_mu(get_distr(particle, get_par(particle, rv8, const(0))))), subst_rv(get_mu(get_distr(particle, rv8)), get_par(particle, rv8, const(0)), const(0.0))), add(times(const(1.0), get_var(get_distr(particle, get_par(particle, rv8, const(0))))), get_var(get_distr(particle, rv8))))
    set_distr(particle, get_par(particle, rv8, const(0)), gaussian(times(add(div(add(times(const(1.0), get_mu(get_distr(particle, get_par(particle, rv8, const(0))))), subst_rv(get_mu(get_distr(particle, rv8)), get_par(particle, rv8, const(0)), const(0.0))), times(const(1.0), get_var(get_distr(particle, get_par(particle, rv8, const(0)))))), rv8), div(const(1.0), add(div(const(1.0), times(const(1.0), get_var(get_distr(particle, get_par(particle, rv8, const(0)))))), const(1.0)))), div(div(const(1.0), add(div(const(1.0), times(const(1.0), get_var(get_distr(particle, get_par(particle, rv8, const(0)))))), const(1.0))), const(1.0))))
    set_distr(particle, rv8, __siren_update_marginal_11)
    set_distr(particle, rv8, gaussian(get_mu(get_distr(particle, rv8)), get_var(get_distr(particle, rv8))))
    observe_inner(particle, rv8, obs)
    __siren_tmp_1 = x
  else:
    rv12 = new_rv(particle)
    set_distr(particle, rv12, gaussian(get_mu(get_distr(particle, x_pre)), add(get_var(get_distr(particle, x_pre)), const(1.0))))
    x = rv12
    rv14 = new_rv(particle)
    set_distr(particle, rv14, gaussian(const(0.0), const(100.0)))
    set_distr(particle, rv14, gaussian(const(0.0), const(100.0)))
    set_distr(particle, rv14, gaussian(const(0.0), const(100.0)))
    observe_inner(particle, rv14, obs)
    __siren_tmp_1 = x
  return __siren_tmp_1

def program(particle):
  data = list_range(1, 10)
  rv1 = new_rv(particle)
  set_distr(particle, rv1, gaussian(const(0.0), const(100.0)))
  x_init = rv1
  xs = fold(particle, new_make_obs1, data, x_init)
  particle.finished = True
  return xs

run(program, 1000)

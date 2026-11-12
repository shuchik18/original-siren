from siren.libsiren import *

def new_make_obs1(particle, obs, prev_x):
  rv3 = new_rv(particle)
  set_distr(particle, rv3, gaussian(prev_x, const(1.0)))
  x = rv3
  rv5 = new_rv(particle)
  set_distr(particle, rv5, gaussian(x, const(1.0)))
  set_distr(particle, rv5, gaussian(x, const(1.0)))
  set_distr(particle, get_par(particle, get_par(particle, rv5, const(0)), const(0)), gaussian(times(get_par(particle, rv5, const(0)), const(0.9900990099009901)), const(0.9900990099009901)))
  set_distr(particle, get_par(particle, rv5, const(0)), gaussian(const(0.0), const(101.0)))
  set_distr(particle, get_par(particle, rv5, const(0)), gaussian(times(rv5, const(0.9901960784313726)), const(0.9901960784313726)))
  set_distr(particle, rv5, gaussian(const(0.0), const(102.0)))
  set_distr(particle, rv5, gaussian(const(0.0), const(102.0)))
  observe_inner(particle, rv5, obs)
  return x

def new_make_obs2(particle, obs, prev_x):
  rv10 = new_rv(particle)
  set_distr(particle, rv10, gaussian(prev_x, const(1.0)))
  x = rv10
  rv12 = new_rv(particle)
  set_distr(particle, rv12, gaussian(x, const(1.0)))
  set_distr(particle, rv12, gaussian(get_mu(get_distr(particle, rv12)), const(1.0)))
  set_distr(particle, rv12, gaussian(get_mu(get_distr(particle, rv12)), const(1.0)))
  observe_inner(particle, rv12, obs)
  return x

def program(particle):
  data = list_range(1, 5)
  rv1 = new_rv(particle)
  set_distr(particle, rv1, gaussian(const(0.0), const(100.0)))
  x0 = rv1
  obs1 = list_hd(data)
  rest1 = list_tl(data)
  x1 = fold(particle, new_make_obs1, obs1, x0)
  obs4 = list_hd(rest1)
  x4 = fold(particle, new_make_obs2, obs4, x1)
  last_state = x4
  past_states = x1
  particle.finished = True
  return pair(last_state, past_states)

run(program, 1000)

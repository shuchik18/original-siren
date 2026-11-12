from siren.libsiren import *

def make_obs(particle, obs, prev_x):
  x = assume(particle, gaussian(prev_x, const(1.0)))
  observe(particle, gaussian(x, const(1.0)), obs)
  return x

def program(particle):
  data = list_range(1, 5)
  x0 = assume(particle, gaussian(const(0.0), const(100.0)))
  obs1 = list_hd(data)
  rest1 = list_tl(data)
  x1 = fold(particle, make_obs, obs1, x0)
  obs4 = list_hd(rest1)
  x4 = fold(particle, make_obs, obs4, x1)
  last_state = x4
  past_states = x1
  particle.finished = True
  return pair(last_state, past_states)

run(program, 1000)

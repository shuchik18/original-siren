from siren.libsiren import *

def make_obs(particle, obs, x_pre):
  x = assume(particle, gaussian(x_pre, const(1.0)))
  observe(particle, gaussian(x, const(1.0)), obs)
  return x

def program(particle):
  data = list_range(1, 3)
  x_init = assume(particle, gaussian(const(0.0), const(100.0)))
  xs = fold(particle, make_obs, data, x_init)
  particle.finished = True
  return xs

run(program, 1000)

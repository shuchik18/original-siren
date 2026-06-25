from siren.libsiren import *

def make_obs(particle, obs, x_pre):
  out = sample(particle, bernoulli(const(0.9)))
  if out:
    x = assume(particle, gaussian(x_pre, const(1.0)))
    observe(particle, gaussian(x, const(1.0)), obs)
    __siren_tmp_1 = x
  else:
    x = assume(particle, gaussian(x_pre, const(1.0)))
    observe(particle, gaussian(const(0.0), const(100.0)), obs)
    __siren_tmp_1 = x
  return __siren_tmp_1

def program(particle):
  data = list_range(1, 10)
  x_init = assume(particle, gaussian(const(0.0), const(100.0)))
  xs = fold(particle, make_obs, data, x_init)
  particle.finished = True
  return xs

run(program, 1000)

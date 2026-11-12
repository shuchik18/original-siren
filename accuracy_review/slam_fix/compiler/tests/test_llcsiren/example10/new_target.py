from siren.libsiren import *

def program(particle):
  rv1 = new_rv(particle)
  set_distr(particle, rv1, gaussian(const(0.0), const(100.0)))
  set_lookup("x",rv1)
  x = rv1
  rv3 = new_rv(particle)
  set_distr(particle, rv3, gaussian(x, times(const(5.0), const(1.0))))
  set_distr(particle, rv3, gaussian(x, const(5.0)))
  set_distr(particle, rv3, gaussian(x, times(const(5.0), const(1.0))))
  set_distr(particle, get_lookup("x"), gaussian(times(div(rv3, gaussian(x, times(const(5.0), const(1.0)))), const(4.761904761904762)), const(4.761904761904762)))
  set_distr(particle, rv3, gaussian(const(0.0), const(105.0)))
  observe_inner(particle, rv3, const(3.0))
  particle.finished = True
  return x

run(program, 1000)

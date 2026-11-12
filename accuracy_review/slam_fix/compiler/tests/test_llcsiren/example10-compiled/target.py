from siren.libsiren import *

def program(particle):
  rv1 = new_rv(particle)
  set_distr(particle, rv1, gaussian(const(0.0), const(100.0)))
  x = rv1
  rv3 = new_rv(particle)
  set_distr(particle, rv3, gaussian(x, times(const(5.0), const(1.0))))
  set_distr(particle, rv3, gaussian(get_mu(get_distr(particle, rv3)), const(5.0)))
  set_distr(particle, rv3, gaussian(get_mu(get_distr(particle, rv3)), get_var(get_distr(particle, rv3))))
  set_distr(particle, get_par(particle, rv3, const(0)), gaussian(times(div(rv3, get_distr(particle, rv3)), const(4.761904761904762)), const(4.761904761904762)))
  set_distr(particle, rv3, gaussian(const(0.0), const(105.0)))
  observe_inner(particle, rv3, const(3.0))
  particle.finished = True
  return x

print(run(program, 1000))

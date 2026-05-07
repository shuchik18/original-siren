from siren.libsiren import *

def program(particle):
  rv1 = new_rv(particle)
  set_distr(particle, rv1, gaussian(const(0.0),const(100.0)))
  x = rv1
  rv3 = new_rv(particle)
  set_distr(particle, rv3, gaussian(x, const(1.0)))
  set_distr(particle, rv3, gaussian(get_mu(get_distr(particle,rv3)), get_var(get_distr(particle,rv3))))
  set_distr(particle, rv3, gaussian(get_mu(get_distr(particle,rv3)), get_var(get_distr(particle,rv3))))
  set_distr(particle, get_par(particle, rv3, const(0)), gaussian(times(rv3, const(0.9900990099009901)), const(0.9900990099009901)))
  set_distr(particle, rv3, gaussian(const(0.0), const(101.0)))
  observe_inner(particle, rv3, const(3.0))
  particle.finished = True
  return x


print(run(program, 1000))

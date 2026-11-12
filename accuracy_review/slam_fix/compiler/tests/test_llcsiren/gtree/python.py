from siren.libsiren import *

def program(particle):
  rv1 = new_rv(particle)
  set_distr(particle, rv1, gaussian(const(0.0), const(100.0)))
  a = rv1
  rv3 = new_rv(particle)
  set_distr(particle, rv3, gaussian(a, const(10.0)))
  b = rv3
  rv5 = new_rv(particle)
  set_distr(particle, rv5, gaussian(b, const(1000.0)))
  set_distr(particle, rv5, gaussian(b, const(1000.0)))
  set_distr(particle, get_par(particle, get_par(particle, rv5, const(0)), const(0)), gaussian(times(div(get_par(particle, rv5, const(0)), get_var(get_distr(particle, get_par(particle, rv5, const(0))))), const(9.090909090909092)), const(9.090909090909092)))
  set_distr(particle, get_par(particle, rv5, const(0)), gaussian(const(0.0), const(110.0)))
  set_distr(particle, get_par(particle, rv5, const(0)), gaussian(times(div(rv5, get_var(get_distr(particle, rv5))), const(99.09909909909909)), const(99.09909909909909)))
  set_distr(particle, rv5, gaussian(const(0.0), const(1110.0)))
  set_distr(particle, rv5, gaussian(const(0.0), const(1110.0)))
  observe_inner(particle, rv5, const(2.3))
  rv9 = new_rv(particle)
  set_distr(particle, rv9, gaussian(a, const(1000.0)))
  set_distr(particle, rv9, gaussian(get_mu(get_distr(particle, rv9)), const(1000.0)))
  set_distr(particle, rv9, gaussian(get_mu(get_distr(particle, rv9)), const(1000.0)))
  observe_inner(particle, rv9, const(-55.0))
  particle.finished = True
  return a

run(program, 1000)

from siren.libsiren import *

def program(particle):
  rv1 = new_rv(particle)
  set_distr(particle, rv1, gaussian(const(0.0), const(10.0)))
  set_lookup("x",rv1)
  x = rv1
  rv3 = new_rv(particle)
  set_distr(particle, rv3, bernoulli(const(0.5)))
  set_distr(particle, rv3, bernoulli(const(0.5)))
  set_distr(particle, rv3, bernoulli(const(0.5)))
  S5 = sample(particle, get_distr(particle, rv3))
  set_distr(particle, rv3, delta(S5))
  b = S5
  if b:
    rv6 = new_rv(particle)
    set_distr(particle, rv6, gaussian(const(0.0), const(100.0)))
    set_distr(particle, rv6, gaussian(const(0.0), const(100.0)))
    set_distr(particle, rv6, gaussian(const(0.0), const(100.0)))
    observe_inner(particle, rv6, const(3.0))
  else:
    rv8 = new_rv(particle)
    set_distr(particle, rv8, gaussian(x, const(1.0)))
    set_distr(particle, rv8, gaussian(x, const(1.0)))
    set_distr(particle, rv8, gaussian(x, const(1.0)))
    set_distr(particle, get_lookup("x"), gaussian(times(rv8, const(0.9090909090909091)), const(0.9090909090909091)))
    set_distr(particle, rv8, gaussian(const(0.0), const(11.0)))
    observe_inner(particle, rv8, const(3.0))
  particle.finished = True
  return x

run(program, 1000)

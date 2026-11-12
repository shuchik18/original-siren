from siren.libsiren import *

def program(particle):
  input = const(0.0)
  rv1 = new_rv(particle)
  set_distr(particle, rv1, bernoulli(const(0.5)))
  set_lookup("b",rv1)
  set_distr(particle, rv1, bernoulli(const(0.5)))
  set_distr(particle, rv1, bernoulli(const(0.5)))
  S3 = sample(particle, get_distr(particle, rv1))
  set_distr(particle, rv1, delta(S3))
  b = S3
  rv4 = new_rv(particle)
  set_distr(particle, rv4, gaussian(const(0.0), const(10.0)))
  set_lookup(x,rv4)
  x = rv4
  if b:
    rv6 = new_rv(particle)
    set_distr(particle, rv6, gaussian(x, const(1.0)))
    set_distr(particle, rv6, gaussian(x, const(1.0)))
    set_distr(particle, rv6, gaussian(x, const(1.0)))
    set_distr(particle, get_lookup("x"), gaussian(times(rv6, const(0.9090909090909091)), const(0.9090909090909091)))
    set_distr(particle, rv6, gaussian(const(0.0), const(11.0)))
    observe_inner(particle, rv6, input)
  else:
    rv9 = new_rv(particle)
    set_distr(particle, rv9, gaussian(const(0.0), const(100.0)))
    set_distr(particle, rv9, gaussian(const(0.0), const(100.0)))
    set_distr(particle, rv9, gaussian(const(0.0), const(100.0)))
    observe_inner(particle, rv9, input)
  particle.finished = True
  return x

run(program, 1000)

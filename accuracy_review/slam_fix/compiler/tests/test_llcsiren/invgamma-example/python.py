from siren.libsiren import *

def program(particle):
  rv1 = new_rv(particle)
  set_distr(particle, rv1, gamma(const(1.0), const(2.0)))
  set_distr(particle, rv1, gamma(get_distr(particle, rv1), get_distr(particle, rv1)))
  set_distr(particle, rv1, gamma(get_distr(particle, rv1), get_distr(particle, rv1)))
  S3 = sample(particle, get_distr(particle, rv1))
  set_distr(particle, rv1, delta(S3))
  y = S3
  rv4 = new_rv(particle)
  set_distr(particle, rv4, gaussian(const(0.0), const(100.0)))
  x = rv4
  rv6 = new_rv(particle)
  set_distr(particle, rv6, gaussian(const(0.0), const(1.0)))
  set_distr(particle, rv6, normal(get_distr(particle, rv6), get_distr(particle, rv6)))
  set_distr(particle, rv6, normal(get_distr(particle, rv6), get_distr(particle, rv6)))
  observe_inner(particle, rv6, y)
  particle.finished = True
  return x

run(program, 1000)

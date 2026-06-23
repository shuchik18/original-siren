from siren.libsiren import *

def program(particle):
  rv1 = new_rv(particle)
  set_distr(particle, rv1, gaussian(const(0.0), const(1.0)))
  x = rv1
  particle.finished = True
  return x

run(program, 1000)

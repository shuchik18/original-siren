from siren.libsiren import *

def program(particle):
  x = assume(particle, gaussian(const(0.0), const(1.0)))
  particle.finished = True
  return x

run(program, 1000)

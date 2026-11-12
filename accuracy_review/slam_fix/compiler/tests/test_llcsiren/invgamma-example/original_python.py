from siren.libsiren import *

def program(particle):
  y = sample(particle, gamma(const(1.0), const(2.0)))
  x = assume(particle, gaussian(const(0.0), const(100.0)))
  observe(particle, gaussian(const(0.0), const(1.0)), y)
  particle.finished = True
  return x

run(program, 1000)

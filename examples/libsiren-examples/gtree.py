from siren.libsiren import *

def program(particle):
  a = assume(particle, gaussian(const(0.0), const(100.0)))
  b = assume(particle, gaussian(a, const(10.0)))
  observe(particle, gaussian(b, const(1000.0)), const(2.3))
  observe(particle, gaussian(a, const(1000.0)), const(-55.0))
  particle.finished = True
  return a

run(program, 1000)
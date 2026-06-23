from siren.libsiren import *

def program(particle):
  x = assume(particle, gaussian(const(0.0), const(10.0)))
  b = assume(particle, bernoulli(const(0.5)))
  if b:
    observe(particle, gaussian(const(0.0), const(100.0)), const(3.0))
  else:
    observe(particle, gaussian(x, const(1.0)), const(3.0))
  particle.finished = True
  return x

run(program, 1000)

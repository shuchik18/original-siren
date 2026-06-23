from siren.libsiren import *

def program(particle):
  input = const(0.0)
  b = assume(particle, bernoulli(const(0.5)))
  x = assume(particle, gaussian(const(0.0), const(10.0)))
  if b:
    observe(particle, gaussian(x, const(1.0)), input)
  else:
    observe(particle, gaussian(const(0.0), const(100.0)), input)
  particle.finished = True
  return x

run(program, 1000)

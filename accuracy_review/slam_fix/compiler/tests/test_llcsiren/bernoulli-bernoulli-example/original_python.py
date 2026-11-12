from siren.libsiren import *

def program(particle):
  z = assume(particle, bernoulli(const(0.5)))
  observe(particle, bernoulli(ifelse(z, const(0.8), const(0.2))), const(True))
  particle.finished = True
  return z

run(program, 1000)

from siren.libsiren import *

def program(particle):
  b = sample(particle, bernoulli(const(0.5)))
  particle.finished = True
  return b

print(run(program, 1000))

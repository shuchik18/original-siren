from siren.libsiren import *

def program(particle):
  xt = assume(particle, beta(const(1.0), const(1.0)))
  observe(particle, bernoulli(xt), const(False))
  observe(particle, bernoulli(xt), const(True))
  observe(particle, bernoulli(xt), const(True))
  observe(particle, bernoulli(xt), const(True))
  observe(particle, bernoulli(xt), const(True))
  observe(particle, bernoulli(xt), const(True))
  observe(particle, bernoulli(xt), const(True))
  observe(particle, bernoulli(xt), const(True))
  particle.finished = True
  return xt

run(program, 1000)

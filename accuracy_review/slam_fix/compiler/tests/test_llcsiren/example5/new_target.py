from siren.libsiren import *

def program(particle):
  rv1 = new_rv(particle)
  set_distr(particle, rv1, bernoulli(const(0.5)))
  set_lookup(b,rv1)
  set_distr(particle, rv1, bernoulli(((const(0.5)))))
  set_distr(particle, rv1, bernoulli(((const(0.5)))))
  S3 = sample(particle, bernoulli(const(0.5)))
  set_distr(particle, rv1, delta(S3))
  b = S3
  particle.finished = True
  return b

run(program, 1000)

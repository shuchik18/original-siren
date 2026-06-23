from siren.libsiren import *

def program(particle):
  rv1 = new_rv(particle)
  set_distr(particle, rv1, bernoulli(const(0.5)))
  set_distr(particle, rv1, bernoulli(get_p(get_distr(particle, rv1))))
  set_distr(particle, rv1, bernoulli(get_p(get_distr(particle, rv1))))
  S3 = sample(particle, get_distr(particle, rv1))
  set_distr(particle, rv1, delta(S3))
  b = S3
  particle.finished = True
  return b

print(run(program, 1000))

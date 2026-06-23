from siren.libsiren import *

def program(particle):
  rv1 = new_rv(particle)
  set_distr(particle, rv1, gaussian(const(0.0), const(10.0)))
  x = rv1
  rv3 = new_rv(particle)
  set_distr(particle, rv3, bernoulli(const(0.5)))
  set_distr(particle, rv3, bernoulli(get_p(get_distr(particle, rv3))))
  set_distr(particle, rv3, bernoulli(get_p(get_distr(particle, rv3))))
  S5 = sample(particle, get_distr(particle, rv3))
  set_distr(particle, rv3, delta(S5))
  b = S5
  if b:
    rv6 = new_rv(particle)
    set_distr(particle, rv6, gaussian(const(0.0), const(100.0)))
    set_distr(particle, rv6, gaussian(get_mu(get_distr(particle, rv6)), get_var(get_distr(particle, rv6))))
    set_distr(particle, rv6, gaussian(get_mu(get_distr(particle, rv6)), get_var(get_distr(particle, rv6))))
    observe_inner(particle, rv6, const(3.0))
  else:
    rv8 = new_rv(particle)
    set_distr(particle, rv8, gaussian(x, const(1.0)))
    set_distr(particle, rv8, gaussian(get_mu(get_distr(particle, rv8)), get_var(get_distr(particle, rv8))))
    set_distr(particle, rv8, gaussian(get_mu(get_distr(particle, rv8)), get_var(get_distr(particle, rv8))))
    set_distr(particle, get_par(particle, rv8, const(0)), gaussian(times(rv8, const(0.9090909090909091)), const(0.9090909090909091)))
    set_distr(particle, rv8, gaussian(const(0.0), const(11.0)))
    observe_inner(particle, rv8, const(3.0))
  particle.finished = True
  return x

print(run(program, 1000))

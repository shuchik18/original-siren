from siren.libsiren import *

def program(particle):
  rv1 = new_rv(particle)
  set_distr(particle, rv1, bernoulli(const(0.5)))
  z = rv1
  rv3 = new_rv(particle)
  set_distr(particle, rv3, bernoulli(ifelse(z, const(0.8), const(0.2))))
  set_distr(particle, rv3, bernoulli(ifelse(z, const(0.8), const(0.2))))
  set_distr(particle, get_par(particle, rv3, const(0)), Bernoulli(div(times(get_p(get_distr(particle, get_par(particle, rv3, const(0)))), subst_rv(ifelse(rv3, get_p(get_distr(particle, rv3)), add(const(1.0), times(sub(const(0.0), const(1.0)), get_p(get_distr(particle, rv3))))), get_par(particle, rv3, const(0)), const(True))), ifelse(rv3, add(times(get_p(get_distr(particle, get_par(particle, rv3, const(0)))), subst_rv(get_p(get_distr(particle, rv3)), get_par(particle, rv3, const(0)), const(True))), times(const(0.5), subst_rv(get_p(get_distr(particle, rv3)), get_par(particle, rv3, const(0)), const(False)))), add(const(1.0), times(sub(const(0.0), const(1.0)), add(times(get_p(get_distr(particle, get_par(particle, rv3, const(0)))), subst_rv(get_p(get_distr(particle, rv3)), get_par(particle, rv3, const(0)), const(True))), times(const(0.5), subst_rv(get_p(get_distr(particle, rv3)), get_par(particle, rv3, const(0)), const(False))))))))))
  set_distr(particle, rv3, Bernoulli(add(times(get_p(get_distr(particle, get_par(particle, rv3, const(0)))), subst_rv(get_p(get_distr(particle, rv3)), get_par(particle, rv3, const(0)), const(True))), times(const(0.5), subst_rv(get_p(get_distr(particle, rv3)), get_par(particle, rv3, const(0)), const(False))))))
  set_distr(particle, rv3, bernoulli(const(0.5)))
  observe_inner(particle, rv3, const(True))
  particle.finished = True
  return z

run(program, 1000)

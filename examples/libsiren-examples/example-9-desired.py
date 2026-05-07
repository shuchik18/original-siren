from siren.libsiren import *

def program(particle):
  rv1 = new_rv(particle)
  set_distr(particle, rv1, gaussian(const(0.0), const(100.0)))
  # let () = Prob.set_lookup("x",rv1) in
  x = rv1
  rv3 = new_rv(particle)
  set_distr(particle, rv3, gaussian(const(0.0), const(101.0)))
  set_distr(particle, x, gaussian(times(rv3, 0.9900990099009901),  0.9900990099009901))
  observe_inner(particle, rv3, const(3.0))
  particle.finished = True
  return x


print(run(program, 1000))

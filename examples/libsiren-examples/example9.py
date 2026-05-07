from siren.libsiren import *
def program(particle):
  x = assume(particle, gaussian(const(0.0), const(100.0)))
  observe(particle, gaussian(x,const(1.0)),const(3.0))
  particle.finished = True
  return x

print(run(program, 1000))

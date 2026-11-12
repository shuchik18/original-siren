from siren.libsiren import *

def preprocess_data(particle, entry):
  x_obs = list_hd(list_tl(list_tl(list_tl(entry))))
  alt_obs = list_hd(list_tl(list_tl(list_tl(list_tl(entry)))))
  return pair(x_obs, alt_obs)

def step(particle, __siren_tmp_1, __siren_tmp_2):
  (x_obs, alt_obs) = __siren_tmp_1
  (xs, alts, q, r) = __siren_tmp_2
  x = sample(particle, gaussian(list_hd(xs), q))
  alt = sample(particle, gaussian(list_hd(alts), q))
  other = sample(particle, gamma(const(1.0), const(10.0)))
  v = (add(r, div(const(1.0), other)) if lt(alt, const(5.0)) else r)
  observe(particle, gaussian(x, v), x_obs)
  observe(particle, gaussian(alt, v), alt_obs)
  return pair(cons(x, xs), pair(cons(alt, alts), pair(q, r)))

def program(particle):
  data = cons(pair(const(-0.10367769296577148), const(10.15942915818945)), cons(pair(const(0.17862462415256572), const(9.872415793216435)), cons(pair(const(1.6767701597795415), const(10.074207524509706)), lst())))
  invq = assume(particle, gamma(const(1.0), const(1.0)))
  invr = sample(particle, gamma(const(1.0), const(1.0)))
  q = div(const(1.0), invq)
  r = div(const(1.0), invr)
  (xs, alts, q_2, r_2) = fold(particle, step, data, pair(cons(const(0.0), lst()), pair(cons(const(0.0), lst()), pair(q, r))))
  xs_2 = list_tl(list_rev(xs))
  alts_2 = list_tl(list_rev(alts))
  particle.finished = True
  return pair(xs_2, pair(alts_2, pair(q_2, r_2)))

run(program, 1000)

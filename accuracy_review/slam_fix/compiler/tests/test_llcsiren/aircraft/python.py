from siren.libsiren import *

def new_make_obs1(particle, __siren_tmp_1, __siren_tmp_2):
  (x_obs, alt_obs) = __siren_tmp_1
  (xs, alts, q, r) = __siren_tmp_2
  rv6 = new_rv(particle)
  set_distr(particle, rv6, gaussian(list_hd(xs), q))
  set_distr(particle, rv6, gaussian(const(0.0), div(const(1.0), invq)))
  set_distr(particle, rv6, studentT(const(0.0), const(1.0), const(2.0)))
  set_distr(particle, get_par(particle, rv6, const(0)), Gamma(const(1.5), add(get_distr(particle, get_par(particle, rv6, const(0))), times(const(0.5), times(rv6, rv6)))))
  set_distr(particle, rv6, StudentT(get_distr(particle, rv6), const(1.0), const(2.0)))
  S8 = sample(particle, get_distr(particle, rv6))
  set_distr(particle, rv6, delta(S8))
  x = S8
  rv9 = new_rv(particle)
  set_distr(particle, rv9, gaussian(list_hd(alts), q))
  set_distr(particle, rv9, gaussian(const(0.0), div(const(1.0), invq)))
  set_distr(particle, rv9, studentT(const(0.0), get_tau2(get_distr(particle, rv9)), const(3.0)))
  set_distr(particle, get_par(particle, rv9, const(0)), Gamma(const(2.0), add(get_distr(particle, get_par(particle, rv9, const(0))), times(const(0.5), times(rv9, rv9)))))
  set_distr(particle, rv9, StudentT(get_distr(particle, rv9), div(get_distr(particle, get_par(particle, rv9, const(0))), get_distr(particle, get_par(particle, rv9, const(0)))), const(3.0)))
  S11 = sample(particle, get_distr(particle, rv9))
  set_distr(particle, rv9, delta(S11))
  alt = S11
  rv12 = new_rv(particle)
  set_distr(particle, rv12, gamma(const(1.0), const(10.0)))
  set_distr(particle, rv12, gamma(const(1.0), const(10.0)))
  set_distr(particle, rv12, gamma(const(1.0), const(10.0)))
  S14 = sample(particle, get_distr(particle, rv12))
  set_distr(particle, rv12, delta(S14))
  other = S14
  v = (add(r, div(const(1.0), other)) if lt(lt(alt, const(5.0)), const(5.0)) else r)
  rv15 = new_rv(particle)
  set_distr(particle, rv15, gaussian(x, v))
  set_distr(particle, rv15, gaussian(get_mu(get_distr(particle, rv15)), get_var(get_distr(particle, rv15))))
  set_distr(particle, rv15, gaussian(get_mu(get_distr(particle, rv15)), get_var(get_distr(particle, rv15))))
  observe_inner(particle, rv15, x_obs)
  rv17 = new_rv(particle)
  set_distr(particle, rv17, gaussian(alt, v))
  set_distr(particle, rv17, gaussian(get_mu(get_distr(particle, rv17)), get_var(get_distr(particle, rv17))))
  set_distr(particle, rv17, gaussian(get_mu(get_distr(particle, rv17)), get_var(get_distr(particle, rv17))))
  observe_inner(particle, rv17, alt_obs)
  return pair(cons(x, xs), pair(cons(alt, alts), pair(q, r)))

def program(particle):
  data = cons(pair(const(-0.10367769296577148), const(10.15942915818945)), cons(pair(const(0.17862462415256572), const(9.872415793216435)), cons(pair(const(1.6767701597795415), const(10.074207524509706)), lst())))
  rv1 = new_rv(particle)
  set_distr(particle, rv1, gamma(const(1.0), const(1.0)))
  invq = rv1
  rv3 = new_rv(particle)
  set_distr(particle, rv3, gamma(const(1.0), const(1.0)))
  set_distr(particle, rv3, gamma(const(1.0), const(1.0)))
  set_distr(particle, rv3, gamma(const(1.0), const(1.0)))
  S5 = sample(particle, get_distr(particle, rv3))
  set_distr(particle, rv3, delta(S5))
  invr = S5
  q = div(const(1.0), invq)
  r = div(const(1.0), invr)
  (xs, alts, q_2, r_2) = fold(particle, new_make_obs1, data, pair(cons(const(0.0), lst()), pair(cons(const(0.0), lst()), pair(q, r))))
  xs_2 = list_tl(list_rev(xs))
  alts_2 = list_tl(list_rev(alts))
  particle.finished = True
  return pair(xs_2, pair(alts_2, pair(q_2, r_2)))

run(program, 1000)

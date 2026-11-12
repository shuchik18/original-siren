from siren.libsiren import *

def bernoulli_priors(particle, i):
  rv1 = new_rv(particle)
  set_distr(particle, rv1, bernoulli(const(0.5)))
  cell = rv1
  return cell

def new_make_obs2(particle, __siren_tmp_1, __siren_tmp_2):
  (obs, cmd) = __siren_tmp_1
  (max_pos, map, xs) = __siren_tmp_2
  prev_x = const(0.0)
  sensor_noise = const(0.1)
  wheel_noise = const(0.1)
  rv23 = new_rv(particle)
  set_distr(particle, rv23, bernoulli(wheel_noise))
  wheel_slip = rv23
  cmd2 = cmd
  x = (prev_x if wheel_slip else add(prev_x, cmd2))
  x_2 = x
  (_, __2, __3, res) = fold(particle, new_make_obs1, map, pair(x_2, pair(const(0.0), pair(const(False), const(-1.0)))))
  o = res
  rv26 = new_rv(particle)
  set_distr(particle, rv26, bernoulli(ifelse(o, sub(const(1.0), sensor_noise), sensor_noise)))
  set_distr(particle, rv26, bernoulli(get_p(get_distr(particle, rv26))))
  set_distr(particle, get_par(particle, rv26, const(1)), bernoulli(div(times(get_p(get_distr(particle, get_par(particle, rv26, const(1)))), subst_rv(ifelse(rv26, get_p(get_distr(particle, rv26)), add(const(1.0), times(sub(const(0.0), const(1.0)), get_p(get_distr(particle, rv26))))), get_par(particle, rv26, const(1)), const(True))), ifelse(rv26, add(times(get_p(get_distr(particle, get_par(particle, rv26, const(1)))), subst_rv(get_p(get_distr(particle, rv26)), get_par(particle, rv26, const(1)), const(True))), times(const(0.5), subst_rv(get_p(get_distr(particle, rv26)), get_par(particle, rv26, const(1)), const(False)))), add(const(1.0), times(sub(const(0.0), const(1.0)), add(times(get_p(get_distr(particle, get_par(particle, rv26, const(1)))), subst_rv(get_p(get_distr(particle, rv26)), get_par(particle, rv26, const(1)), const(True))), times(const(0.5), subst_rv(get_p(get_distr(particle, rv26)), get_par(particle, rv26, const(1)), const(False))))))))))
  set_distr(particle, rv26, bernoulli(add(times(get_p(get_distr(particle, get_par(particle, rv26, const(1)))), subst_rv(get_p(get_distr(particle, rv26)), get_par(particle, rv26, const(1)), const(True))), times(const(0.5), subst_rv(get_p(get_distr(particle, rv26)), get_par(particle, rv26, const(1)), const(False))))))
  set_distr(particle, get_par(particle, rv26, const(0)), bernoulli(div(times(get_p(get_distr(particle, get_par(particle, rv26, const(0)))), subst_rv(ifelse(rv26, get_p(get_distr(particle, rv26)), add(const(1.0), times(sub(const(0.0), const(1.0)), get_p(get_distr(particle, rv26))))), get_par(particle, rv26, const(0)), const(True))), ifelse(rv26, add(times(get_p(get_distr(particle, get_par(particle, rv26, const(0)))), subst_rv(get_p(get_distr(particle, rv26)), get_par(particle, rv26, const(0)), const(True))), times(const(0.9), subst_rv(get_p(get_distr(particle, rv26)), get_par(particle, rv26, const(0)), const(False)))), add(const(1.0), times(sub(const(0.0), const(1.0)), add(times(get_p(get_distr(particle, get_par(particle, rv26, const(0)))), subst_rv(get_p(get_distr(particle, rv26)), get_par(particle, rv26, const(0)), const(True))), times(const(0.9), subst_rv(get_p(get_distr(particle, rv26)), get_par(particle, rv26, const(0)), const(False))))))))))
  set_distr(particle, rv26, bernoulli(add(times(get_p(get_distr(particle, get_par(particle, rv26, const(0)))), subst_rv(get_p(get_distr(particle, rv26)), get_par(particle, rv26, const(0)), const(True))), times(const(0.9), subst_rv(get_p(get_distr(particle, rv26)), get_par(particle, rv26, const(0)), const(False))))))
  set_distr(particle, rv26, bernoulli(const(0.8600000000000001)))
  observe_inner(particle, rv26, obs)
  return pair(max_pos, pair(map, cons(x_2, xs)))

def new_make_obs1(particle, curr, __siren_tmp_3):
  (n, i, found, res) = __siren_tmp_3
  found2 = (const(True) if eq(n, const(0.0)) else found)
  res2 = (RV(rv2) if eq(n, const(0.0)) else res)
  return pair(n, pair(add(i, const(1.0)), pair(found2, res2)))

def program(particle):
  data = cons(cons(const(0.0), cons(const(1.0), lst())), cons(cons(const(0.0), cons(const(-1.0), lst())), lst()))
  max_pos = const(10.0)
  map = cons(bernoulli_priors(particle, const(0.0)), cons(bernoulli_priors(particle, const(1.0)), cons(bernoulli_priors(particle, const(2.0)), cons(bernoulli_priors(particle, const(3.0)), cons(bernoulli_priors(particle, const(4.0)), cons(bernoulli_priors(particle, const(5.0)), cons(bernoulli_priors(particle, const(6.0)), cons(bernoulli_priors(particle, const(7.0)), cons(bernoulli_priors(particle, const(8.0)), cons(bernoulli_priors(particle, const(9.0)), cons(bernoulli_priors(particle, const(10.0)), lst())))))))))))
  (_, map_2, xs) = fold(particle, new_make_obs2, data, pair(max_pos, pair(map, lst())))
  x = list_hd(xs)
  particle.finished = True
  return pair(map_2, x)

run(program, 1000)

from siren.libsiren import *


def bernoulli_priors(particle, i):
 rv1 = new_rv(particle)
 set_distr(particle, rv1, bernoulli(const(0.5)))
 cell = rv1
 return cell


def new_make_obs1(particle, __siren_tmp_1, __siren_tmp_2):
 (obs, cmd) = __siren_tmp_1
 (max_pos, map, xs) = __siren_tmp_2
 prev_x = const(0.0)
 sensor_noise = const(0.1)
 wheel_noise = const(0.1)
 rv5 = new_rv(particle)
 set_distr(particle, rv5, bernoulli(wheel_noise))
 set_distr(particle, rv5, bernoulli(const(0.1)))
 set_distr(particle, rv5, bernoulli(const(0.1)))
 S7 = sample(particle, get_distr(particle, rv5))
 set_distr(particle, rv5, delta(S7))
 wheel_slip = S7
 cmd2 = cmd
 x = (prev_x if wheel_slip else add(prev_x, cmd2))
 x_2 = x
 c0 = list_hd(map)
 c1 = list_hd(list_tl(map))
 o = (c0 if eq(x_2, const(0.0)) else c1)
 rv9 = new_rv(particle)
 set_distr(particle, rv9, bernoulli(ifelse(o, sub(const(1.0), sensor_noise), sensor_noise)))
 set_distr(particle, rv9, bernoulli(get_p(get_distr(particle, rv9))))
 set_distr(particle, get_par(particle, rv9, const(0)), bernoulli(div(times(get_p(get_distr(particle, get_par(particle, rv9, const(0)))), subst_rv(ifelse(rv9, get_p(get_distr(particle, rv9)), add(const(1.0), times(sub(const(0.0), const(1.0)), get_p(get_distr(particle, rv9))))), get_par(particle, rv9, const(0)), const(True))), ifelse(rv9, add(times(get_p(get_distr(particle, get_par(particle, rv9, const(0)))), subst_rv(get_p(get_distr(particle, rv9)), get_par(particle, rv9, const(0)), const(True))), times(const(0.5), subst_rv(get_p(get_distr(particle, rv9)), get_par(particle, rv9, const(0)), const(False)))), add(const(1.0), times(sub(const(0.0), const(1.0)), add(times(get_p(get_distr(particle, get_par(particle, rv9, const(0)))), subst_rv(get_p(get_distr(particle, rv9)), get_par(particle, rv9, const(0)), const(True))), times(const(0.5), subst_rv(get_p(get_distr(particle, rv9)), get_par(particle, rv9, const(0)), const(False))))))))))
 set_distr(particle, rv9, bernoulli(add(times(get_p(get_distr(particle, get_par(particle, rv9, const(0)))), subst_rv(get_p(get_distr(particle, rv9)), get_par(particle, rv9, const(0)), const(True))), times(const(0.5), subst_rv(get_p(get_distr(particle, rv9)), get_par(particle, rv9, const(0)), const(False))))))
 set_distr(particle, rv9, bernoulli(const(0.5)))
 observe_inner(particle, rv9, obs)
 return pair(max_pos, pair(map, cons(x_2, xs)))


def program(particle):
 data = cons(pair(const(0.0), const(1.0)), cons(pair(const(0.0), const(-1.0)), lst()))
 max_pos = const(1.0)
 map = cons(bernoulli_priors(particle, const(0.0)), cons(bernoulli_priors(particle, const(1.0)), lst()))
 (_, map_2, xs) = fold(particle, new_make_obs1, data, pair(max_pos, pair(map, lst())))
 xs_2 = list_rev(xs)
 particle.finished = True
 return pair(map_2, xs_2)


run(program, 1000)





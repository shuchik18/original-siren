from siren.libsiren import *

def program(particle):
  data = cons(pair(const(0.0), const(1.0)), cons(pair(const(0.0), const(-1.0)), lst()))
  max_pos_3 = const(1.0)
  rv1 = new_rv(particle)
  set_distr(particle, rv1, bernoulli(const(0.5)))
  cell = rv1
  __siren_tmp_1 = cell
  rv3 = new_rv(particle)
  set_distr(particle, rv3, bernoulli(const(0.5)))
  cell_2 = rv3
  map_3 = lst(__siren_tmp_1, cell_2)
  (max_pos_2, map_2, xs) = pair(max_pos_3, pair(map_3, lst()))
  (obs_3, cmd_3) = list_hd(data)
  prev_x_2 = const(0.0)
  sensor_noise = const(0.1)
  max_pos = max_pos_2
  prev_x = prev_x_2
  cmd_2 = cmd_3
  wheel_noise = const(0.1)
  rv5 = new_rv(particle)
  set_distr(particle, rv5, bernoulli(wheel_noise))
  set_distr(particle, rv5, bernoulli(const(0.1)))
  set_distr(particle, rv5, bernoulli(const(0.1)))
  set_distr(particle, rv5, bernoulli(const(0.1)))
  S7 = sample(particle, get_distr(particle, rv5))
  set_distr(particle, rv5, delta(S7))
  wheel_slip = S7
  cmd2 = cmd_2
  if particle.state.value_expr(wheel_slip).v:
    x = prev_x
    x_2 = x
    map = map_2
    n_2 = x_2
    c0 = list_hd(map)
    c1 = list_hd(list_tl(map))
    o = c0
    rv8 = new_rv(particle)
    set_distr(particle, rv8, bernoulli(ifelse(o, sub(const(1.0), sensor_noise), sensor_noise)))
    set_distr(particle, rv8, bernoulli(get_p(get_distr(particle, rv8))))
    set_distr(particle, rv8, bernoulli(get_p(get_distr(particle, rv8))))
    set_distr(particle, get_par(particle, rv8, const(0)), bernoulli(const(0.5)))
    __siren_update_posterior_10 = bernoulli(div(times(get_p(get_distr(particle, get_par(particle, rv8, const(0)))), subst_rv(ifelse(rv8, get_p(get_distr(particle, rv8)), add(const(1), times(sub(const(0), const(1)), get_p(get_distr(particle, rv8))))), get_par(particle, rv8, const(0)), const(True))), ifelse(rv8, const(0.5), const(0.5))))
    __siren_update_marginal_10 = bernoulli(const(0.5))
    set_distr(particle, get_par(particle, rv8, const(0)), __siren_update_posterior_10)
    set_distr(particle, rv8, __siren_update_marginal_10)
    set_distr(particle, rv8, bernoulli(const(0.5)))
    observe_inner(particle, rv8, obs_3)
    (max_pos_2_2, map_2_2, xs_4) = pair(max_pos_2, pair(map_2, cons(x_2, xs)))
    (obs_3_2, cmd_3_2) = list_hd(list_tl(data))
    prev_x_2_2 = list_hd(xs_4)
    sensor_noise_2 = const(0.1)
    max_pos_4 = max_pos_2_2
    prev_x_3 = prev_x_2_2
    cmd_2_2 = cmd_3_2
    wheel_noise_2 = const(0.1)
    rv11 = new_rv(particle)
    set_distr(particle, rv11, bernoulli(wheel_noise_2))
    set_distr(particle, rv11, bernoulli(const(0.1)))
    set_distr(particle, rv11, bernoulli(const(0.1)))
    set_distr(particle, rv11, bernoulli(const(0.1)))
    S13 = sample(particle, get_distr(particle, rv11))
    set_distr(particle, rv11, delta(S13))
    wheel_slip_2 = S13
    cmd2_2 = const(0.0)
    if particle.state.value_expr(wheel_slip_2).v:
      x_3 = prev_x_3
      x_2_2 = x_3
      map_5 = map_2_2
      n_2_2 = x_2_2
      c0_2 = list_hd(map_5)
      c1_2 = list_hd(list_tl(map_5))
      o_2 = c0_2
      rv14 = new_rv(particle)
      set_distr(particle, rv14, bernoulli(ifelse(o_2, sub(const(1.0), sensor_noise_2), sensor_noise_2)))
      set_distr(particle, rv14, bernoulli(get_p(get_distr(particle, rv14))))
      set_distr(particle, rv14, bernoulli(get_p(get_distr(particle, rv14))))
      set_distr(particle, get_par(particle, rv14, const(0)), bernoulli(const(0.09999999999999998)))
      __siren_update_posterior_16 = bernoulli(div(times(get_p(get_distr(particle, get_par(particle, rv14, const(0)))), subst_rv(ifelse(rv14, get_p(get_distr(particle, rv14)), add(const(1), times(sub(const(0), const(1)), get_p(get_distr(particle, rv14))))), get_par(particle, rv14, const(0)), const(True))), ifelse(rv14, const(0.18), const(0.8200000000000001))))
      __siren_update_marginal_16 = bernoulli(const(0.18))
      set_distr(particle, get_par(particle, rv14, const(0)), __siren_update_posterior_16)
      set_distr(particle, rv14, __siren_update_marginal_16)
      set_distr(particle, rv14, bernoulli(const(0.18)))
      observe_inner(particle, rv14, obs_3_2)
      (max_pos_2_3, map_2_3, xs_5) = pair(max_pos_2_2, pair(map_2_2, cons(x_2_2, xs_4)))
      (_, map_4, xs_2) = pair(max_pos_2_3, pair(map_2_3, xs_5))
      xs_3 = list_rev(xs_2)
      __siren_tmp_2 = pair(map_4, xs_3)
    else:
      x_3 = add(prev_x_3, cmd2_2)
      x_2_2 = x_3
      map_5 = map_2_2
      n_2_2 = x_2_2
      c0_2 = list_hd(map_5)
      c1_2 = list_hd(list_tl(map_5))
      o_2 = c0_2
      rv14 = new_rv(particle)
      set_distr(particle, rv14, bernoulli(ifelse(o_2, sub(const(1.0), sensor_noise_2), sensor_noise_2)))
      set_distr(particle, rv14, bernoulli(get_p(get_distr(particle, rv14))))
      set_distr(particle, rv14, bernoulli(get_p(get_distr(particle, rv14))))
      set_distr(particle, get_par(particle, rv14, const(0)), bernoulli(const(0.09999999999999998)))
      __siren_update_posterior_16 = bernoulli(div(times(get_p(get_distr(particle, get_par(particle, rv14, const(0)))), subst_rv(ifelse(rv14, get_p(get_distr(particle, rv14)), add(const(1), times(sub(const(0), const(1)), get_p(get_distr(particle, rv14))))), get_par(particle, rv14, const(0)), const(True))), ifelse(rv14, const(0.18), const(0.8200000000000001))))
      __siren_update_marginal_16 = bernoulli(const(0.18))
      set_distr(particle, get_par(particle, rv14, const(0)), __siren_update_posterior_16)
      set_distr(particle, rv14, __siren_update_marginal_16)
      set_distr(particle, rv14, bernoulli(const(0.18)))
      observe_inner(particle, rv14, obs_3_2)
      (max_pos_2_3, map_2_3, xs_5) = pair(max_pos_2_2, pair(map_2_2, cons(x_2_2, xs_4)))
      (_, map_4, xs_2) = pair(max_pos_2_3, pair(map_2_3, xs_5))
      xs_3 = list_rev(xs_2)
      __siren_tmp_2 = pair(map_4, xs_3)
    __siren_tmp_4 = __siren_tmp_2
  else:
    x = add(prev_x, cmd2)
    x_2 = x
    map = map_2
    n_2 = x_2
    c0 = list_hd(map)
    c1 = list_hd(list_tl(map))
    o = c1
    rv8 = new_rv(particle)
    set_distr(particle, rv8, bernoulli(ifelse(o, sub(const(1.0), sensor_noise), sensor_noise)))
    set_distr(particle, rv8, bernoulli(get_p(get_distr(particle, rv8))))
    set_distr(particle, rv8, bernoulli(get_p(get_distr(particle, rv8))))
    set_distr(particle, get_par(particle, rv8, const(0)), bernoulli(const(0.5)))
    __siren_update_posterior_10 = bernoulli(div(times(get_p(get_distr(particle, get_par(particle, rv8, const(0)))), subst_rv(ifelse(rv8, get_p(get_distr(particle, rv8)), add(const(1), times(sub(const(0), const(1)), get_p(get_distr(particle, rv8))))), get_par(particle, rv8, const(0)), const(True))), ifelse(rv8, const(0.5), const(0.5))))
    __siren_update_marginal_10 = bernoulli(const(0.5))
    set_distr(particle, get_par(particle, rv8, const(0)), __siren_update_posterior_10)
    set_distr(particle, rv8, __siren_update_marginal_10)
    set_distr(particle, rv8, bernoulli(const(0.5)))
    observe_inner(particle, rv8, obs_3)
    (max_pos_2_2, map_2_2, xs_4) = pair(max_pos_2, pair(map_2, cons(x_2, xs)))
    (obs_3_2, cmd_3_2) = list_hd(list_tl(data))
    prev_x_2_2 = list_hd(xs_4)
    sensor_noise_2 = const(0.1)
    max_pos_4 = max_pos_2_2
    prev_x_3 = prev_x_2_2
    cmd_2_2 = cmd_3_2
    wheel_noise_2 = const(0.1)
    rv11 = new_rv(particle)
    set_distr(particle, rv11, bernoulli(wheel_noise_2))
    set_distr(particle, rv11, bernoulli(const(0.1)))
    set_distr(particle, rv11, bernoulli(const(0.1)))
    set_distr(particle, rv11, bernoulli(const(0.1)))
    S13 = sample(particle, get_distr(particle, rv11))
    set_distr(particle, rv11, delta(S13))
    wheel_slip_2 = S13
    cmd2_2 = cmd_2_2
    if particle.state.value_expr(wheel_slip_2).v:
      x_3 = prev_x_3
      x_2_2 = x_3
      map_5 = map_2_2
      n_2_2 = x_2_2
      c0_2 = list_hd(map_5)
      c1_2 = list_hd(list_tl(map_5))
      o_2 = c1_2
      rv14 = new_rv(particle)
      set_distr(particle, rv14, bernoulli(ifelse(o_2, sub(const(1.0), sensor_noise_2), sensor_noise_2)))
      set_distr(particle, rv14, bernoulli(get_p(get_distr(particle, rv14))))
      set_distr(particle, rv14, bernoulli(get_p(get_distr(particle, rv14))))
      set_distr(particle, get_par(particle, rv14, const(0)), bernoulli(const(0.09999999999999998)))
      __siren_update_posterior_16 = bernoulli(div(times(get_p(get_distr(particle, get_par(particle, rv14, const(0)))), subst_rv(ifelse(rv14, get_p(get_distr(particle, rv14)), add(const(1), times(sub(const(0), const(1)), get_p(get_distr(particle, rv14))))), get_par(particle, rv14, const(0)), const(True))), ifelse(rv14, const(0.18), const(0.8200000000000001))))
      __siren_update_marginal_16 = bernoulli(const(0.18))
      set_distr(particle, get_par(particle, rv14, const(0)), __siren_update_posterior_16)
      set_distr(particle, rv14, __siren_update_marginal_16)
      set_distr(particle, rv14, bernoulli(const(0.18)))
      observe_inner(particle, rv14, obs_3_2)
      (max_pos_2_3, map_2_3, xs_5) = pair(max_pos_2_2, pair(map_2_2, cons(x_2_2, xs_4)))
      (_, map_4, xs_2) = pair(max_pos_2_3, pair(map_2_3, xs_5))
      xs_3 = list_rev(xs_2)
      __siren_tmp_3 = pair(map_4, xs_3)
    else:
      x_3 = add(prev_x_3, cmd2_2)
      x_2_2 = x_3
      map_5 = map_2_2
      n_2_2 = x_2_2
      c0_2 = list_hd(map_5)
      c1_2 = list_hd(list_tl(map_5))
      o_2 = c0_2
      rv14 = new_rv(particle)
      set_distr(particle, rv14, bernoulli(ifelse(o_2, sub(const(1.0), sensor_noise_2), sensor_noise_2)))
      set_distr(particle, rv14, bernoulli(get_p(get_distr(particle, rv14))))
      set_distr(particle, rv14, bernoulli(get_p(get_distr(particle, rv14))))
      set_distr(particle, get_par(particle, rv14, const(0)), bernoulli(const(0.5)))
      __siren_update_posterior_16 = bernoulli(div(times(get_p(get_distr(particle, get_par(particle, rv14, const(0)))), subst_rv(ifelse(rv14, get_p(get_distr(particle, rv14)), add(const(1), times(sub(const(0), const(1)), get_p(get_distr(particle, rv14))))), get_par(particle, rv14, const(0)), const(True))), ifelse(rv14, const(0.5), const(0.5))))
      __siren_update_marginal_16 = bernoulli(const(0.5))
      set_distr(particle, get_par(particle, rv14, const(0)), __siren_update_posterior_16)
      set_distr(particle, rv14, __siren_update_marginal_16)
      set_distr(particle, rv14, bernoulli(const(0.5)))
      observe_inner(particle, rv14, obs_3_2)
      (max_pos_2_3, map_2_3, xs_5) = pair(max_pos_2_2, pair(map_2_2, cons(x_2_2, xs_4)))
      (_, map_4, xs_2) = pair(max_pos_2_3, pair(map_2_3, xs_5))
      xs_3 = list_rev(xs_2)
      __siren_tmp_3 = pair(map_4, xs_3)
    __siren_tmp_4 = __siren_tmp_3
  particle.finished = True
  return __siren_tmp_4

run(program, 1000)

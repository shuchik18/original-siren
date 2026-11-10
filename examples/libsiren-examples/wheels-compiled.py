from siren.libsiren import *


def new_make_obs1(particle, __siren_tmp_1, __siren_tmp_2):
 (left_wheel_rate, right_wheel_rate) = __siren_tmp_1
 (start, vs, os) = __siren_tmp_2
 prev_o = list_hd(os)
 prev_v = list_hd(vs)
 wb = const(2.0)
 sensor_err_l = const(1.0)
 sensor_err_r = const(0.95)
 omega_noise = const(2500.0)
 velocity_noise = const(2500.0)
 rv1 = new_rv(particle)
 set_distr(particle, rv1, gaussian(prev_o, omega_noise))
 omega = rv1
 rv3 = new_rv(particle)
 set_distr(particle, rv3, gaussian(prev_v, velocity_noise))
 velocity = rv3
 rv5 = new_rv(particle)
 set_distr(particle, rv5, gaussian(sub(velocity, times(wb, omega)), sensor_err_l))
 set_distr(particle, rv5, gaussian(add(velocity, times(sub(const(0.0), const(2.0)), omega)), const(1.0)))
 set_distr(particle, get_par(particle, rv5, const(1)), gaussian(div(add(times(add(div(velocity, const(10000.0)), rv5), const(0.9999000099990001)), times(sub(const(0.0), const(1.0)), velocity)), sub(const(0.0), const(2.0))), const(0.24997500249975002)))
 set_distr(particle, rv5, gaussian(velocity, const(10001.0)))
 set_distr(particle, get_par(particle, rv5, const(0)), gaussian(times(div(rv5, get_var(get_distr(particle, rv5))), const(2000.039996800256)), const(2000.039996800256)))
 set_distr(particle, rv5, gaussian(const(0.0), const(12501.0)))
 set_distr(particle, rv5, gaussian(const(0.0), const(12501.0)))
 observe_inner(particle, rv5, left_wheel_rate)
 rv9 = new_rv(particle)
 set_distr(particle, rv9, gaussian(add(velocity, times(wb, omega)), sensor_err_r))
 set_distr(particle, rv9, gaussian(add(get_mu(get_distr(particle, rv9)), times(const(2.0), get_mu(get_distr(particle, rv9)))), const(0.95)))
 set_distr(particle, rv9, gaussian(get_mu(get_distr(particle, rv9)), const(0.95)))
 observe_inner(particle, rv9, right_wheel_rate)
 return pair(const(False), pair(cons(velocity, vs), cons(omega, os)))


def program(particle):
 data = cons(pair(const(201.79897693), const(-32.779256728)), cons(pair(const(206.423095274), const(-31.9258541027)), cons(pair(const(207.900486767), const(-31.5701198412)), cons(pair(const(210.298303206), const(-32.0217383347)), cons(pair(const(206.010088006), const(-31.3001236526)), cons(pair(const(207.230561559), const(-35.9919009388)), cons(pair(const(207.802456076), const(-34.6068500903)), lst())))))))
 (_, vs, os) = fold(particle, new_make_obs1, data, pair(const(True), pair(cons(const(0.0), lst()), cons(const(0.0), lst()))))
 vs_2 = list_tl(list_rev(vs))
 os_2 = list_tl(list_rev(os))
 particle.finished = True
 return pair(vs_2, os_2)


run(program, 1000)





from siren.libsiren import *

def preprocess_data(particle, entry):
  left_wheel_rate = list_hd(entry)
  right_wheel_rate = list_hd(list_tl(entry))
  return pair(left_wheel_rate, right_wheel_rate)

def step(particle, __siren_tmp_1, __siren_tmp_2):
  (left_wheel_rate, right_wheel_rate) = __siren_tmp_1
  (start, vs, os) = __siren_tmp_2
  prev_o = list_hd(os)
  prev_v = list_hd(vs)
  wb = const(2.0)
  sensor_err_l = const(1.0)
  sensor_err_r = const(0.95)
  is_start = particle.state.value_expr(start).v
  omega_noise = (const(2500.0) if is_start else const(1.0))
  velocity_noise = (const(2500.0) if is_start else const(1.0))
  omega = assume(particle, gaussian(prev_o, omega_noise))
  velocity = assume(particle, gaussian(prev_v, velocity_noise))
  observe(particle, gaussian(sub(velocity, times(wb, omega)), sensor_err_l), left_wheel_rate)
  observe(particle, gaussian(add(velocity, times(wb, omega)), sensor_err_r), right_wheel_rate)
  return pair(const(False), pair(cons(velocity, vs), cons(omega, os)))

def program(particle):
  data = cons(pair(const(201.79897693), const(-32.779256728)), cons(pair(const(206.423095274), const(-31.9258541027)), cons(pair(const(207.900486767), const(-31.5701198412)), cons(pair(const(210.298303206), const(-32.0217383347)), cons(pair(const(206.010088006), const(-31.3001236526)), cons(pair(const(207.230561559), const(-35.9919009388)), cons(pair(const(207.802456076), const(-34.6068500903)), lst())))))))
  (_, vs, os) = fold(particle, step, data, pair(const(True), pair(cons(const(0.0), lst()), cons(const(0.0), lst()))))
  vs_2 = list_tl(list_rev(vs))
  os_2 = list_tl(list_rev(os))
  particle.finished = True
  return pair(vs_2, os_2)

run(program, 1000)

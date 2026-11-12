from siren.libsiren import *

def preprocess_data(particle, entry):
  obs = list_hd(list_tl(entry))
  obs_2 = (const(True) if particle.state.value_expr(eq(obs, const(1.0))).v else const(False))
  cmd = list_hd(list_tl(list_tl(entry)))
  return pair(obs_2, cmd)

def bernoulli_priors(particle, i):
  cell = assume(particle, bernoulli(const(0.5)))
  return cell

def move(particle, max_pos, prev_x, cmd):
  wheel_noise = const(0.1)
  wheel_slip = sample(particle, bernoulli(wheel_noise))
  cmd2 = (const(0.0) if particle.state.value_expr(lt(add(prev_x, cmd), const(0.0))).v else (const(0.0) if particle.state.value_expr(lt(max_pos, add(prev_x, cmd))).v else cmd))
  x = (prev_x if particle.state.value_expr(wheel_slip).v else add(prev_x, cmd2))
  return x

def find(particle, curr, __siren_tmp_1):
  (n, i, res) = __siren_tmp_1
  new_res = ((curr if particle.state.value_expr(eq(i, n)).v else res) if particle.state.value_expr(eq(res, const(-1.0))).v else res)
  return pair(n, pair(add(i, const(1.0)), new_res))

def get(particle, map, n):
  c0 = list_hd(map)
  c1 = list_hd(list_tl(map))
  return (c0 if particle.state.value_expr(eq(n, const(0.0))).v else c1)

def step(particle, __siren_tmp_2, __siren_tmp_3):
  (obs, cmd) = __siren_tmp_2
  (max_pos, map, xs) = __siren_tmp_3
  prev_x = (const(0.0) if particle.state.value_expr(eq(list_len(xs), const(0.0))).v else list_hd(xs))
  sensor_noise = const(0.1)
  x = move(particle, max_pos, prev_x, cmd)
  o = get(particle, map, x)
  observe(particle, bernoulli(ifelse(o, sub(const(1.0), sensor_noise), sensor_noise)), obs)
  return pair(max_pos, pair(map, cons(x, xs)))

def program(particle):
  data = cons(pair(const(0.0), const(1.0)), cons(pair(const(0.0), const(-1.0)), lst()))
  max_pos = const(1.0)
  map = list_map(particle, bernoulli_priors, list_range(0, add(max_pos, const(1.0))))
  (_, map_2, xs) = fold(particle, step, data, pair(max_pos, pair(map, lst())))
  xs_2 = list_rev(xs)
  particle.finished = True
  return pair(map_2, xs_2)

run(program, 1000)

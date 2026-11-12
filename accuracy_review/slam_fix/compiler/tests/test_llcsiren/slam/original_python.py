from siren.libsiren import *

def preprocess_data(particle, entry):
  obs = list_hd(list_tl(entry))
  obs_2 = (const(True) if eq(obs, const(1.0)) else const(False))
  cmd = list_hd(list_tl(list_tl(entry)))
  return pair(obs_2, cmd)

def bernoulli_priors(particle, i):
  cell = assume(particle, bernoulli(const(0.5)))
  return cell

def move(particle, max_pos, prev_x, cmd):
  wheel_noise = const(0.1)
  wheel_slip = assume(particle, bernoulli(wheel_noise))
  cmd2 = (const(0.0) if lt(add(prev_x, cmd), const(0.0)) else (const(0.0) if lt(max_pos, add(prev_x, cmd)) else cmd))
  x = (prev_x if wheel_slip else add(prev_x, cmd2))
  return x

def find(particle, curr, __siren_tmp_1):
  (n, i, found, res) = __siren_tmp_1
  found2 = (const(True) if eq(n, i) else found)
  res2 = (curr if eq(n, i) else res)
  return pair(n, pair(add(i, const(1.0)), pair(found2, res2)))

def get(particle, map, n):
  (_, __2, __3, res) = fold(particle, find, map, pair(n, pair(const(0.0), pair(const(False), const(-1.0)))))
  return res

def step(particle, __siren_tmp_2, __siren_tmp_3):
  (obs, cmd) = __siren_tmp_2
  (max_pos, map, xs) = __siren_tmp_3
  prev_x = (const(0.0) if eq(List.len(xs), const(0.0)) else list_hd(xs))
  sensor_noise = const(0.1)
  x = move(particle, max_pos, prev_x, cmd)
  o = get(particle, map, x)
  observe(particle, bernoulli(ifelse(o, sub(const(1.0), sensor_noise), sensor_noise)), obs)
  return pair(max_pos, pair(map, cons(x, xs)))

def program(particle):
  data = cons(cons(const(0.0), cons(const(1.0), lst())), cons(cons(const(0.0), cons(const(-1.0), lst())), lst()))
  max_pos = const(10.0)
  map = List.map(bernoulli_priors, list_range(0, add(max_pos, const(1.0))))
  (_, map_2, xs) = fold(particle, step, data, pair(max_pos, pair(map, lst())))
  x = list_hd(xs)
  particle.finished = True
  return pair(map_2, x)

run(program, 1000)

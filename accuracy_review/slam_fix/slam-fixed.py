from siren.libsiren import *

def _sirenc_sample_value(particle, distribution):
    # Realize only samples emitted by this backend; the legacy runtime's sample
    # function and every other generated example retain their existing behavior.
    rv = particle.state.assume(None, None, distribution)
    return particle.state.value_impl(rv)

def _sirenc_observe_bernoulli(particle, distribution, observation):
    import math
    state = particle.state
    observed = bool(state.value_expr(observation).v)
    parents = list(dict.fromkeys(distribution.p.rvs()))
    if len(parents) != 1:
        return observe(particle, distribution, observation)
    parent = parents[0]
    prior = state.distr(parent)
    if not isinstance(prior, Bernoulli) or not isinstance(prior.p, Const):
        return observe(particle, distribution, observation)
    yes = state.eval(distribution.p.subst_rv(parent, const(True)))
    no = state.eval(distribution.p.subst_rv(parent, const(False)))
    if not isinstance(yes, Const) or not isinstance(no, Const):
        return observe(particle, distribution, observation)
    probability = prior.p.v
    likelihood_true = yes.v if observed else 1.0 - yes.v
    likelihood_false = no.v if observed else 1.0 - no.v
    evidence = probability * likelihood_true + (1.0 - probability) * likelihood_false
    if evidence <= 0:
        particle.score = float('-inf')
        return
    particle.score += math.log(evidence)
    state.set_distr(parent, bernoulli(const(probability * likelihood_true / evidence)))

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
  wheel_slip = _sirenc_sample_value(particle, bernoulli(wheel_noise))
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
  _sirenc_observe_bernoulli(particle, bernoulli(ifelse(o, sub(const(1.0), sensor_noise), sensor_noise)), obs)
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

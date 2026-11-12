"""Conservative compilation of Bernoulli folds with sampled control flow.

A helper specialized with the first accumulator is not valid on later paths.
Retain these folds and emit scalar conjugate updates against current priors.
Other programs keep their existing compilation path.
"""
from dataclasses import fields, is_dataclass
from siren.grammar import Annotation, Fold, GenericOp, LetRV, Observe, Operator, Resample


def nodes(value):
    yield value
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from nodes(item)
    elif is_dataclass(value) and not isinstance(value, type):
        for field in fields(value):
            yield from nodes(getattr(value, field.name))


def needs_dynamic_bernoulli_fold(program):
    items = list(nodes(program))
    bindings = [node for node in items if isinstance(node, LetRV)]
    observations = [node for node in items if isinstance(node, Observe)]
    return (
        any(isinstance(node, Fold) for node in items)
        and any(node.annotation is Annotation.sample for node in bindings)
        and any(node.annotation is Annotation.symbolic for node in bindings)
        and bool(observations)
        and not any(isinstance(node, Resample) for node in items)
        and all(isinstance(node.distribution, GenericOp)
                and node.distribution.op is Operator.bernoulli for node in bindings)
        and all(isinstance(node.condition, GenericOp)
                and node.condition.op is Operator.bernoulli for node in observations)
    )


def _sirenc_sample_value(particle, distribution):
    # Realize only samples emitted by this backend; the legacy runtime's sample
    # function and every other generated example retain their existing behavior.
    rv = particle.state.assume(None, None, distribution)
    return particle.state.value_impl(rv)


def _sirenc_observe_bernoulli(particle, distribution, observation):
    import math
    from siren.grammar import Bernoulli, Const
    from siren.libsiren import bernoulli, const, observe
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

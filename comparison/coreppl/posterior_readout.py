"""Conservative output adapter for cppl --print-model output.

This is NOT a conjugacy transformation. The original compiler supplies every
posterior parameter. Only an immediately returned, outermost Beta/Gaussian draw
can be replaced; all preceding observations, computations and draws stay verbatim.
An immediately returned scalar fold can also defer its callback's terminal draw
until the next iteration needs it, then query the final distribution's moments.
Unrecognized output shapes are left alone. No reference answers are imported.
"""

import re


_NAME = r"[A-Za-z_][A-Za-z_0-9']*"
_TERMINAL = re.compile(
    rf"^let (?P<name>{_NAME}) = assume\s+"
    rf"\((?P<family>Beta|Gaussian) (?P<first>{_NAME}) (?P<second>{_NAME})\)"
    rf" in\s*(?P=name)\s*\Z",
    re.MULTILINE,
)

# Recognize the complete extracted scalar-fold tail, including its uses. Do not
# adapt folds whose result feeds further observations or other computation.
_SCALAR_FOLD = re.compile(
    rf"^let (?P<fold>{_NAME}) = foldl in\n"
    rf"let (?P<step>{_NAME}) =\n  lam (?P<previous>{_NAME})\.\n"
    rf"    let (?P<inner>{_NAME}) =\n      lam (?P<obs>{_NAME})\.\n"
    rf"(?P<body>(?:^ {{8}}[^\n]*\n)*)"
    rf"        let (?P<draw>{_NAME}) = assume\s+"
    rf"\(Gaussian (?P<mu>{_NAME}) (?P<sigma>{_NAME})\) in\n"
    rf"        (?P=draw)\n    in\n    (?P=inner)\nin\n"
    rf"let (?P<apply>{_NAME}) = (?P=fold) (?P=step) in\n"
    rf"let (?P<start>{_NAME}) = (?P=apply) (?P<initial>{_NAME}) in\n"
    rf"let (?P<result>{_NAME}) = (?P=start) (?P<data>{_NAME}) in\n"
    rf"(?P=result)\s*\Z", re.MULTILINE,
)


def adapt_final_scalar_fold(model):
    """Query only the last callback's distribution; earlier states still draw.

    The accumulator carries a sampling thunk and its conditional moments. Each
    subsequent callback invokes that thunk before its original computations.
    Thus the last draw disappears, but every draw consumed by another iteration
    and every compiler-generated update is preserved. No loop is unrolled.
    """
    match = _SCALAR_FOLD.search(model)
    fresh = "readoutPendingState"
    if match is None or re.search(rf"\b{fresh}\b", model):
        return None, {"applied": False, "reason": "no supported immediately returned scalar fold"}
    g = match.groupdict()
    distribution = f"(Gaussian {g['mu']} {g['sigma']})"
    moments = f"(expectation {distribution}, mulf {g['sigma']} {g['sigma']})"
    callback = (f"let {g['step']} =\n  lam {fresh}.\n"
                f"    let {g['previous']} = {fresh}.0 () in\n"
                f"    let {g['inner']} =\n      lam {g['obs']}.\n" + g['body'] +
                f"        ((lam. assume {distribution}), {moments})\n"
                f"    in\n    {g['inner']}\nin\n")
    replacement = (f"let {g['fold']} = foldl in\n" + callback +
                   f"let {g['apply']} = {g['fold']} {g['step']} in\n"
                   f"let {g['start']} = {g['apply']} ((lam. {g['initial']}), ({g['initial']}, 0.0)) in\n"
                   f"let {g['result']} = {g['start']} {g['data']} in\n{g['result']}.1\n")
    return model[:match.start()] + replacement, {
        "applied": True, "kind": "final_scalar_fold", "family": "Gaussian",
        "distribution": distribution, "removed_variable": g['draw'],
        "unchanged_prefix_characters": match.start(), "unchanged_callback_body": g['body'],
        "original_tail": match.group(), "replacement_tail": replacement,
        "reason": "draw intermediate states on entry to the next iteration; query only the final conditional distribution",
    }


def adapt_posterior_readout(model):
    adapted, audit = adapt_terminal_draw(model)
    if adapted is not None:
        return adapted, audit
    return adapt_final_scalar_fold(model)


def adapt_terminal_draw(model):
    """Return (adapted model or None, audit details).

The column-zero anchor deliberately relies on the compiler's pretty-printer:
draws inside callbacks, matches, and loops are indented, and are never eligible.
The exact tail grammar rules out subsequent observations or uses of the draw.
Parameters must already be ANF-bound names; arbitrary expressions are rejected.
"""
    match = _TERMINAL.search(model)
    if match is None:
        return None, {"applied": False, "reason": "no supported outermost immediately returned draw"}
    family, first, second = match.group("family", "first", "second")
    distribution = f"({family} {first} {second})"
    if family == "Gaussian":
        variance = f"(mulf {second} {second})"
    else:
        total = f"(addf {first} {second})"
        variance = f"(divf (mulf {first} {second}) (mulf (mulf {total} {total}) (addf {total} 1.0)))"
    replacement = f"((expectation {distribution}), {variance})\n"
    prefix = model[:match.start()]
    adapted = prefix + replacement
    return adapted, {
        "applied": True, "family": family, "removed_variable": match.group("name"),
        "distribution": distribution, "original_tail": match.group(),
        "replacement_tail": replacement, "unchanged_prefix_characters": len(prefix),
        "remaining_assume_sites": len(re.findall(r"\bassume\b", prefix)),
        "reason": "query the terminal compiler-generated distribution without drawing from it",
    }

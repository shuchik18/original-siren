# SLAM compiler accuracy diagnosis

SLAM was freshly regenerated from
`/Users/skundaliya/Documents/SirenC/SirenC/tests/test_llcsiren/slam-example/source.si`
using the current SirenC evaluator and `llcsiren.src.compiler`. The resulting
`generated.py` matches the benchmark's `slam-compiled.py` byte-for-byte. Both
compiler stages were executed read-only, with outputs confined to this directory.
Existing compiler modifications were preserved.

## Where accuracy is lost

1. SirenC's `siren/evaluate.py`, Fold case (lines 821–963), specializes one SLAM
   iteration using the initial empty trajectory and emits a helper reused for
   the remaining iterations. The helper contains `prev_x = const(0.0)` and
   `cmd2 = cmd`; the actual source carries the last position and checks movement
   bounds. Those runtime-dependent decisions have been removed before llcsiren
   emits Python. The scalar-Gaussian runtime-parameter guard does not apply to
   this accumulator shape.
2. The generated Bernoulli marginal/update embeds `const(0.5)` for the false-cell
   prior term. That is correct for the initial prior, but not when the selected
   cell has already been conditioned. The current value `1 - p` is required.
   The Bernoulli formulas in `siren/compiler/conjugate.py` initially construct
   `1 - p`, but compile-time arithmetic simplification uses the initial abstract
   probability and the resulting expression is reused in the loop. Fixing only
   that literal will not fix trajectory semantics or control-flow lowering.
3. `llcsiren/src/compiler.py`, `_emit_condition` (lines 393–396), unwraps symbolic
   conditions only in the optional linear-Gaussian mode. Normal lowering emits
   `if wheel_slip` and `if eq(...)` against symbolic Python objects. Their object
   truthiness is not their probabilistic value. Independently, libsiren's current
   `sample` returns an unevaluated RandomVar rather than a realized sample.

The durable repair belongs in the compiler transformation/lowering and sampling
contract, followed by regeneration. Hand-editing the generated benchmark model
or changing the RMSE/reference would hide the compiler discrepancy.

## Is simplify_observed necessary for accuracy?

No such helper is needed to repair the issues above. It was an expression
simplification optimization and is absent from the active libsiren runtime.
It cannot restore omitted trajectory state, a stale prior term, or incorrectly
lowered Python control flow. A semantics-preserving simplifier should not change
the intended posterior.

To check this, `original_generated.py` translates the ORIGINAL source directly,
without SirenC's prior symbolic specialization. The isolated diagnostic adapts
only Python condition value extraction, supplies all four possible realized
motion decisions, and uses the existing libsiren/SSI conjugacy and mean readout.
It does not call simplify_observed, optimize the readout, rewrite the model's
probability equations, or use reference answers for inference.

Enumerating motion decisions integrates their Bernoulli prior exactly; map cells
remain symbolic and are conditioned by the existing runtime. The original-source
translation's posterior mean agrees with the exact reference to 2.8e-17 and its
log evidence agrees to 2.3e-16. This is an isolated correctness diagnostic, not an
installed compiler repair or a speed comparison.

| Diagnostic case | Maximum posterior mean error | Absolute log-evidence error |
|---|---:|---:|
| Transformed code as emitted, with motion decisions supplied | 0.8537803138 | 0.4275680535 |
| Transformed code with value-based Python conditions | 1.0641932910 | 0.3492293196 |
| Original-source translation with value-based Python conditions | 2.78e-17 | 2.22e-16 |

Correcting Python condition lowering alone is insufficient; the transformed
SLAM helper still has the wrong trajectory and prior update.

## Reproduce

From `/Users/skundaliya/Documents/libsiren/original-siren`:

```sh
../.venv11/bin/python accuracy_review/llcsiren_slam/check_accuracy.py
```

`accuracy_diagnostic.json` retains the conditional means and log likelihood for
each motion path. `accuracy_diagnostic.csv` provides the aggregate comparison.
`transformed.si`, `generated.py`, `original_generated.py`, `compile.log`, and
`metadata.json` preserve the pipeline evidence and source hashes.

No compiler source, generated benchmark example, runtime, benchmark, or inference
interface was changed during this investigation. `interface.py` remains identical
to Git HEAD.

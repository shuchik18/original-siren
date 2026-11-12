# Current Kalman/SLAM performance fix (no additional implementation files)

The CSE emission pass and its `llcsiren/src/optimize.py` module have been removed.
The generated examples contain no `__siren_cse_*` assignments. The implementation
now uses the existing compiler files:

- `siren/compiler/conjugate.py`: specialize known Gaussian affine offsets and
  observation variances; simplify Bernoulli likelihoods conditional on each
  parent value while retaining runtime expressions for unknown values.
- `siren/compiler/interface.py`: simplify multiplication by a concrete one.
- `siren/evaluate.py`: retain the checked terminal scalar filter optimization.
- `llcsiren/src/compiler.py`: use the existing emission and temporary-inlining
  pipeline, with no optimizer-module import.

Kalman, SLAM and outlier were regenerated through normal symbolic evaluation
and llcsiren emission. Uncompiled SLAM retains the Boolean correctness repair.
Existing `__siren_update_marginal_*` temporaries capture a marginal distribution
before a parent update; they preserve the required state-update ordering.

Fresh end-to-end measurements: 1,000 particles, five warmups, 20 measured runs,
same Python environment and numerical library thread limits. Arithmetic mean
and sample standard deviation, including process startup and imports:

| Model | Uncompiled SirenC (s) | Compiled SirenC (s) | Mean time reduction |
|---|---:|---:|---:|
| Kalman | 0.647612 ± 0.011583 | 0.451331 ± 0.010216 | 30.3% |
| SLAM | 0.237302 ± 0.003249 | 0.214223 ± 0.014689 | 9.7% |

The timing results are recorded above. The added benchmark-output directory,
CSE-based performance report and staging helpers have been removed; their
earlier timing numbers describe the replaced implementation.

Regression checks were merged into the existing `tests/test_sirenc_conjugacy.py`.
28 targeted checks pass, with one strict expected failure: the non-unit affine
filter variant differs from original Siren in the pre-existing compiler as well
as after this simplification (both returned 2.001862197392924 in the diagnostic,
versus Siren's 2.7316611838167364). This is a separate existing correctness issue.
Checks cover exact Kalman mean, variance and evidence; all SLAM motion branches;
changed priors, sensor/process noise and data; nonterminal dependencies; absence
of CSE names; and reproduction through the installed compiler.

```sh
/Users/skundaliya/Documents/SirenC/SirenC/.venv/bin/python -B -m pytest -p no:cacheprovider -q tests/test_sirenc_conjugacy.py tests/test_ssi.py tests/test_benchmark_sirenc.py
```

# Earlier SLAM/outlier accuracy fix

The fix is installed in SirenC's normal compilation pipeline:
`symbolic evaluation -> residual trace -> llcsiren -> libsiren`.
The rejected evaluator early return and special-case llcsiren inference backends
have been removed. The evaluator does not import llcsiren or bypass evaluation.

## Changes

- `SirenC/siren/evaluate.py`: widen sampled folds' abstract input values and
  accumulator parameters so reusable fold functions read current values rather
  than the first iteration's values. Empty growing histories remain unknown.
  Recognize pure list operations when specializing dynamic conditions.
- In the same evaluator, a terminal scalar Gaussian filter integrates out a dead
  previous state using its current mean and variance. This optimization is limited
  to the checked filter structure; trajectories and smoothing keep their graph.
  Standard compiler conjugacy still emits the observation updates.
- `SirenC/llcsiren/src/compiler.py`: read evaluated Boolean values for control
  flow. Keep conditionals symbolic inside residual Beta constructors, protecting
  coin's probability updates.
- Regenerate only SLAM/outlier's existing Python outputs in both checkouts.
  Their sole import is `from siren.libsiren import *`; no injected runtime helpers,
  extra imports, or `isinstance` calls appear in these outputs.

No interface, conjugacy, or SSI implementation file was changed for this fix.
The existing libsiren sampler correction is retained: explicit samples realize
values using the particle RNG, with symbolic-parent conditioning preserved.
No `simplify_observed` step was added. No new files were created by this fix.

The subsequent evaluator refactor places sampling detection, abstract-value
generalization, and temporary filter-body preparation inside the `Fold` case.
It retains the enclosing expression solely to check whether that exact fold is
terminal. Prediction distributions use temporary compile-time context bindings;
ordinary `Identifier` and `LetRV` evaluation handles the rewritten body. The
general `LetRV` case is restored, and `_fold_filter_parent` is removed.
All eight examples' direct and transformed generated Python hashes match the
version before this refactor. Existing generated benchmark files need no update.

## Validation

22 checks pass in the existing benchmark, SSI, and conjugacy test files. They
cover exhaustive SLAM/outlier branch combinations (posterior means and evidence),
repeated SLAM cell observations, actual normal-pipeline reproduction of the two
artifacts, and variants with changed priors, data, noise, and function names.
Coin and a hybrid variant with a nonzero observation also match original Siren's
means and per-particle observation scores.

The other six benchmark artifacts match their saved baseline hashes. Gaussian,
gtree, kalman, and wheels also regenerate byte-for-byte identically through the
normal symbolic pipeline. Regenerated coin and hybrid code receives the Boolean/
symbolic-expression corrections above and passes the Siren comparison checks.

SLAM and outlier each completed five paired repetitions at 1, 10, 100, 1000, and
10000 particles (50 paired runs total), with matching seeds and timing boundaries.
The maximum estimate difference from original Siren was 6.8e-13.
Compiled median inference times were 2.04–2.90 times faster across these measurements.

| Model | Particles | Siren median (ms) | Compiled median (ms) | Maximum estimate difference |
|---|---:|---:|---:|---:|
| slam | 1 | 1.221 | 0.421 | 0 |
| slam | 10 | 10.422 | 3.772 | 2.22e-16 |
| slam | 100 | 89.471 | 37.336 | 1.55e-15 |
| slam | 1000 | 841.005 | 394.936 | 2.7e-14 |
| slam | 10000 | 8617.553 | 4108.772 | 1.96e-13 |
| outlier | 1 | 2.210 | 1.086 | 0 |
| outlier | 10 | 22.677 | 10.179 | 3.55e-15 |
| outlier | 100 | 202.233 | 88.721 | 1.07e-14 |
| outlier | 1000 | 2056.908 | 920.953 | 6.57e-14 |
| outlier | 10000 | 21974.627 | 9431.419 | 6.8e-13 |

These timings cover the checked workloads and do not guarantee performance for
arbitrary programs. Per-run measurements are in the existing
`accuracy_review/slam_fix/validation.json`.

Existing benchmark CSVs and plots were not refreshed and may reflect older
artifacts. Other files under `accuracy_review` include historical and rejected
implementations; the installed sources and regenerated outputs define the fix.

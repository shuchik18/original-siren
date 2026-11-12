# Consistent Siren / SirenC rerun and three-way comparison

Every Siren and SirenC point is from one complete sequential benchmark session
on November 12, 2026. All eight models were rerun; no earlier Outlier, SLAM,
or other Siren/SirenC measurements are mixed into these graphs.

Seven models use 1, 10, 100, and 1,000 particles; Wheels uses one particle.
Every engine/model/particle group contains five successful measured repetitions.
Each worker has an independent untimed warm-up. Engine order alternates between
repetitions. Both engines use paired seed labels and independent particle RNG
streams, a shared Python environment, and one thread per numerical library.
One worker runs at a time; all plotting occurs after measurement. Normal apps
and OS services remain active, so a completely idle machine is not claimed.

CorePPL's existing measurements were retained at the user's explicit request.
They were measured separately; its compiler toolchain is unavailable locally.
The matched figure uses identical particle counts across all three systems.
The all-counts figure also shows CorePPL's separately measured 10,000-particle
points and larger Wheels counts. No corresponding Siren/SirenC points were
invented. All plotted points summarize five runs.

The horizontal axis is median inference time in milliseconds (left is faster).
The vertical axis is RMS standardized posterior-mean error across repetitions.
Errors below 1e-12 are floored for display only. Timing and error are both
recomputed from the same raw runs for every Siren/SirenC point.

Siren/SirenC timing includes particle creation, execution, and weighted mean
evaluation, and excludes startup, imports, parsing, error scoring, and plotting.
CorePPL's recorded inference timer excludes final mixture aggregation. These
timer boundaries and separately recorded machine conditions preclude a fully
controlled three-system speedup claim.

CorePPL static gtree retains its documented posterior bias. CorePPL readout uses
retained moments where available, and samples for Wheels, Outlier, SLAM, and
the hybrid. Siren/SirenC use symbolic component means. The hybrid's zero mean
cannot verify mixture weights. Only returned posterior means are compared here;
matching means does not establish equal posterior distributions or evidence.

Run `python plot_comparison.py` to regenerate both PNG/SVG figures from the
bundled CSVs. `data/sequential_raw.csv` preserves the complete 290-run session.
`validation.json` independently checks every timing/error aggregate and particle
intersection. Original implementation sources were unchanged during the run.

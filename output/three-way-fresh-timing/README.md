# Three-way comparison with fresh Siren / SirenC timings

The six refreshed models use the sequential paired benchmark from November 12,
2026: coin, gaussian-gaussian, gtree, kalman, gaussian-gaussian-hybrid, and wheels.
Both timing and returned-mean error are recomputed from the fresh raw runs.
No older particle counts are appended to these six models.

Fresh smaller models have 1, 10, 100, and 1,000 particles with five runs per point.
Wheels has one particle with three runs per engine. Outlier and SLAM were not
rerun; their panels retain the earlier results.
All CorePPL data is unchanged historical data, measured separately.

The matched graph uses the intersection of particle counts for each model:
1, 10, 100, 1,000 for the five refreshed smaller models; 1 for Wheels; and
1, 10, 100, 1,000, 10,000 for historical Outlier and SLAM.
The all-counts graph additionally shows historical CorePPL counts without
inventing Siren / SirenC measurements at unmeasured counts.

The x-axis is median inference time in milliseconds, so farther left is faster.
The y-axis is the RMS across runs of standardized posterior-mean error.
Values below 1e-12 are floored only for display; CSVs retain the actual values.
CorePPL static gtree retains recorded posterior bias. The hybrid's zero means
cannot verify mixture weights. Readout types and timer boundaries differ among
systems; no controlled three-system speedup is claimed.

Fresh Siren/SirenC runs used one worker at a time, alternating engine order,
paired seeds, independent untimed warm-ups, and single-thread numerical libraries.
Normal apps and OS services remained active. Their timer included particle
creation, execution, and weighted means; imports, parsing, startup, and plotting
were excluded. CorePPL's recorded timer excludes final mixture aggregation.

Run `python plot_comparison.py` to regenerate PNG/SVG graphs and processed CSVs.
The script reads only bundled data snapshots. Original CSVs are preserved as
`data/historical_siren_raw.csv` and `data/historical_sirenc_raw.csv`.
`data/sequential_raw.csv` preserves the full fresh paired measurements.
`source_manifest.json` records input provenance. Each plotted point also carries
`run_origin` in the processed CSVs. No benchmark or inference source was modified.

The one-particle points for the five smaller models were measured in a subsequent
sequential session on the same day, with identical source hashes and the same
benchmark protocol. All 50 supplemental runs succeeded. Existing larger-particle
measurements and Wheels results were preserved. `data/particle_one_raw.csv` and
`particle_one_validation.json` record these supplemental runs.

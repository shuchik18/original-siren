# Sequential Siren / SirenC comparison

Measured November 12, 2026. All 156 measured runs succeeded. No inference or
compiler sources were changed for this experiment.

Before launch, the process inventory showed no competing Python benchmark
workers. Normal macOS services and applications remained active (including
PyCharm, Codex, and Defender), so this is not a fully idle-machine experiment.
Background load was not eliminated, and this experiment does not isolate it as
the cause of differences from historical runs.

One worker ran at a time. Both engines used the same Python environment, inputs,
particle counts, and paired seeds. The engine running first alternated between
repetitions. Each worker performed one independent untimed warm-up. Numerical
libraries were limited to one thread for both engines. Timing included particle
creation, inference, and weighted posterior mean calculation, excluding imports,
startup, parsing, and plotting. All graphs were rendered after timing finished.

Five repetitions were measured at 10, 100, and 1,000 particles for each smaller
model. Wheels used three repetitions at one particle. Configuration, load
averages before and after each worker, source hashes, per-run measurements,
estimates, and errors are preserved in this directory.

Representative arithmetic means (1,000 particles, except Wheels at 1):

| Model | Siren mean (ms) | SirenC mean (ms) | SirenC execution time |
|---|---:|---:|---|
| Gaussian–Gaussian | 76.560 | 38.213 | 50.1% lower |
| gtree | 210.494 | 134.756 | 36.0% lower |
| Kalman | 862.770 | 951.356 | 10.3% higher |
| Coin | 290.695 | 230.975 | 20.5% lower |
| Gaussian–Gaussian–hybrid | 84.552 | 46.960 | 44.5% lower |
| Wheels | 43,494.922 | 41,212.995 | 5.2% lower |

SirenC's arithmetic mean and median were lower for all measured particle counts
in five of the six models. Kalman remained slower at every measured count.
The maximum difference between paired posterior mean estimates was 3.553e-15.
This checks returned means only, not full posterior distributions or evidence;
the hybrid's zero means cannot detect incorrect mixture weights. Wheels' maximum
standardized mean RMSE against the reference was 7.259e-12 in both engines.
The source validation found no changed sources during the experiment.

- `raw.csv`: all measured runs, load averages, estimates, and worker wall times.
- `summary.csv`: arithmetic means, medians, sample standard deviations, errors.
- `speedup.csv`: Siren time / SirenC time; above one favors SirenC.
- `mean_ms.png` / `.pdf`: mean time graphs; error bars are one sample SD.
- `median_ms.png` / `.pdf`: median time graphs.
- `run_comparison.py`: reproducible driver; running it replaces this directory's results.

Run from the repository with the environment used for this experiment:

```sh
/Users/skundaliya/Documents/SirenC/SirenC/.venv/bin/python output/sequential-comparison-2026-11-12/run_comparison.py
```

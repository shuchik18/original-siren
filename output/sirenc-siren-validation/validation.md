# Paired SirenC / Siren validation

Five repetitions at 100 particles per model, independent matching particle RNG streams, one excluded warm-up per repetition. Times include creation, execution, and weighted means. These local measurements do not establish a speedup at every budget. Wheels is validated separately against its exact posterior; it is excluded from this paired table.

| Model | SirenC median ms | Siren median ms | Speedup | SirenC standardized RMSE | Siren standardized RMSE | Maximum paired mean difference |
|---|---:|---:|---:|---:|---:|---:|
| gaussian-gaussian | 8.656 | 30.064 | 3.47× | 9.3153e-16 | 1.8631e-15 | 8.88e-16 |
| gtree | 43.244 | 58.590 | 1.35× | 9.7255e-17 | 2.9176e-16 | 3.55e-15 |
| kalman | 123.170 | 130.482 | 1.06× | 2.2596e-15 | 6.7787e-15 | 3.55e-15 |
| slam | 12.664 | 72.237 | 5.70× | 0.086478 | 0.086478 | 1.55e-15 |
| outlier | 37.375 | 152.789 | 4.09× | 0.0081986 | 0.0081986 | 1.24e-14 |
| coin | 67.455 | 47.324 | 0.70× | 1.8411e-15 | 4.6027e-15 | 7.77e-16 |
| gaussian-gaussian-hybrid | 5.842 | 10.699 | 1.83× | 0 | 0 | 0 |

Values above 1× indicate SirenC is faster. Coin is slower in this measurement. The comparison uses the local reproducible benchmark_siren.py adapter, rather than the external compare_libsiren.py OS-entropy seed policy. CorePPL was inspected but not executed in this validation.

The complete corrected SirenC sweep is in ../sirenc-corrected/.

# Siren
Siren is a first-order functional probabilistic programming language, implemented with the hybrid inference interface, with support for inference plans. Distributions encoding annotations can be added to random variables to select the representation of the variable's distribution to be used during inference. Siren is also equipped with the inference plan satisfiability analysis, which statically infers if the annotated inference plan is satisfiable. 

The Siren interpreter, including the inference plan satisfiability analysis, is implemented with semi-symbolic inference, delayed sampling, and SMC with belief propagation. It can be extended with other hybrid inference algorithms that can implement the hybrid inference interface. 

## Install
Checkout the repository on the Artifact Evaluation branch:
```
git clone https://github.com/psg-mit/siren -b oopsla-artifact
cd siren
```

### Docker (Recommended)
We also provide a Docker file to build an image. The artifact was tested with Docker Desktop v4.30.0. 
```bash
docker build -t siren .
docker run -it siren
```

### From Source
This software was tested on M1 MacBook and requires Python >= 3.10. To install dependencies:
```bash
pip install -e .
```

## Quickstart
Here is an example of a Siren modeling a Kalman filter:
```ocaml
val make_observations = fun (yobs, xs) ->
  let pre_x = List.hd(xs) in
  let sample x <- gaussian(pre_x, 1.) in
  let () = observe(gaussian(x, 1.), yobs) in

  let () = resample() in
  cons(x, xs)
in

let data = List.range(1, 101) in

let x0 = 0 in
let xs = fold(make_observations, data, [x0]) in
List.rev(xs)
```
The program iterates over a range of values from 1 to 100 (inclusive) as the observed data. The variable `x` is annotated with `sample` to indicate `x` should be represented as samples during inference. To indicate `x` should be represented as a symbolic distribution, replace `sample` with `symbolic`.

To run the inference plan satisfiability analysis, and execute the program if the analysis succeeds:
```bash
siren path/to/program.si -m {method} -p {particles} --analyze
```

For example, to run the analysis and execute the program using the semi-symbolic inference algorithm with 100 particles:
```bash
siren examples/kalman.si -m ssi -p 100 --analyze
```
The analysis will throw an `AnalysisViolatedAnnotationError` and abort without running the program if the annotated inference plan is not satisfiable.

To execute without the analysis:
```bash
siren path/to/program.si -m {method}
```

To run the analysis only:
```bash
siren path/to/program.si -m {method} --analyze-only
```

## Tests
To run the test suite for a quick check everything works:
```bash
python -m pytest tests/
```

## Kick the Tires
To do a smoke test that the benchmarks can run correctly:
```bash
cd benchmarks/
python harness.py kicktires
```
This should take ~10 minutes.

To check the visualization script works correctly:
```bash
python visualize.py --task table
python visualize.py --task plot
```
The plots will be located at `benchmarks/outlier/{BENCHMARK}/{METHOD}_particles.png` for each BENCHMARK and METHOD.

## Benchmarking
The experiments from the paper were conducted on a 60-core Intel Xeon Cascade Lake (up to 3.9 GHz) node with 240 GB RAM. The full set of experiments in the paper takes about 23 days of computation. The experiments can run on a general-purpose computer as well, requiring only enough computation time. 

### Replicating Trend
Due to the long amount of time needed to compute the full set of benchmarks from the paper, which uses `n=100` iterations per particle setting for each benchmark and method, to only replicate the trends of the main paper figures:
```bash
cd benchmarks/
python harness.py artifact-eval
```
This executes the example for Figure 4 for `n=10`, the programs for Figure 15 for `n=5`, and the programs in the appendix for `n=1`. This will take ~4-5 hours.

Then, to visualize the results for Section 2 Figure 4:
```bash
python visualize.py --example
```
The plot will be located at `benchmarks/example/output/ssi_example_1245.png`.

Then, to visualize the results for Section 5 Figure 15:
```bash
python visualize.py --task plot -b outlier noise
```
The plot will be located at `benchmarks/outlier/output/ssi_particles.png` and `benchmarks/noise/output/ssi_particles.png`.

Then, to produce Section 5 Table 1:
```bash
python visualize.py --task table
```

To visualize the results of Appendix E: 
```bash
python visualize.py --task plot
```
The plot will be located at `benchmarks/outlier/{BENCHMARK}/{METHOD}_particles.png` for each BENCHMARK and METHOD.

### Full Replication
To perform the full replication of the figures in the paper:
```bash
cd benchmarks/
python harness.py full-replication
python visualize.py --example
python visualize.py --task table
python visualize.py --task plot
```
This will take ~23 days.

## Syntax
The Siren syntax can be found in `siren/parser.py`.

## Guide
To view a description of the source files in this repository and instructions on how to extend Siren, please see [DESCRIPTION.md](DESCRIPTION.md).

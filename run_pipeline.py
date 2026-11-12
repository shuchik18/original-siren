# libsiren/run_pipeline.py
import argparse
import sys
import os
from pathlib import Path
import time
import csv
import statistics
import subprocess
import json
import math

REFERENCE_MEAN = (
    85.29496564509887, 86.92981152501379, 87.91619939397266,
    88.38972043747494, 87.3451685893735, 86.25552531943156,
    86.48589176818021, -58.74720153468439, -59.53081176585215,
    -59.88469714980448, -60.4047438909965, -59.56455646926765,
    -60.67132597885489, -60.608538923626305,
)


def validate_wheels_mean(value):
    def flatten(item):
        if isinstance(item, (list, tuple)):
            return [number for part in item for number in flatten(part)]
        return [float(item)]

    values = flatten(value)
    if len(values) != len(REFERENCE_MEAN):
        raise ValueError(f'Wheels returned {len(values)} means; expected 14')
    if not all(math.isfinite(x) for x in values):
        raise ValueError('Wheels returned a non-finite posterior mean')
    error = max(abs(x-y) for x, y in zip(values, REFERENCE_MEAN))
    if error > 1e-7:
        raise ValueError(f'Wheels posterior means differ from exact reference by {error:.9g}')
    return values, error


# Complete default target array list matching your workspace layout
DEFAULT_EXAMPLES = [
    "example1", "example-1-compiled", "example-5-compiled", 
    "example5", "example9", "example10", "example11", "bernoulli-bernoulli-compiled",
    "example-9-compiled", "example-10-compiled", "example-11-compiled", "example12", "example12-compiled", "outlier", "outlier-compiled", "gaussian-gaussian", "gaussian-gaussian-compiled", "kalman", "kalman-compiled", "gtree", "gtree-compiled", "slam", "slam-compiled", "wheels", "wheels-compiled", "coin", "coin-compiled", "gaussian-gaussian-hybrid", "gaussian-gaussian-hybrid-compiled"
]

# Each subprocess removes existing entry calls and runs the program exactly once.
PYTHON_RUNNER = """
import ast
from pathlib import Path
import sys
import json
import numpy as np

target = Path(sys.argv[1])
tree = ast.parse(target.read_text(), filename=str(target))
def is_entry_call(node):
    if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
        return False
    call = node.value
    if isinstance(call.func, ast.Name) and call.func.id == "print":
        if len(call.args) != 1 or not isinstance(call.args[0], ast.Call):
            return False
        call = call.args[0]
    return (isinstance(call.func, ast.Name) and call.func.id == "run"
            and bool(call.args) and isinstance(call.args[0], ast.Name)
            and call.args[0].id == "program")
tree.body = [node for node in tree.body if not is_entry_call(node)]
namespace = {"__name__": "__benchmark__", "__file__": str(target)}
exec(compile(tree, str(target), "exec"), namespace)
particles = namespace["run"](namespace["program"], int(sys.argv[2]))
def flatten_mean(value, state):
    if isinstance(value, (tuple, list)):
        return [number for item in value for number in flatten_mean(item, state)]
    return [float(state.mean(value))]
values = np.asarray([flatten_mean(p.cont, p.state) for p in particles])
scores = np.asarray([p.score for p in particles])
if (np.any(np.isnan(scores)) or np.any(np.isposinf(scores))
        or not np.any(np.isfinite(scores))):
    raise ValueError("Invalid or all-zero particle weights")
weights = np.exp(scores - scores.max())
estimate = np.average(values, axis=0, weights=weights)
if not np.all(np.isfinite(estimate)):
    raise ValueError("Non-finite numeric posterior means")
print("BENCHMARK_RESULT=" + json.dumps(estimate.tolist()))
"""

def run_single_example(example_name, num_iterations, runtime_root, csv_writer=None, particles=1000):
    """Measure end-to-end execution including weighted numeric mean readout."""
    is_wheels = example_name in {"wheels", "wheels-compiled"}
    if is_wheels:
        # Wheels runs exactly five measured iterations with no warmups.
        num_iterations = 5
    num_particles = 1 if is_wheels else particles
    if num_iterations < 1 or num_particles < 1:
        raise ValueError("num_iterations and particles must be positive")
    runtime_root = Path(runtime_root).resolve()
    num_warmups = 0 if is_wheels else 5
    target_file = runtime_root / "examples" / "libsiren-examples" / f"{example_name}.py"
    
    if not target_file.exists():
        print(f"[-] Error: Could not find {example_name}.py at:\n    {target_file}\n")
        return False
        
    print(f"\n[*] BENCHMARKING: {example_name}.py "
          f"({num_iterations} measured iterations | {num_warmups} warmups | {num_particles} particles)",
          flush=True)
    
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(runtime_root), env.get("PYTHONPATH")) if part
    )
    command = [sys.executable, "-c", PYTHON_RUNNER, str(target_file), str(num_particles)]
    durations = []

    # Warmup runs use the same execution path but are excluded from statistics.
    for i in range(1 - num_warmups, num_iterations + 1):
        is_warmup = i <= 0
        iteration_label = f"warmup {i + num_warmups}" if is_warmup else f"iteration {i}"
        try:
            start_time = time.perf_counter()
            result = subprocess.run(command, capture_output=True, text=True, check=True,
                           cwd=str(runtime_root), env=env)
            end_time = time.perf_counter()
            if is_wheels:
                result_line = next(line for line in reversed(result.stdout.splitlines())
                                   if line.startswith('BENCHMARK_RESULT='))
                means, error = validate_wheels_mean(json.loads(result_line.split('=', 1)[1]))

            if is_warmup:
                print(f"    -> Warmup {i + num_warmups}/{num_warmups} complete")
            else:
                duration = end_time - start_time
                durations.append(duration)
                print(f"    -> Iteration {i}/{num_iterations}: {duration:.6f} seconds", flush=True)
            
        except subprocess.CalledProcessError as e:
            print(f"[-] Execution Error in {example_name}.py on {iteration_label}:")
            print(e.stderr or e.stdout)
            return False
        except (OSError, ValueError, StopIteration) as e:
            print(f"[-] Execution Error in {example_name}.py on {iteration_label}: {e}")
            return False

    # Calculate statistical metrics
    avg_time = statistics.mean(durations)
    std_dev = statistics.stdev(durations) if len(durations) > 1 else 0.0
    
    print(f"[+] {example_name} benchmark complete!")
    print(f"    Average Time: {avg_time:.6f}s | Std Dev: {std_dev:.6f}s")
    
    if csv_writer:
        csv_writer.writerow([example_name, num_particles, num_iterations, f"{avg_time:.6f}", f"{std_dev:.6f}"])
        
    return True

def main():
    runtime_root = Path(__file__).resolve().parent
    sys.path.insert(0, str(runtime_root))
    
    # --- Structural CLI Parser Integration ---
    parser = argparse.ArgumentParser(description="Siren Performance Pipeline Regression Suite")
    parser.add_argument('--benchmarks', '-b', type=str, help="Comma separated targets")
    parser.add_argument('--output', '-o', type=str, default='output', help="Output folder target")
    
    parser.add_argument('--iterations', '-n', type=int, default=100, help="Measured iterations after 5 warmups (wheels variants use 5 iterations, no warmups)")
    parser.add_argument("--particles", "-p", type=int, default=1000,
                        help="Particles per run (wheels variants always use 1)")
    
    args = parser.parse_args()
    print('[*] Timing includes weighted numeric posterior means; Wheels results are validated against an exact reference.', flush=True)
    if args.iterations < 1 or args.particles < 1:
        parser.error("--iterations and --particles must be positive")
    
    # Setup output environment variables directory
    out_dir = runtime_root / args.output
    out_dir.mkdir(parents=True, exist_ok=True)
    output_csv_path = out_dir / "benchmark_results.csv"
    
    # Unpack comma-separated benchmark options or map down to default array sweep
    target_benchmarks = [b.strip() for b in args.benchmarks.split(',')] if args.benchmarks else DEFAULT_EXAMPLES

    # print(f"[*] Harness launched. Processing {len(target_benchmarks)} target scenarios...")
    # print(f"[*] Telemetry report tracking target file path: {output_csv_path.name}")

    # Open CSV file context matrix to initialize headers and append logs
    with open(output_csv_path, mode="w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["Example Name", "Particles", "Iterations", "Average Execution Time (s)", "Standard Deviation (s)"])
        
        passed_count = 0
        for example in target_benchmarks:
            # Clean extension format if user typed 'example9.py' instead of 'example9'
            if example.endswith(".py"):
                example = example[:-3]
                
            if run_single_example(example, args.iterations, runtime_root, csv_writer, particles=args.particles):
                passed_count += 1
            csv_file.flush()
                
        print("\n" + "="*50)
        print(f"[+] Batch execution finished: {passed_count}/{len(target_benchmarks)} files logged successfully.")
        print(f"[+] Operational data saved to: {output_csv_path}")
        print("="*50)
    return 0 if passed_count == len(target_benchmarks) else 1

if __name__ == "__main__":
    sys.exit(main())

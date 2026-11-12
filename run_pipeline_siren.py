# run_pipeline_siren.py
import argparse
import sys
import os
import subprocess
from pathlib import Path
import time
import csv
import statistics

# Default list matching your exact .si files
DEFAULT_EXAMPLES = [
    "coin",
    "gaussian-gaussian",
    "kalman",
    "outlier",
    "gtree",
    "gaussian-gaussian-hybrid",
    "slam",
    "wheels",
    "coin-compiled",
    "gaussian-gaussian-compiled",
    "kalman-compiled",
    "outlier-compiled",
    "gtree-compiled",
    "gaussian-gaussian-hybrid-compiled",
    "slam-compiled",
    "wheels-compiled"
]


def run_single_example(example_name, num_iterations, particles, runtime_root, csv_writer=None):
    """Executes a siren (.si) benchmark file using subprocess and tracks execution statistics."""
    is_wheels = example_name == "wheels"
    if is_wheels:
        num_iterations = 5
        particles = 3
    if num_iterations < 1 or particles < 1:
        raise ValueError("num_iterations and particles must be positive")
    warmup_iterations = 0 if is_wheels else 5
    target_file = runtime_root / "examples" / "libsiren-examples"/f"{example_name}.si"

    if not target_file.exists():
        print(f"[-] Error: Could not find {example_name}.si at:\n    {target_file}\n")
        return False

    print(f"\n[*] ==================== BENCHMARKING: {example_name}.si ({num_iterations} Iterations | Particles: {particles}) ====================")

    durations = []

    # Explicitly configure environment variables so Python registers the workspace root
    env = os.environ.copy()
    env["PYTHONPATH"] = str(runtime_root)

    # Pass arguments separately so file paths are never interpolated into Python code.
    inline_code = "import sys; from siren.siren import main; sys.argv = sys.argv[1:]; main()"
    command = [sys.executable, "-c", inline_code, "siren", str(target_file),
               "--particles", str(particles)]

    # Warmups are excluded from the measured durations and CSV statistics.
    for i in range(1, warmup_iterations + num_iterations + 1):
        is_warmup = i <= warmup_iterations
        run_number = i if is_warmup else i - warmup_iterations
        run_label = f"warmup {run_number}" if is_warmup else f"iteration {run_number}"
        try:
            start_time = time.perf_counter()

            # Invoking via the explicit inline python execution method
            subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=True,
                cwd=str(runtime_root),  # Force the working directory to root
                env=env
            )

            end_time = time.perf_counter()
            duration = end_time - start_time
            if is_warmup:
                print(f"    -> Warmup {i}/{warmup_iterations}: {duration:.6f} seconds", flush=True)
            else:
                durations.append(duration)
                print(f"    -> Iteration {run_number}/{num_iterations}: {duration:.6f} seconds", flush=True)

        except subprocess.CalledProcessError as e:
            print(f"[-] Execution Error in {example_name}.si on {run_label}:")
            print(e.stderr or e.stdout)
            return False
        except Exception as e:
            print(f"[-] System Error on {run_label}: {e}")
            return False

    # Calculate statistical metrics
    avg_time = statistics.mean(durations)
    std_dev = statistics.stdev(durations) if len(durations) > 1 else 0.0

    print(f"[+] {example_name} benchmark complete!")
    print(f"    Average Time: {avg_time:.6f}s | Std Dev: {std_dev:.6f}s")

    if csv_writer:
        csv_writer.writerow([example_name, particles, num_iterations, f"{avg_time:.6f}", f"{std_dev:.6f}"])

    return True


def main():
    runtime_root = Path(__file__).resolve().parent
    sys.path.insert(0, str(runtime_root))

    # --- Structural CLI Parser Integration ---
    parser = argparse.ArgumentParser(description="Siren Performance Pipeline Regression Suite")
    parser.add_argument('--benchmarks', '-b', type=str, help="Comma separated targets (e.g., example1,kalman)")
    parser.add_argument('--output', '-o', type=str, default='output', help="Output folder target")
    parser.add_argument('--iterations', '-n', type=int, default=100,
                        help="Measured iterations per example after 5 warmups (wheels uses 5 iterations, no warmups)")

    # Added explicit particle count configuration parameter (defaulting to 1000)
    parser.add_argument('--particles', '-p', type=int, default=1000,
                        help="Particles per example (wheels always uses 3)")

    args = parser.parse_args()
    if args.iterations < 1 or args.particles < 1:
        parser.error('--iterations and --particles must be positive')

    # Setup output environment variables directory
    out_dir = runtime_root / args.output
    out_dir.mkdir(parents=True, exist_ok=True)
    output_csv_path = out_dir / "benchmark_results.csv"

    # Unpack comma-separated benchmark options or map down to default array sweep
    target_benchmarks = [b.strip() for b in args.benchmarks.split(',')] if args.benchmarks else DEFAULT_EXAMPLES

    # print(f"[*] Harness launched. Processing {len(target_benchmarks)} target scenarios...")
    # print(f"[*] Testing with {args.particles} particles.")
    # print(f"[*] Telemetry report tracking target file path: {output_csv_path.name}")

    # Open CSV file context matrix to initialize headers and append logs
    with open(output_csv_path, mode="w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(
            ["Example Name", "Particles", "Iterations", "Average Execution Time (s)", "Standard Deviation (s)"])

        passed_count = 0
        for example in target_benchmarks:
            # Clean extension formats safely
            if example.endswith(".py"):
                example = example[:-3]
            elif example.endswith(".py"):
                example = example[:-3]

            if run_single_example(example, args.iterations, args.particles, runtime_root, csv_writer):
                passed_count += 1

        # print("\n" + "="*50)
        # print(f"[+] Batch execution finished: {passed_count}/{len(target_benchmarks)} files logged successfully.")
        # print(f"[+] Operational data saved to: {output_csv_path}")
        # print("="*50)


if __name__ == "__main__":
    main()

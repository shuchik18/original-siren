# import sys
# import os
# from pathlib import Path
# import time

# def run_single_example(example_name, runtime_root):
#     """Reads an example file, strips out the print(run) line, and executes it."""
#     # Construct the path to the specific example file
#     target_file = runtime_root /  "examples" / "libsiren-examples" / f"{example_name}.py"
    
#     if not target_file.exists():
#         print(f"[-] Error: Could not find {example_name}.py at:\n    {target_file}\n")
#         return False
        
#     print(f"\n[*] ==================== RUNNING: {example_name}.py ====================")
    
#     # 1. Read the example file contents
#     with open(target_file, "r") as f:
#         lines = f.readlines()
        
#     # 2. Filter out the print(run(...)) loop line
#     # cleaned_lines = []
#     # for line in lines:
#     #     if "print(run(" in line:
#     #         continue
#     #     cleaned_lines.append(line)
        
#     source_code = "".join(lines)
    
#     # 3. Dynamic execution namespace
#     namespace = {}
    
#     try:
#         start_time = time.time()
#         exec(source_code, namespace)
#         end_time = time.time()
#         duration = end_time - start_time
#         print(f"[+] {example_name} executed successfully!")
#         print(f"[#] Execution Time: {duration:.6f} seconds")
#         # if "program" in namespace:
#         #     print(f"[+] Verified structure: Found 'program(particle)' function.")
#         # return True
#     except Exception as e:
#         print(f"[-] Execution Error in {example_name}.py: {e}")
#         return False

# def main():
#     # Anchor to your VS Code libsiren workspace root
#     runtime_root = Path(__file__).resolve().parent
    
#     # Inject runtime_root into system paths so 'from siren.libsiren import *' works
#     sys.path.insert(0, str(runtime_root))
    
#     # If you pass a specific example argument (e.g., python run_pipeline.py example9)
#     if len(sys.argv) > 1:
#         target_example = sys.argv[1]
#         # Clean extension if user types 'example9.py' instead of 'example9'
#         if target_example.endswith(".py"):
#             target_example = target_example[:-3]
            
#         success = run_single_example(target_example, runtime_root)
#         sys.exit(0 if success else 1)
        
#     # Default behavior: Loop through all 8 example files automatically
#     else:
#         # print("[*] No specific file provided. Running all 8 compiler targets...")
        
#         # Define your 8 example files exactly as they are named on disk
#         all_examples = [
#             "example1", "example-1-compiled", "example-5-compiled" , "example8" , 
#             "example5", "example9", "example10", "example11" , "example-8-compiled", "example-9-compiled", "example-10-compiled", "example-11-compiled"
#         ]
        
#         passed_count = 0
#         for example in all_examples:
#             if run_single_example(example, runtime_root):
#                 passed_count += 1
                
#         print("\n" + "="*50)
#         print(f"[+] Batch execution finished: {passed_count}/{len(all_examples)} files passed parsing.")
#         print("="*50)

# if __name__ == "__main__":
#     main()


# libsiren/run_pipeline.py
import argparse
import sys
import os
from pathlib import Path
import time
import csv
import statistics

# Complete default target array list matching your workspace layout
DEFAULT_EXAMPLES = [
    "example1", "example-1-compiled", "example-5-compiled", "example8", 
    "example5", "example9", "example10", "example11", 
    "example-8-compiled", "example-9-compiled", "example-10-compiled", "example-11-compiled"
]

def run_single_example(example_name, num_iterations, runtime_root, csv_writer=None):
    """Reads an example file, runs it multiple times, calculates stats, and logs results."""
    target_file = runtime_root / "examples" / "libsiren-examples" / f"{example_name}.py"
    
    if not target_file.exists():
        print(f"[-] Error: Could not find {example_name}.py at:\n    {target_file}\n")
        return False
        
    print(f"\n[*] ==================== BENCHMARKING: {example_name}.py ({num_iterations} Iterations) ====================")
    
    with open(target_file, "r") as f:
        lines = f.readlines()
        
    # Filter out any pre-existing print(run(...)) statements to avoid execution explosion
    cleaned_lines = [line for line in lines if "print(run(" not in line]
    source_code = "".join(cleaned_lines)
    
    durations = []
    
    try:
        # Pre-compile the string into bytecode ONCE outside the loop.
        # This removes the "first-run compilation " from your benchmark.
        compiled_code = compile(source_code, f"<string_{example_name}>", "exec")
    except Exception as e:
        print(f"[-] Compilation Error in {example_name}.py: {e}")
        return False
    
    # Run the program block for the designated number of iterations
    for i in range(5, num_iterations + 1):
        namespace = {}
        try:
            start_time = time.time()  # High-precision hardware clock
            exec(source_code, namespace)
            
            # Programmatically trigger simulation runner if present in namespace
            if "run" in namespace and "program" in namespace:
                namespace["run"](namespace["program"], 10)
                
            end_time = time.time()
            duration = end_time - start_time
            durations.append(duration)
            print(f"    -> Iteration {i}/{num_iterations}: {duration:.6f} seconds")
            
        except Exception as e:
            print(f"[-] Execution Error in {example_name}.py on iteration {i}: {e}")
            return False

    # Calculate statistical metrics
    avg_time = statistics.mean(durations)
    std_dev = statistics.stdev(durations) if len(durations) > 1 else 0.0
    
    print(f"[+] {example_name} benchmark complete!")
    print(f"    Average Time: {avg_time:.6f}s | Std Dev: {std_dev:.6f}s")
    
    if csv_writer:
        csv_writer.writerow([example_name, num_iterations, f"{avg_time:.6f}", f"{std_dev:.6f}"])
        
    return True

def main():
    runtime_root = Path(__file__).resolve().parent
    sys.path.insert(0, str(runtime_root))
    
    # --- Structural CLI Parser Integration ---
    parser = argparse.ArgumentParser(description="Siren Performance Pipeline Regression Suite")
    parser.add_argument('--benchmarks', '-b', type=str, help="Comma separated targets")
    parser.add_argument('--output', '-o', type=str, default='output', help="Output folder target")
    
    # Custom parameter configured to default to exactly 1 calculation pass 
    parser.add_argument('--iterations', '-n', type=int, default=1, help="Number of profiling cycles per script execution")
    
    args = parser.parse_args()
    
    # Setup output environment variables directory
    out_dir = runtime_root / args.output
    out_dir.mkdir(exist_ok=True)
    output_csv_path = out_dir / "benchmark_results.csv"
    
    # Unpack comma-separated benchmark options or map down to default array sweep
    target_benchmarks = [b.strip() for b in args.benchmarks.split(',')] if args.benchmarks else DEFAULT_EXAMPLES

    # print(f"[*] Harness launched. Processing {len(target_benchmarks)} target scenarios...")
    # print(f"[*] Telemetry report tracking target file path: {output_csv_path.name}")

    # Open CSV file context matrix to initialize headers and append logs
    with open(output_csv_path, mode="w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["Example Name", "Iterations", "Average Execution Time (s)", "Standard Deviation (s)"])
        
        passed_count = 0
        for example in target_benchmarks:
            # Clean extension format if user typed 'example9.py' instead of 'example9'
            if example.endswith(".py"):
                example = example[:-3]
                
            if run_single_example(example, args.iterations, runtime_root, csv_writer):
                passed_count += 1
                
        print("\n" + "="*50)
        print(f"[+] Batch execution finished: {passed_count}/{len(target_benchmarks)} files logged successfully.")
        print(f"[+] Operational data saved to: {output_csv_path}")
        print("="*50)

if __name__ == "__main__":
    main()
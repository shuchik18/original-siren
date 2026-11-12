# # import argparse
# # import os
# # import time


# # from . import parser, evaluate, analyze
# # from .compiler import SSIState
# # from .analysis import AbsSSIState, AbsDSState, AbsBPState
# # from .inference_plan import runtime_inference_plan
# # import sys


# # from llcsiren.siren.compiler import compile_program, compile_source

# # # Maps the command line arguments to the symbolic state classes
# # method_states = {
# #     "ssi": (SSIState, AbsSSIState),

# # }

# # def main():
# #     sys.setrecursionlimit(5000)
# #     p = argparse.ArgumentParser()
# #     p.add_argument("filename", type=str)
# #     # p.add_argument("--verbose", "-v", action="store_true")
# #     # p.add_argument("--particles", "-p", type=int, default=100, help="Number of particles to use during inference")
# #     # p.add_argument("--analyze", "-a", action="store_true", help="Apply the inference plan satisfiability analysis during compilation")
# #     # p.add_argument("--analyze-only", "-ao", action="store_true", help="Only apply the inference plan satisfiability analysis, does not run the program")
# #     p.add_argument(
# #         "--method",
# #         "-m",
# #         type=str,
# #         default="ssi",
# #         choices=["ssi", "ds", "bp"],
# #     )
# #     # p.add_argument("--seed", "-s", type=int, default=None)
# #     # p.add_argument("--max-rvs", type=int, default=4, help="Maximum number of random variables to track in abstract expression types")
# #     args = p.parse_args()

# #     # with open(args.filename, "r") as f:
# #     #     program = parser.parse_program(f.read())
# #     #     print(program)
# #     with open(args.filename, "r") as f:
# #         program = parser.parse_program(f.read())
# #         # print("===== Original Parsed Program =====")
# #         # print(program)

# #     # Stop early if only performing analysis
# #     # if args.analyze_only:
# #     #     return
# #     # Get the symbolic state classes for the selected inference method
# #     (inference_method, analysis_method) = method_states[args.method]

# #     print("===== Inference Algorithm =====")
# #     match args.method:
# #         case "ssi":
# #             print("SSI")
# #         case "ds":
# #             print("DS")
# #         case "bp":
# #             print("BP")
# #         case _:
# #             raise ValueError("Invalid method")

# #     # Run the inference plan satisfiability analysis
# #     # if args.analyze or args.analyze_only:
# #     #     print("===== Inferred Inference Plan =====")
# #     #     # t1 = time.time()
# #     #     inferred_plan = analyze.analyze(
# #     #         program, analysis_method, args.max_rvs
# #     #     )
# #     #     # t2 = time.time()
# #     #     print(inferred_plan)
# #     #     print("-----------------------------------------------")
# #         # key_value=[]
# #         # dict_value=[]
# #         # temp_dict={}
# #         # key_value= list(inferred_plan.state.state.keys())
# #         # dict_value=list(inferred_plan.state.state.values())



# #         # val={}
# #         # dist_val=[]
# #         # val = inferred_plan.state.state
# #         # # print("dsaafvw",val)
# #         #
# #         # for key, values in val.items():
# #         #     if isinstance(values, dict):
# #         #         if 'distribution' in values:
# #         #             val[key] = values['distribution']
# #         #         else:
# #         #             val.update(values)

# #         # print("final value with the key of RamdomVar and the values of that dict with distribution value-------------",val)
# #         # for key, values in val.items():
# #         #     if isinstance(values,str):
# #                 # print("afdgwertbvgrwetb, ",inferred_plan.state.state.val_new_res(values))

# #         # for i in len(val[value]):
# #         #
# #         #     temp_dict = inferred_plan.state.state.val_new_res(val[value[i]])
# #         # print("check value", temp_dict)
# #         # for key, value in val.items():
# #         #     if isinstance(value, dict) and 'distribution' in value:
# #         #         val[key][key] = value['distribution']
# #         #         temp_dict = val[key][key]
# #         #
# #         # print("distribution value: ",  temp_dict)


# #         # for i in range(len(dict_value)):
# #         #     temp_dict[i]=dict_value[i]
# #         #
# #         # print("keys: ", key_value)
# #         # print(" ")
# #         # print(" ")
# #         # print("values: ",temp_dict)
# #         # annotations = []
# #         # pvs = []
# #         # distributions = []
# #         #
# #         # for key, value in temp_dict.items():
# #         #     if 'annotation' in value:
# #         #         annotations.append(value['annotation'])
# #         #     if 'pv' in value:
# #         #         pvs.append(value['pv'])
# #         #     if 'distribution' in value:
# #         #         distributions.append(value['distribution'])
# #         #
# #         # print("Distributions:", distributions)
# #         # print("inferred_plan.state.state: ", inferred_plan.state.state.keys())
# #         # print("==========")
# #         # print()
# #         # print("===== Analysis Time =====")
# #         # print(f"{t2 - t1}")

# #     # If the user only wants to analyze the program, we can stop here
# #     # if not args.analyze_only:
# #     #     file_dir = os.path.dirname(os.path.realpath(args.filename))
# #     #     t1 = time.time()
# #     #     new_expr = evaluate.evaluate(
# #     #         program,
# #     #         args.particles,
# #     #         inference_method,
# #     #         file_dir,
# #     #         args.seed,
# #     #     )
# #     #     t2 = time.time()

# #     #     print("===== Evaluation Time =====")
# #     #     print(f"{t2 - t1}")

# #     #     print("===== Result =====")
# #     #     print(new_expr)
    
# #     file_dir = os.path.dirname(os.path.realpath(args.filename))
# #     t1 = time.time()
# #     new_expr = evaluate.evaluate(
# #         program,
# #         # args.particles,
# #         # inference_method,
# #         file_dir,
# #         # args.seed,
# #     )
# #     t2 = time.time()

# #     # print("\n===== Evaluation Time =====")
# #     # print(f"{t2 - t1}")

# #     print("===== Result (new_expr) =====")
# #     print(new_expr)

# #     # Step 2: Pass `new_expr` into the llcsiren compiler to generate Python code
# #     print("\n===== Generating Python Code from new_expr =====")
# #     compiled_python = None
# #     try:
# #         if isinstance(new_expr, str):
# #             compiled_python = compile_source(new_expr)
# #         else:
# #             compiled_python = compile_program(new_expr)

# #     except CompilerError as err:
# #         print(f"Compiler Error: {err}", file=sys.stderr)
# #     except Exception:
# #         try:
# #             # Fallback if new_expr needs string conversion before compilation
# #             compiled_python = compile_source(str(new_expr))
# #         except Exception as fallback_err:
# #             print(f"Failed to compile new_expr to Python: {fallback_err}", file=sys.stderr)
            
# #     if compiled_python:
# #         output_file_path = os.path.join(file_dir, "python.py")
        
# #         # Prepend runtime imports if not already included in the generated string
# #         runtime_header = "from siren.libsiren import *\n\n"
# #         if "from siren.libsiren import" not in compiled_python:
# #             compiled_python = runtime_header + compiled_python

# #         with open(output_file_path, "w") as out_file:
# #             out_file.write(compiled_python)
        
# #         print("\n===== Saved Generated Code =====")
# #         print(f"[+] Output successfully saved to: {output_file_path}")

# #     # Step 3: Write the output to python.py in the same folder as source.si
# #     if compiled_python:
# #         output_file_path = os.path.join(file_dir, "target.py")
# #         with open(output_file_path, "w") as out_file:
# #             out_file.write(compiled_python)
        
# #         print("\n===== Saved Generated Code =====")
# #         print(f"[+] Output successfully saved to: {output_file_path}")

# #         # Get the runtime inference plan by inspecting the particles
# #         # plan = runtime_inference_plan(new_expr)

# #         # if args.verbose:
# #             # Only for debugging to reduce the expressions
# #             # particles.simplify()
# #             # print("===== Mixture =====")
# #             # print(particles.mixture())
# #             # print("===== Particles =====")
# #             # print(particles)

# #         # file_dir = os.path.dirname(os.path.realpath(args.filename))
# #         # t1 = time.time()
# #         # new_expr1 = evaluate.evaluate(
# #         #     program,
# #         #     args.particles,
# #         #     inference_method,
# #         #     file_dir,
# #         #     args.seed,
# #         # )
# #         #
# #         # print("===== Runtime Inference Plan =====")
# #         # print(new_expr1)

# # if __name__ == "__main__":
# #     main()

# import argparse
# import os
# import sys
# import time

# from . import evaluate, parser
# from .analysis import AbsSSIState
# from .compiler import SSIState

# # Import compilation functions from llcsiren
# from llcsiren.siren.compiler import compile_program, compile_source, CompilerError

# method_states = {
#     "ssi": (SSIState, AbsSSIState),
# }

# def main():
#     sys.setrecursionlimit(5000)
#     p = argparse.ArgumentParser()
#     p.add_argument("filename", type=str)
#     p.add_argument(
#         "--method",
#         "-m",
#         type=str,
#         default="ssi",
#         choices=["ssi", "ds", "bp"],
#     )
#     p.add_argument("--particles", "-p", type=int, default=1000, help="Number of particles for libsiren.run()")
#     args = p.parse_args()

#     with open(args.filename, "r") as f:
#         program = parser.parse_program(f.read())

#     file_dir = os.path.dirname(os.path.realpath(args.filename))

#     # Step 1: Run Siren evaluation to obtain new_expr
#     print("===== Evaluating Siren Program =====")
#     t1 = time.time()
#     new_expr = evaluate.evaluate(program, file_dir)
#     t2 = time.time()

#     print(f"Evaluation Time: {t2 - t1:.4f}s")
#     print("===== Result (new_expr) =====")
#     print(new_expr)

#     # Step 2: Compile new_expr to Python using llcsiren
#     # (compiler.py will inject 'from siren.libsiren import *' and 'run()' automatically)
#     print("\n===== Compiling new_expr via llcsiren =====")
#     try:
#         if isinstance(new_expr, str):
#             compiled_python = compile_source(new_expr, run_particles=args.particles)
#         else:
#             compiled_python = compile_program(new_expr, run_particles=args.particles)

#         # Step 3: Save to python.py in the source directory
#         output_file_path = os.path.join(file_dir, "python.py")
#         with open(output_file_path, "w") as out_file:
#             out_file.write(compiled_python)

#         print("\n===== Output Successfully Generated =====")
#         print(f"[+] Python code saved to: {output_file_path}")

#     except CompilerError as err:
#         print(f"Compiler Error: {err}", file=sys.stderr)
#     except Exception as e:
#         print(f"Failed to generate code: {e}", file=sys.stderr)

# if __name__ == "__main__":
#     main()

import argparse
import os
import sys
import time

from . import evaluate, parser
from .analysis import AbsSSIState
from .compiler import SSIState

# Import compilation functions from llcsiren
from llcsiren.siren.compiler import compile_program, compile_source, CompilerError

method_states = {
    "ssi": (SSIState, AbsSSIState),
}

def compile_and_save(data, output_filename, file_dir, particles):
    """Compiles an AST or source string to Python code and writes it to disk."""
    try:
        if isinstance(data, str):
            compiled_python = compile_source(data, run_particles=particles)
        else:
            compiled_python = compile_program(data, run_particles=particles)

        # Prepend runtime imports for libsiren if missing
        runtime_header = "from siren.libsiren import *\n\n"
        if not compiled_python.startswith("from siren.libsiren import"):
            compiled_python = runtime_header + compiled_python

        output_file_path = os.path.join(file_dir, output_filename)
        with open(output_file_path, "w") as out_file:
            out_file.write(compiled_python)

        print(f"[+] Output successfully saved to: {output_file_path}")
        return compiled_python

    except CompilerError as err:
        print(f"Compiler Error for {output_filename}: {err}", file=sys.stderr)
    except Exception as e:
        print(f"Failed to generate code for {output_filename}: {e}", file=sys.stderr)

def main():
    sys.setrecursionlimit(5000)
    p = argparse.ArgumentParser()
    p.add_argument("filename", type=str)
    p.add_argument(
        "--method",
        "-m",
        type=str,
        default="ssi",
        choices=["ssi", "ds", "bp"],
    )
    p.add_argument("--particles", "-p", type=int, default=1000, help="Number of particles for libsiren.run()")
    args = p.parse_args()

    # Read raw source text and parse the program AST
    with open(args.filename, "r") as f:
        raw_source = f.read()
        program = parser.parse_program(raw_source)

    file_dir = os.path.dirname(os.path.realpath(args.filename))

    # Step 1: Run Siren evaluation to obtain new_expr
    print("===== Evaluating Siren Program =====")
    t1 = time.time()
    new_expr = evaluate.evaluate(program, file_dir)
    t2 = time.time()

    print(f"Evaluation Time: {t2 - t1:.4f}s")
    print("===== Result (new_expr) =====")
    print(new_expr)

    # Step 2: Compile evaluated program (new_expr) to python.py
    print("\n===== Compiling Evaluated Program (new_expr) =====")
    compile_and_save(new_expr, "python.py", file_dir, args.particles)

    # Step 3: Compile original program (program / AST) to original_python.py
    print("\n===== Compiling Original Program (AST) =====")
    compile_and_save(program, "original_python.py", file_dir, args.particles)

if __name__ == "__main__":
    main()
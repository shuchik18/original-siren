#!/usr/bin/env python3
"""Read-only adapter for the original Siren artifact; emits one JSON row/run.

Parsing/imports and optional input repair are outside the inference timer.
Siren's SMC.infer includes symbolic mean evaluation, as does its own CLI timer.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent


def flatten(value):
    if isinstance(value, (list, tuple)):
        return [item for part in value for item in flatten(part)]
    return [float(value)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--method", choices=["ssi", "ds", "bp"], default="ssi")
    p.add_argument("--particles", type=int, required=True)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--fix-wheels-input", action="store_true")
    args = p.parse_args()
    repo = args.repo.expanduser().resolve()
    sys.path.insert(0, str(repo))
    sys.setrecursionlimit(10000)
    from siren import parser
    from siren.evaluate import SMC
    from siren.inference import SSIState, DSState, BPState
    if not Path(parser.__file__).resolve().is_relative_to(repo):
        raise RuntimeError("Loaded a different Siren installation")
    state = {"ssi": SSIState, "ds": DSState, "bp": BPState}[args.method]
    source_path = HERE / (args.model + ".si")
    source = source_path.read_text()
    repaired = args.model == "wheels" and args.fix_wheels_input
    if repaired:
        old = "fold(step, data,"
        new = "fold(step, List.map(preprocess_data, data),"
        if source.count(old) != 1:
            raise ValueError("Expected exactly one wheels fold to repair")
        source = source.replace(old, new)
        copy = HERE / "_build/siren-input/wheels.si"
        copy.parent.mkdir(parents=True, exist_ok=True)
        copy.write_text(source)
        source_path = copy
    program = parser.parse_program(source)
    # These programs have no explicit resample. The artifact's final scores
    # therefore retain the complete importance log weights, permitting log Z.
    if "resample" in source:
        raise ValueError("This adapter's log-evidence calculation requires no resample statements")
    for _ in range(args.repeats):
        try:
            start = time.perf_counter()
            result, particles = SMC().infer(program, state, str(HERE), seed=None,
                                            n_particles=args.particles)
            elapsed = time.perf_counter() - start
            scores = [float(p.score) for p in particles]
            maximum = max(scores)
            weights = [math.exp(s - maximum) for s in scores]
            total = sum(weights)
            mean = flatten(result.v)
            log_z = maximum + math.log(total) - math.log(len(scores))
            if not all(map(math.isfinite, mean + [log_z])):
                raise ValueError("Non-finite Siren result")
            record = {"status": "ok", "inference_seconds": elapsed,
                      "summary_seconds": None, "mean": mean, "variance": None,
                      "log_z": log_z, "ess": total**2 / sum(w*w for w in weights)}
        except Exception as exc:
            record = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
        record.update(input_repair="missing wheels preprocess_data map" if repaired else None,
                      siren_source_sha256=hashlib.sha256(source.encode()).hexdigest())
        print(json.dumps(record, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()

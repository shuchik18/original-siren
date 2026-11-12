#!/usr/bin/env python3
"""Reproduce baseline compiler failures without modifying the compiler.

The alternative Wheels file is a diagnostic translation using tuple projections
and affine addition/multiplication. It preserves the model and its loops.
"""

import json
from pathlib import Path
import subprocess
import time

from benchmark import HERE, REPO, compiler_environment, digest, revision


def main():
    directory = HERE / "diagnostics/generated"
    directory.mkdir(parents=True, exist_ok=True)
    source = HERE / "diagnostics/wheels-projections.dppl"
    compiler = REPO / "build/cppl"
    command = [str(compiler), str(source), "--implicit-infer", "-m", "is-lw", "--cps", "none",
               "--static-delay", "--extract-simplification", "inline", "--print-model",
               "--output", str(directory / "wheels")]
    start = time.perf_counter()
    p = subprocess.run(command, cwd=directory, env=compiler_environment(), text=True,
                       capture_output=True, timeout=300)
    elapsed = time.perf_counter() - start
    (directory / "compile.stdout").write_text(p.stdout)
    (directory / "compile.stderr").write_text(p.stderr)
    error_text = (p.stderr + p.stdout).rsplit("standard error:", 1)[-1].strip()
    evidence = {"case": "wheels-affine-projections", "command": command,
                "compiler_sha256": digest(compiler), "compiler_repo_revision": revision(REPO),
                "source_sha256": digest(source), "returncode": p.returncode,
                "status": "compile_error" if p.returncode else "compiled",
                "seconds": elapsed, "error_excerpt": error_text[-1500:]}
    (HERE / "diagnostics/wheels-compiler-result.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(evidence, indent=2))
    # A failure is the expected observation, but preserve success if upstream changes.


if __name__ == "__main__":
    main()

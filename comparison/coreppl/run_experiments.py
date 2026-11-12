#!/usr/bin/env python3
"""Rerun the agreed two-configuration comparison, without sampling controls."""

import argparse
from datetime import datetime
import json
import subprocess
import sys

from benchmark import HERE, local_path, write_csv
from reference import MODELS


def comparison_row(row):
    """Exclude plain and explicitly sampled CorePPL controls, including old data.

    A static run can still contain mandatory draws. Its requested posterior
    readout distinguishes it from the retired explicit sample-output controls.
    """
    return row["engine"] == "siren" or (
        row["engine"] == "coreppl" and row["mode"] == "static" and
        (row.get("requested_readout") == "posterior" or
         (not row.get("requested_readout") and row.get("readout") == "posterior")))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/symbolic-" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    parser.add_argument("--timeout", type=int, default=300, help="seconds per build or count batch")
    args = parser.parse_args()
    output = local_path(args.output)
    if output.exists() and any(output.iterdir()):
        parser.error(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    common = [sys.executable, "-B", str(HERE / "benchmark.py"), "--timeout", str(args.timeout)]
    schedule = [
        ("coreppl", ["--engines", "coreppl", "--modes", "static", "--readout", "posterior",
                     "--particles", "1", "10", "100", "1000", "10000", "--repeats", "5"]),
        ("siren", ["--engines", "siren", "--models", *[n for n in MODELS if n != "wheels"],
                   "--particles", "1", "10", "100", "--repeats", "3"]),
        ("siren-wheels", ["--engines", "siren", "--models", "wheels", "--particles", "1", "2",
                          "--repeats", "1", "--warmups", "0", "--fix-wheels-input"]),
        # Preserve the supplied Wheels input failure separately from accuracy.
        ("siren-wheels-original", ["--engines", "siren", "--models", "wheels", "--particles", "1",
                                   "--repeats", "1", "--warmups", "0"]),
    ]
    manifest = {"configurations": ["Siren SSI", "Miking static DS with best available posterior readout"],
                "batches": []}
    import csv
    rows, summaries = [], []
    for name, options in schedule:
        directory = output / name
        command = common + options + ["--output", str(directory)]
        completed = subprocess.run(command, cwd=HERE)
        manifest["batches"].append({"name": name, "command": command, "returncode": completed.returncode,
                                    "diagnostic": name == "siren-wheels-original"})
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        if (directory / "runs.jsonl").exists():
            rows.extend({**row, "source_results": str(directory.relative_to(HERE))}
                        for line in (directory / "runs.jsonl").read_text().splitlines()
                        if comparison_row(row := json.loads(line)))
        if (directory / "summary.csv").exists():
            with (directory / "summary.csv").open() as f:
                summaries.extend({**row, "source_results": str(directory.relative_to(HERE))}
                                 for row in csv.DictReader(f) if comparison_row(row))
        write_csv(output / "runs.csv", rows)
        write_csv(output / "summary.csv", summaries)
        (output / "runs.jsonl").write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in rows))
    failures = [row for row in rows if row["status"] != "ok"]
    (output / "failures.json").write_text(json.dumps(failures, indent=2) + "\n")
    print(f"Comparison: {output}")
    # The original Wheels input failure is an explicitly scheduled diagnostic.
    return int(any(b["returncode"] for b in manifest["batches"] if not b["diagnostic"]))


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    sys.exit(main())

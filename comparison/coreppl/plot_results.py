#!/usr/bin/env python3
"""Plot accuracy versus inference time from benchmark summary.csv files.

Requires matplotlib; the configured Siren Python environment supplies it.
"""

import argparse
import csv
import os
from pathlib import Path
import math

from reference import MODELS
from run_experiments import comparison_row

HERE = Path(__file__).resolve().parent
# Keep matplotlib's font/config cache within the comparison directory too.
os.environ["MPLCONFIGDIR"] = str(HERE / "_build/matplotlib")
os.environ["XDG_CACHE_HOME"] = str(HERE / "_build/cache")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", type=Path, help="result directories or summary CSV files")
    parser.add_argument("--output", type=Path, default=HERE / "results/accuracy-performance.png")
    parser.add_argument("--metric", choices=["mean", "log-evidence", "variance"], default="mean",
                        help="hybrid mean is zero by symmetry; log-evidence also tests its branch integration")
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(HERE):
        parser.error("Output must be inside the comparison directory")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    metric, ylabel = {
        "mean": ("mean_standardized_rmse_rms", "Standardized mean RMSE"),
        "log-evidence": ("log_z_abs_error_rms", "Log-evidence RMSE"),
        "variance": ("variance_rmse_rms", "Variance RMSE"),
    }[args.metric]

    rows = []
    for path in args.results:
        path = path / "summary.csv" if path.is_dir() else path
        with path.open() as f:
            rows.extend(row for row in csv.DictReader(f) if comparison_row(row))
    nrows = math.ceil((len(MODELS) + 1) / 3)
    figure, axes = plt.subplots(nrows, 3, figsize=(13, 3.8 * nrows))
    for ax, model in zip(axes.flat, MODELS):
        groups = {}
        for row in rows:
            if row["model"] != model or int(row["successful_runs"]) == 0 or not row.get(metric):
                continue
            label = ("Miking static DS" if row["engine"] == "coreppl" else "Siren " + row["method"])
            if row["engine"] == "coreppl" and row["method"] != "is-lw":
                label += " " + row["method"]
            groups.setdefault(label, []).append(row)
        for group_index, (label, group) in enumerate(groups.items()):
            group.sort(key=lambda r: int(r["particles"]))
            group = [r for r in group if float(r["inference_seconds_median"]) > 0]
            x = [1000 * float(r["inference_seconds_median"]) for r in group]
            y = [max(1e-12, float(r[metric])) for r in group]
            ax.plot(x, y, "o-", label=label, markersize=4, linewidth=1)
            for point_index, (xi, yi, row) in enumerate(zip(x, y, group)):
                if point_index in (0, len(group) - 1):
                    left = point_index == len(group) - 1 or group_index % 2 == 1
                    offset = (-5 if left else 5, 6 if point_index == len(group) - 1 or group_index % 2 == 0 else -10)
                    ax.annotate(row["particles"], (xi, yi), xytext=offset, textcoords="offset points",
                                ha="right" if left else "left", fontsize=6)
        ax.set(xscale="log", yscale="log", title=model + (" (static DS biased)" if model == "gtree" else ""),
               xlabel="Median inference time (ms)", ylabel=ylabel)
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=7, loc="best")
    for ax in list(axes.flat)[len(MODELS):]:
        ax.axis("off")
    axes.flat[len(MODELS)].text(0, 0.95,
        "Endpoint labels denote particle counts.\n\n"
        "Errors below 1e-12 are displayed at 1e-12.\n"
        "Siren SSI uses symbolic mixture means;\n"
        "Miking queries retained posterior moments.\n"
        "Wheels/outlier/slam/hybrid retain\n"
        "point particles.\n"
        "No plain or explicit sample-readout controls.\n\n"
        "CorePPL timing excludes final aggregation;\n"
        "Siren timing includes symbolic mean evaluation.\n\n"
        "The original compiler is unchanged.\n"
        "gtree's static result has a known bias.\n"
        "Wheels Siren input uses the documented\n"
        "list-to-pair repair.\n\n"
        "See README.md and raw failure records.",
        va="top", fontsize=10, linespacing=1.6)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    figure.savefig(output.with_suffix(".pdf"))
    print(output)


if __name__ == "__main__":
    main()

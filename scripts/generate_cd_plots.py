#!/usr/bin/env python
"""Generate Critical Difference plots.

Runs summarize_benchmark.py --cd on a results directory, then copies
the CD plot PNGs into an output directory.

Usage:
    python scripts/generate_cd_plots.py results/my_run
    python scripts/generate_cd_plots.py results/my_run --output-dir figures/

Requires: autorank, matplotlib, pandas, joblib
"""

import argparse
import subprocess
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description="Generate Critical Difference plots")
    parser.add_argument("results_dir", type=Path,
                        help="Directory containing .joblib result files")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "figures",
                        help="Output directory for figures (default: figures/)")
    parser.add_argument("--exclude-models", nargs='+', default=[], metavar='MODEL',
                        help="Models to exclude (passed through to summarize_benchmark)")
    args = parser.parse_args()

    # 1. Run summarize_benchmark.py --cd
    print("Running summarize_benchmark.py --cd ...", flush=True)
    cmd = ["python", str(ROOT / "scripts" / "summarize_benchmark.py"),
           str(args.results_dir), "--cd"]
    for m in args.exclude_models:
        cmd.extend(["--exclude-models", m])
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"Error: summarize_benchmark.py exited with code {result.returncode}")

    # 2. Copy CD plots to output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary_dir = args.results_dir / "benchmark_summary"
    cd_plots = list(summary_dir.glob("cd_plot_*.png"))

    if not cd_plots:
        print(f"WARNING: No CD plots found in {summary_dir}", flush=True)
        return

    for src in cd_plots:
        dst = args.output_dir / src.name
        shutil.copy2(src, dst)
        print(f"  Copied {src.name} -> {dst}", flush=True)

    print("Done.", flush=True)


if __name__ == "__main__":
    main()

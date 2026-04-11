#!/usr/bin/env python3
"""Plot GPU vs CPU solver crossover from benchmark CSV.

Usage:
    # 1. Run your pipeline with logging enabled:
    VIPE_SOLVER_BENCH_CSV=solver_bench.csv python run_slam.py ...

    # 2. Plot the results:
    python plot_solver_crossover.py solver_bench.csv
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main(csv_path: str) -> None:
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} samples from {csv_path}")
    print(f"  dim range: [{df['dim'].min()}, {df['dim'].max()}]")
    print(f"  speedup range: [{df['speedup'].min():.2f}, {df['speedup'].max():.2f}]")

    # --- Aggregate by dim ---
    agg = df.groupby("dim").agg(
        gpu_ms_mean=("gpu_ms", "mean"),
        gpu_ms_std=("gpu_ms", "std"),
        cpu_ms_mean=("cpu_ms", "mean"),
        cpu_ms_std=("cpu_ms", "std"),
        speedup_mean=("speedup", "mean"),
        speedup_std=("speedup", "std"),
        count=("dim", "size"),
    ).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Left panel: absolute times ---
    ax = axes[0]
    ax.errorbar(
        agg["dim"], agg["gpu_ms_mean"], yerr=agg["gpu_ms_std"],
        fmt="o-", markersize=4, capsize=3, label="Dense GPU (Cholesky)", color="#534AB7",
    )
    ax.errorbar(
        agg["dim"], agg["cpu_ms_mean"], yerr=agg["cpu_ms_std"],
        fmt="s-", markersize=4, capsize=3, label="Sparse CPU (scipy)", color="#D85A30",
    )
    ax.set_xlabel("System dimension")
    ax.set_ylabel("Solve time (ms)")
    ax.set_title("Solver time vs. system dimension")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Right panel: speedup ---
    ax = axes[1]
    ax.bar(agg["dim"], agg["speedup_mean"], width=agg["dim"].diff().median() * 0.6,
           color="#0F6E56", alpha=0.7, label="GPU speedup (CPU_time / GPU_time)")
    ax.axhline(y=1.0, color="#993C1D", linestyle="--", linewidth=1.5, label="Breakeven (1.0×)")
    ax.set_xlabel("System dimension")
    ax.set_ylabel("Speedup (×)")
    ax.set_title("GPU speedup over CPU")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # --- Find crossover ---
    # Fit a simple model: speedup ≈ a * dim + b, find where it crosses 1.0
    if len(agg) >= 2:
        coeffs = np.polyfit(agg["dim"], agg["speedup_mean"], deg=1)
        crossover_dim = (1.0 - coeffs[1]) / coeffs[0] if abs(coeffs[0]) > 1e-12 else None
        if crossover_dim is not None and crossover_dim > 0:
            ax.axvline(x=crossover_dim, color="#854F0B", linestyle=":", linewidth=1.5,
                       label=f"Est. crossover ≈ dim {int(crossover_dim)}")
            ax.legend()
            print(f"\n  Estimated crossover dimension: {int(crossover_dim)}")
            print(f"  Recommendation: set _DENSE_GPU_DIM_THRESHOLD = {int(crossover_dim)}")

    plt.tight_layout()
    out_path = Path(csv_path).with_suffix(".png")
    plt.savefig(out_path, dpi=150)
    print(f"\n  Saved plot to {out_path}")
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <solver_bench.csv>")
        sys.exit(1)
    main(sys.argv[1])
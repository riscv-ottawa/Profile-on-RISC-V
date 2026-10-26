
#!/usr/bin/env python3

import argparse
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
GEM5_CONFIG = PROJECT_ROOT / "configs/gem5/riscv_baseline.py"
PARSER = PROJECT_ROOT / "scripts/parse_gem5_stats.py"

KERNELS = ("scalar", "rvv")
DEFAULT_SIZES = (32, 64, 128, 256, 528)


def run_experiment(gem5_bin: Path, kernel: str, size: int) -> None:
    output_dir = PROJECT_ROOT / "m5out" / f"{kernel}_{size}"
    stats_path = output_dir / "stats.txt"

    print(f"\nRunning {kernel} GEMM: {size}x{size}", flush=True)

    # Run gem5 with an isolated output directory.
    subprocess.run(
        [
            str(gem5_bin),
            f"--outdir={output_dir}",
            str(GEM5_CONFIG),
            kernel,
            str(size),
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )

    if not stats_path.is_file():
        raise RuntimeError(f"Missing gem5 statistics: {stats_path}")

    # Use our existing parser to save metrics and raw results.
    subprocess.run(
        [
            sys.executable,
            str(PARSER),
            "--implementation",
            kernel,
            "--size",
            str(size),
            "--stats",
            str(stats_path),
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run scalar and RVV GEMM scaling experiments."
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=list(DEFAULT_SIZES),
        help="Matrix dimensions to simulate.",
    )
    parser.add_argument(
        "--kernels",
        nargs="+",
        choices=KERNELS,
        default=list(KERNELS),
        help="GEMM implementations to simulate.",
    )

    args = parser.parse_args()

    if any(size <= 0 for size in args.sizes):
        parser.error("Matrix sizes must be positive.")

    gem5_env = os.environ.get("GEM5_BIN")
    if not gem5_env:
        parser.error("GEM5_BIN is not defined.")

    gem5_bin = Path(gem5_env).expanduser().resolve()
    if not gem5_bin.is_file():
        parser.error(f"gem5 executable not found: {gem5_bin}")

    for kernel in args.kernels:
        for size in args.sizes:
            run_experiment(gem5_bin, kernel, size)

    print("\nAll GEMM experiments completed successfully.")


if __name__ == "__main__":
    main()


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


def parse_shape(text: str) -> tuple[int, int, int]:
    """
    Parse a rectangular GEMM shape written as MxNxK.
    Example: 1x128x128
    """
    parts = text.lower().split("x")

    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"Invalid shape '{text}'. Expected MxNxK, e.g. 1x128x128."
        )

    try:
        M, N, K = (int(part) for part in parts)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Invalid shape '{text}'. Dimensions must be integers."
        )

    if min(M, N, K) <= 0:
        raise argparse.ArgumentTypeError(
            f"Invalid shape '{text}'. Dimensions must be positive."
        )

    return M, N, K


def run_experiment(
    gem5_bin: Path,
    kernel: str,
    shape: tuple[int, int, int],
) -> None:
    M, N, K = shape
    shape_name = f"{M}x{N}x{K}"

    # Preserve existing directory names for square GEMM.
    if M == N == K:
        output_name = f"{kernel}_{M}"
        workload_args = [kernel, str(M)]
    else:
        output_name = f"{kernel}_{shape_name}"
        workload_args = [kernel, str(M), str(N), str(K)]

    output_dir = PROJECT_ROOT / "m5out" / output_name
    stats_path = output_dir / "stats.txt"

    print(
        f"\nRunning {kernel} GEMM: "
        f"M={M}, N={N}, K={K}",
        flush=True,
    )

    # Run gem5 with an isolated output directory.
    subprocess.run(
        [
            str(gem5_bin),
            f"--outdir={output_dir}",
            str(GEM5_CONFIG),
            *workload_args,
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )

    if not stats_path.is_file():
        raise RuntimeError(
            f"Missing gem5 statistics: {stats_path}"
        )

    # Save metrics and archive raw gem5 results.
    subprocess.run(
        [
            sys.executable,
            str(PARSER),
            "--implementation",
            kernel,
            "--m",
            str(M),
            "--n",
            str(N),
            "--k",
            str(K),
            "--stats",
            str(stats_path),
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run scalar and RVV GEMM experiments "
            "for square and rectangular matrices."
        )
    )

    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=None,
        help="Square GEMM dimensions, e.g. --sizes 32 64 128.",
    )

    parser.add_argument(
        "--shapes",
        type=parse_shape,
        nargs="+",
        default=None,
        help=(
            "Rectangular GEMM shapes in MxNxK format, "
            "e.g. --shapes 1x128x128 8x128x128."
        ),
    )

    parser.add_argument(
        "--kernels",
        nargs="+",
        choices=KERNELS,
        default=list(KERNELS),
        help="GEMM implementations to simulate.",
    )

    args = parser.parse_args()

    if args.sizes is not None:
        if any(size <= 0 for size in args.sizes):
            parser.error("Matrix sizes must be positive.")

    shapes = []

    # Default behavior: run original square scaling experiments
    # only when neither --sizes nor --shapes is specified.
    if args.sizes is None and args.shapes is None:
        sizes = DEFAULT_SIZES
    else:
        sizes = args.sizes or []

    for size in sizes:
        shapes.append((size, size, size))

    if args.shapes:
        shapes.extend(args.shapes)

    # Remove duplicate shapes while preserving their order.
    shapes = list(dict.fromkeys(shapes))

    gem5_env = os.environ.get("GEM5_BIN")

    if not gem5_env:
        parser.error("GEM5_BIN is not defined.")

    gem5_bin = Path(gem5_env).expanduser().resolve()

    if not gem5_bin.is_file():
        parser.error(
            f"gem5 executable not found: {gem5_bin}"
        )

    total = len(args.kernels) * len(shapes)
    completed = 0

    print(
        f"Starting {total} GEMM experiments "
        f"across {len(shapes)} unique shapes."
    )

    for kernel in args.kernels:
        for shape in shapes:
            run_experiment(
                gem5_bin,
                kernel,
                shape,
            )

            completed += 1

            print(
                f"Completed {completed}/{total} experiments.",
                flush=True,
            )

    print("\nAll GEMM experiments completed successfully.")


if __name__ == "__main__":
    main()

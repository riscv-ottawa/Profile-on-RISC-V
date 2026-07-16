#!/usr/bin/env python3

import argparse
import csv
import shutil
from pathlib import Path
from typing import Optional


def read_stats(stats_path: Path) -> dict[str, float]:
    """Read numeric statistics from a gem5 stats.txt file."""
    stats: dict[str, float] = {}

    with stats_path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if (
                not line
                or line.startswith("-")
                or line.startswith("#")
            ):
                continue

            parts = line.split()

            if len(parts) < 2:
                continue

            name = parts[0]

            try:
                value = float(parts[1])
            except ValueError:
                continue

            stats[name] = value

    return stats


def find_stat(
    stats: dict[str, float],
    exact_name: Optional[str] = None,
    suffixes: tuple[str, ...] = (),
) -> Optional[float]:
    """
    Find a statistic by its exact name or by a suffix.

    Suffix matching makes the parser less dependent on the complete
    gem5 object path.
    """
    if exact_name is not None and exact_name in stats:
        return stats[exact_name]

    for suffix in suffixes:
        for name, value in stats.items():
            if name.endswith(suffix):
                return value

    return None


def safe_ratio(
    numerator: Optional[float],
    denominator: Optional[float],
) -> Optional[float]:
    if numerator is None or denominator in (None, 0):
        return None

    return numerator / denominator


def format_value(value: Optional[float]) -> str:
    if value is None:
        return ""

    return f"{value:.9f}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract selected GEMM metrics from gem5 stats.txt."
    )

    parser.add_argument(
        "--stats",
        default="m5out/stats.txt",
        help="Path to gem5 stats.txt.",
    )

    parser.add_argument(
        "--size",
        type=int,
        required=True,
        help="Square GEMM matrix dimension.",
    )

    parser.add_argument(
        "--implementation",
        default="scalar",
        help="Kernel implementation name.",
    )

    parser.add_argument(
        "--output",
        help="Output CSV path. Defaults to results/gemm/gemm_<size>.csv.",
    )

    args = parser.parse_args()

    if args.size <= 0:
        raise SystemExit("Matrix size must be greater than zero.")

    stats_path = Path(args.stats)

    if not stats_path.is_file():
        raise SystemExit(f"Stats file not found: {stats_path}")

    stats = read_stats(stats_path)

    sim_seconds = find_stat(stats, exact_name="simSeconds")
    sim_ticks = find_stat(stats, exact_name="simTicks")
    sim_insts = find_stat(stats, exact_name="simInsts")
    sim_ops = find_stat(stats, exact_name="simOps")

    cycles = find_stat(
        stats,
        suffixes=(
            ".numCycles",
            ".num_cycles",
        ),
    )

    l1d_hits = find_stat(
        stats,
        suffixes=(
            "l1d-cache-0.demandHits::total",
            "l1d-cache-0.overallHits::total",
        ),
    )

    l1d_misses = find_stat(
        stats,
        suffixes=(
            "l1d-cache-0.demandMisses::total",
            "l1d-cache-0.overallMisses::total",
        ),
    )

    l1d_accesses = find_stat(
        stats,
        suffixes=(
            "l1d-cache-0.demandAccesses::total",
            "l1d-cache-0.overallAccesses::total",
        ),
    )

    l1i_hits = find_stat(
        stats,
        suffixes=(
            "l1i-cache-0.demandHits::total",
            "l1i-cache-0.overallHits::total",
        ),
    )

    l1i_misses = find_stat(
        stats,
        suffixes=(
            "l1i-cache-0.demandMisses::total",
            "l1i-cache-0.overallMisses::total",
        ),
    )

    l2_hits = find_stat(
        stats,
        suffixes=(
            "l2-cache-0.demandHits::total",
            "l2-cache-0.overallHits::total",
        ),
    )

    l2_misses = find_stat(
        stats,
        suffixes=(
            "l2-cache-0.demandMisses::total",
            "l2-cache-0.overallMisses::total",
        ),
    )

    ipc = safe_ratio(sim_insts, cycles)
    l1d_miss_rate = safe_ratio(l1d_misses, l1d_accesses)

    output_path = (
        Path(args.output)
        if args.output
        else Path(f"results/gemm/gemm_{args.size}.csv")
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "operator",
        "implementation",
        "size",
        "sim_seconds",
        "sim_ticks",
        "cycles",
        "instructions",
        "operations",
        "ipc",
        "l1i_hits",
        "l1i_misses",
        "l1d_hits",
        "l1d_misses",
        "l1d_accesses",
        "l1d_miss_rate",
        "l2_hits",
        "l2_misses",
    ]

    row = {
        "operator": "gemm",
        "implementation": args.implementation,
        "size": args.size,
        "sim_seconds": format_value(sim_seconds),
        "sim_ticks": format_value(sim_ticks),
        "cycles": format_value(cycles),
        "instructions": format_value(sim_insts),
        "operations": format_value(sim_ops),
        "ipc": format_value(ipc),
        "l1i_hits": format_value(l1i_hits),
        "l1i_misses": format_value(l1i_misses),
        "l1d_hits": format_value(l1d_hits),
        "l1d_misses": format_value(l1d_misses),
        "l1d_accesses": format_value(l1d_accesses),
        "l1d_miss_rate": format_value(l1d_miss_rate),
        "l2_hits": format_value(l2_hits),
        "l2_misses": format_value(l2_misses),
    }

    file_exists = output_path.exists()
    file_is_empty = not file_exists or output_path.stat().st_size == 0

    with output_path.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        if file_is_empty:
            writer.writeheader()

        writer.writerow(row)

    raw_result_dir = Path(
        f"results/gemm/raw/{args.size}/{args.implementation}"
    )
    raw_result_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(
        stats_path,
        raw_result_dir / "stats.txt",
    )

    for filename in ("config.ini", "config.json"):
        source = stats_path.parent / filename

        if source.is_file():
            shutil.copy2(source, raw_result_dir / filename)

    print(f"Metrics saved to {output_path}")
    print(f"Raw gem5 output saved to {raw_result_dir}")


if __name__ == "__main__":
    main()
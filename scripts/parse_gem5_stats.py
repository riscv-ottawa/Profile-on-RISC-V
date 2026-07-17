#!/usr/bin/env python3

import argparse
import csv
import shutil
from pathlib import Path
from typing import Optional


FIELDNAMES = [
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


def read_stats(stats_path: Path) -> dict[str, float]:
    """
    Read numeric statistics from a gem5 stats.txt file.

    Each valid gem5 statistic normally follows this structure:

        statistic_name value # optional description

    Non-numeric values and separator lines are ignored.
    """
    stats: dict[str, float] = {}

    with stats_path.open("r", encoding="utf-8") as stats_file:
        for line in stats_file:
            line = line.strip()

            if (
                not line
                or line.startswith("#")
                or line.startswith("-")
            ):
                continue

            parts = line.split()

            if len(parts) < 2:
                continue

            name = parts[0]
            value_text = parts[1]

            try:
                value = float(value_text)
            except ValueError:
                continue

            stats[name] = value

    return stats


def find_stat(
    stats: dict[str, float],
    exact_names: tuple[str, ...] = (),
    suffixes: tuple[str, ...] = (),
) -> Optional[float]:
    """
    Find one gem5 statistic.

    Exact names are checked first. Suffix matching is then used because
    the complete gem5 object path can vary between configurations.
    """
    for name in exact_names:
        if name in stats:
            return stats[name]

    for suffix in suffixes:
        for name, value in stats.items():
            if name.endswith(suffix):
                return value

    return None


def safe_ratio(
    numerator: Optional[float],
    denominator: Optional[float],
) -> Optional[float]:
    """Return numerator / denominator when both values are valid."""
    if numerator is None:
        return None

    if denominator is None or denominator == 0:
        return None

    return numerator / denominator


def format_value(value: Optional[float]) -> str:
    """Convert a metric to a consistent CSV representation."""
    if value is None:
        return ""

    return f"{value:.9f}"


def read_existing_rows(
    output_path: Path,
) -> list[dict[str, str]]:
    """Read rows already present in the unified CSV file."""
    if not output_path.is_file():
        return []

    if output_path.stat().st_size == 0:
        return []

    with output_path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        reader = csv.DictReader(csv_file)

        if reader.fieldnames != FIELDNAMES:
            raise SystemExit(
                f"Unexpected CSV columns in {output_path}.\n"
                "Move or delete the old CSV before using this parser."
            )

        return list(reader)


def write_unified_csv(
    output_path: Path,
    new_row: dict[str, str | int],
) -> None:
    """
    Add one experiment to the unified CSV.

    If the same operator, implementation, and matrix size already exist,
    the old row is replaced rather than duplicated.
    """
    existing_rows = read_existing_rows(output_path)

    filtered_rows = [
        existing_row
        for existing_row in existing_rows
        if not (
            existing_row.get("operator") == str(new_row["operator"])
            and existing_row.get("implementation")
            == str(new_row["implementation"])
            and existing_row.get("size") == str(new_row["size"])
        )
    ]

    normalized_new_row = {
        field: str(new_row.get(field, ""))
        for field in FIELDNAMES
    }

    filtered_rows.append(normalized_new_row)

    filtered_rows.sort(
        key=lambda row: (
            row.get("operator", ""),
            row.get("implementation", ""),
            int(row["size"]),
        )
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=FIELDNAMES,
        )

        writer.writeheader()
        writer.writerows(filtered_rows)


def preserve_raw_output(
    stats_path: Path,
    size: int,
    implementation: str,
) -> Path:
    """
    Preserve stats.txt and gem5 configuration files for reproducibility.
    """
    raw_result_dir = (
        Path("results")
        / "gemm"
        / "raw"
        / str(size)
        / implementation
    )

    raw_result_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copy2(
        stats_path,
        raw_result_dir / "stats.txt",
    )

    for filename in (
        "config.ini",
        "config.json",
    ):
        source_path = stats_path.parent / filename

        if source_path.is_file():
            shutil.copy2(
                source_path,
                raw_result_dir / filename,
            )

    return raw_result_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract selected kernel metrics from gem5 stats.txt "
            "and store them in one unified CSV."
        )
    )

    parser.add_argument(
        "--stats",
        type=Path,
        default=Path("m5out/stats.txt"),
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
        type=Path,
        default=Path("results/gemm/gemm_scalar.csv"),
        help="Path to the unified output CSV.",
    )

    args = parser.parse_args()

    if args.size <= 0:
        raise SystemExit(
            "Matrix size must be greater than zero."
        )

    if not args.stats.is_file():
        raise SystemExit(
            f"Stats file not found: {args.stats}"
        )

    stats = read_stats(args.stats)

    if not stats:
        raise SystemExit(
            f"No numeric gem5 statistics found in {args.stats}"
        )

    sim_seconds = find_stat(
        stats,
        exact_names=("simSeconds",),
    )

    sim_ticks = find_stat(
        stats,
        exact_names=("simTicks",),
    )

    instructions = find_stat(
        stats,
        exact_names=("simInsts",),
    )

    operations = find_stat(
        stats,
        exact_names=("simOps",),
    )

    cycles = find_stat(
        stats,
        suffixes=(
            ".numCycles",
            ".num_cycles",
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

    ipc = safe_ratio(
        instructions,
        cycles,
    )

    # Some gem5 statistics are omitted when their value is zero.
    # If accesses and hits exist, derive the miss count.
    if (
        l1d_misses is None
        and l1d_accesses is not None
        and l1d_hits is not None
    ):l1d_misses = l1d_accesses - l1d_hits

    l1d_miss_rate = safe_ratio(
        l1d_misses,
        l1d_accesses,
    )

    row: dict[str, str | int] = {
        "operator": "gemm",
        "implementation": args.implementation,
        "size": args.size,
        "sim_seconds": format_value(sim_seconds),
        "sim_ticks": format_value(sim_ticks),
        "cycles": format_value(cycles),
        "instructions": format_value(instructions),
        "operations": format_value(operations),
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

    write_unified_csv(
        args.output,
        row,
    )

    raw_result_dir = preserve_raw_output(
        args.stats,
        args.size,
        args.implementation,
    )

    print(
        f"Metrics saved to: {args.output}"
    )
    print(
        f"Raw gem5 output saved to: {raw_result_dir}"
    )
    print(
        f"Recorded GEMM size: {args.size} x {args.size}"
    )


if __name__ == "__main__":
    main()
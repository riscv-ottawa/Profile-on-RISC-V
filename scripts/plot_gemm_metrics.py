
#!/usr/bin/env python3

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt


METRICS = [
    {
        "column": "sim_seconds",
        "title": "Simulated execution time",
        "ylabel": "Time (seconds)",
        "scale": 1.0,
    },
    {
        "column": "instructions",
        "title": "Instruction count",
        "ylabel": "Instructions",
        "scale": 1.0,
    },
    {
        "column": "operations",
        "title": "Operation count",
        "ylabel": "Operations",
        "scale": 1.0,
    },
    {
        "column": "cycles",
        "title": "Cycle count",
        "ylabel": "Cycles",
        "scale": 1.0,
    },
    {
        "column": "ipc",
        "title": "Instructions per cycle",
        "ylabel": "IPC",
        "scale": 1.0,
    },
    {
        "column": "l1i_misses",
        "title": "L1 instruction-cache misses",
        "ylabel": "Misses",
        "scale": 1.0,
    },
    {
        "column": "l1d_misses",
        "title": "L1 data-cache misses",
        "ylabel": "Misses",
        "scale": 1.0,
    },
    {
        "column": "l1d_miss_rate",
        "title": "L1 data-cache miss rate",
        "ylabel": "Miss rate (%)",
        "scale": 100.0,
    },
    {
        "column": "l2_misses",
        "title": "L2 cache misses",
        "ylabel": "Misses",
        "scale": 1.0,
    },
]


# Color represents the simulated hardware configuration.
# Additional configurations can be added here later.
HARDWARE_CONFIGS = {
    "baseline": {
        "label": "1 core, 32 KiB L1D, 256 KiB L2",
        "color": "tab:blue",
    },
}


# Line style represents the GEMM implementation.
LINE_STYLES = {
    "scalar": ":",
    "rvv": "-",
}


def parse_float(
    value: str,
    column: str,
    size: int,
) -> float:
    """
    Convert a CSV value to float and report missing or invalid data.
    """
    if value is None or value.strip() == "":
        raise ValueError(
            f"Missing value for '{column}' at matrix size {size}."
        )

    try:
        return float(value)
    except ValueError as error:
        raise ValueError(
            f"Invalid numeric value for '{column}' "
            f"at matrix size {size}: {value}"
        ) from error


def load_results(
    csv_path: Path,
    implementation: str,
) -> list[dict[str, str]]:
    """
    Read and sort GEMM results for one implementation.
    """
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"CSV file not found: {csv_path}"
        )

    with csv_path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        reader = csv.DictReader(csv_file)

        if reader.fieldnames is None:
            raise ValueError(
                f"CSV file has no header: {csv_path}"
            )

        required_columns = {
            "operator",
            "implementation",
            "size",
        }

        required_columns.update(
            metric["column"]
            for metric in METRICS
        )

        missing_columns = (
            required_columns - set(reader.fieldnames)
        )

        if missing_columns:
            missing_text = ", ".join(
                sorted(missing_columns)
            )

            raise ValueError(
                f"CSV is missing required columns: {missing_text}"
            )

        rows = [
            row
            for row in reader
            if (
                row.get("operator") == "gemm"
                and row.get("implementation") == implementation
            )
        ]

    if not rows:
        raise ValueError(
            "No matching GEMM rows were found for "
            f"implementation '{implementation}' "
            f"in {csv_path}."
        )

    try:
        rows.sort(
            key=lambda row: int(row["size"])
        )
    except ValueError as error:
        raise ValueError(
            "Every matrix size must be an integer."
        ) from error

    return rows


def create_plots(
    results: dict[str, list[dict[str, str]]],
    output_path: Path,
    config_name: str,
    show_plot: bool,
) -> None:
    """
    Plot scalar and RVV GEMM on shared axes for each metric.

    Color represents hardware configuration.
    Line style represents implementation.
    """
    config = HARDWARE_CONFIGS[config_name]

    all_sizes = sorted({
        int(row["size"])
        for rows in results.values()
        for row in rows
    })

    plot_count = len(METRICS)
    column_count = 3
    row_count = math.ceil(plot_count / column_count)

    figure, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(16, 12),
    )

    axes_list = axes.flatten()

    for axis, metric in zip(axes_list, METRICS):
        column = metric["column"]
        scale = metric["scale"]

        for implementation in LINE_STYLES:
            rows = results[implementation]

            sizes = [
                int(row["size"])
                for row in rows
            ]

            values = [
                parse_float(
                    row[column],
                    column,
                    int(row["size"]),
                ) * scale
                for row in rows
            ]

            axis.plot(
                sizes,
                values,
                color=config["color"],
                linestyle=LINE_STYLES[implementation],
                marker="o",
                linewidth=2,
                markersize=5,
                label=implementation.upper(),
            )

        axis.set_title(metric["title"])

        axis.set_xlabel(
            "Matrix size (N for N × N)"
        )

        axis.set_ylabel(metric["ylabel"])

        axis.set_xticks(all_sizes)

        axis.grid(
            True,
            alpha=0.3,
        )

        axis.legend()

        # Keep large metric values readable.
        if column in {
            "instructions",
            "operations",
            "cycles",
            "l1i_misses",
            "l1d_misses",
            "l2_misses",
        }:
            axis.ticklabel_format(
                axis="y",
                style="sci",
                scilimits=(0, 0),
            )

    # Hide unused subplot positions.
    for axis in axes_list[plot_count:]:
        axis.set_visible(False)

    figure.suptitle(
        "GEMM gem5 performance comparison\n"
        f"{config['label']}",
        fontsize=16,
    )

    figure.tight_layout(
        rect=(0, 0, 1, 0.94)
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    print(
        f"Plot saved to: {output_path}"
    )

    if show_plot:
        plt.show()

    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare scalar and RVV GEMM performance "
            "using gem5 statistics."
        )
    )

    parser.add_argument(
        "--scalar",
        type=Path,
        default=Path(
            "results/gemm/gemm_scalar.csv"
        ),
        help="Path to scalar GEMM results.",
    )

    parser.add_argument(
        "--rvv",
        type=Path,
        default=Path(
            "results/gemm/gemm_rvv.csv"
        ),
        help="Path to RVV GEMM results.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/gemm/plots/"
            "gemm_comparison.png"
        ),
        help="Path for the generated plot image.",
    )

    parser.add_argument(
        "--config",
        choices=HARDWARE_CONFIGS.keys(),
        default="baseline",
        help="Hardware configuration used for the experiment.",
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="Open the plot window after saving.",
    )

    args = parser.parse_args()

    try:
        results = {
            "scalar": load_results(
                args.scalar,
                "scalar",
            ),
            "rvv": load_results(
                args.rvv,
                "rvv",
            ),
        }

        create_plots(
            results,
            args.output,
            args.config,
            args.show,
        )

    except (
        FileNotFoundError,
        ValueError,
    ) as error:
        raise SystemExit(
            f"Plotting failed: {error}"
        ) from error


if __name__ == "__main__":
    main()

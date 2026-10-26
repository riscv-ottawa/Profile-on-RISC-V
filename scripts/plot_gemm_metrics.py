#!/usr/bin/env python3
"""Plot gem5 GEMM metrics for square scaling or fixed-N,K rectangular scaling."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
HARDWARE_CONFIGS = {
    'baseline': {'label': '1 core, 32 KiB L1D, 256 KiB L2', 'color': 'tab:blue'},
}
LINE_STYLES = {'scalar': ':', 'rvv': '-'}
METRICS = [
    ('sim_seconds', 'Simulated execution time', 'Seconds'),
    ('instructions', 'Instructions', 'Count'),
    ('operations', 'gem5 simulated operations (simOps)', 'Count'),
    ('cycles', 'CPU cycles', 'Cycles'),
    ('ipc', 'Instructions per cycle', 'IPC'),
    ('l1i_misses', 'L1 instruction-cache misses', 'Misses'),
    ('l1d_misses', 'L1 data-cache misses', 'Misses'),
    ('l1d_miss_rate', 'L1 data-cache miss rate', 'Fraction'),
    ('l2_misses', 'L2 cache misses', 'Misses'),
]


def load_rows(path: Path, implementation: str):
    if not path.is_file():
        raise SystemExit(f'CSV file not found: {path}')
    with path.open(newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise SystemExit(f'CSV is empty: {path}')
        rows = []
        for row in reader:
            if row.get('implementation', implementation) != implementation:
                continue
            try:
                if 'M' in row and 'N' in row and 'K' in row:
                    m, n, k = (int(row[d]) for d in ('M', 'N', 'K'))
                elif 'size' in row:  # Read old square CSVs too.
                    m = n = k = int(row['size'])
                else:
                    raise ValueError('missing M/N/K or size columns')
            except (ValueError, TypeError) as exc:
                raise SystemExit(f'Invalid dimensions in {path}: {exc}')
            rows.append((m, n, k, row))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scalar', type=Path, default=PROJECT_ROOT / 'results/gemm/gemm_scalar.csv')
    parser.add_argument('--rvv', type=Path, default=PROJECT_ROOT / 'results/gemm/gemm_rvv.csv')
    parser.add_argument('--mode', choices=('square', 'rectangular'), default='square')
    parser.add_argument('--n', type=int, help='Fixed N for rectangular mode')
    parser.add_argument('--k', type=int, help='Fixed K for rectangular mode')
    parser.add_argument('--config', choices=tuple(HARDWARE_CONFIGS), default='baseline')
    parser.add_argument('--output', type=Path, default=None)
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()

    if args.mode == 'rectangular' and (args.n is None or args.k is None or args.n <= 0 or args.k <= 0):
        parser.error('Rectangular mode requires positive --n and --k.')

    selected = {}
    for implementation, path in (('scalar', args.scalar), ('rvv', args.rvv)):
        rows = load_rows(path, implementation)
        if args.mode == 'square':
            filtered = [(m, row) for m, n, k, row in rows if m == n == k]
        else:
            filtered = [(m, row) for m, n, k, row in rows if n == args.n and k == args.k]
        selected[implementation] = sorted(filtered, key=lambda item: item[0])

    if not any(selected.values()):
        raise SystemExit('No matching GEMM experiments found in the CSV files.')

    if args.output is None:
        filename = ('gemm_comparison.png' if args.mode == 'square'
                    else f'gemm_rectangular_N{args.n}_K{args.k}.png')
        args.output = PROJECT_ROOT / 'results/gemm/plots' / filename

    config = HARDWARE_CONFIGS[args.config]
    fig, axes = plt.subplots(3, 3, figsize=(17, 12), constrained_layout=True)
    for ax, (metric, title, ylabel) in zip(axes.flat, METRICS):
        for implementation in ('scalar', 'rvv'):
            points = []
            for x, row in selected[implementation]:
                value = row.get(metric, '')
                if value is None or value.strip() == '':
                    continue
                try:
                    points.append((x, float(value)))
                except ValueError:
                    continue
            if points:
                ax.plot([p[0] for p in points], [p[1] for p in points],
                        linestyle=LINE_STYLES[implementation], marker='o',
                        color=config['color'], label=implementation.upper())
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel('Square dimension (M = N = K)' if args.mode == 'square' else 'Input rows (M)')
        x_values = sorted({x for group in selected.values() for x, _ in group})
        ax.set_xticks(x_values)
        ax.grid(True, alpha=0.3)
        ax.legend()

    title = ('Square GEMM scaling' if args.mode == 'square'
             else f'Rectangular GEMM scaling (N={args.n}, K={args.k})')
    fig.suptitle(f'{title}\n{config["label"]}', fontsize=15)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    print(f'Plot saved to: {args.output}')
    if args.show:
        # Agg is a noninteractive backend; use a separate image viewer to open the file.
        print('Open the saved PNG in your image viewer (--show does not open a window).')
    plt.close(fig)


if __name__ == '__main__':
    main()

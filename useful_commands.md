# Useful Commands — AI Workload Profiler for RISC-V

This document provides the commands used to build, run, and analyze GEMM workloads using gem5 and RISC-V Vector Extensions (RVV).

All commands should be executed from the **project root directory** unless otherwise specified.

## 1. Environment Setup

### Set the gem5 executable path

```bash
export GEM5_BIN="/path/to/gem5/build/ALL/gem5.opt"
```

Replace the path with the location of your gem5 installation.

This environment variable must be set again when opening a new terminal, unless it has been added to your shell configuration.

### Set the gem5 root directory

```bash
export GEM5_ROOT="$(dirname "$(dirname "$(dirname "$GEM5_BIN")")")"
```

### Activate the Python virtual environment

```bash
source .venv/bin/activate
```

The Python environment should contain the project's required packages, including Matplotlib for plotting.

## 2. Compile the RISC-V GEMM Benchmark

Compile the scalar and RVV kernels into one RISC-V executable:

```bash
mkdir -p build

riscv64-linux-gnu-gcc \
    -O2 -static -march=rv64gcv -mabi=lp64d \
    -I"$GEM5_ROOT/include" \
    benchmark/benchmark_gemm.c \
    kernels/gemm_scalar.c \
    kernels/gemm_rvv.c \
    "$GEM5_ROOT/util/m5/build/riscv/out/libm5.a" \
    -o build/gemm_riscv
```

**Important flags:**

- `-O2`: Enables compiler optimizations.
- `-static`: Statically links the executable for gem5 syscall emulation (SE) mode.
- `-march=rv64gcv`: Targets 64-bit RISC-V with vector extension support.
- `-mabi=lp64d`: Uses the RISC-V LP64D ABI.
- `libm5.a`: Provides the gem5 ROI instrumentation functions.

Recompile after modifying the benchmark or either GEMM kernel.

## 3. Run Individual gem5 Simulations

### Square GEMM

```bash
"$GEM5_BIN" \
    --outdir=m5out/rvv_128 \
    configs/gem5/riscv_baseline.py \
    rvv 128
```

This simulates a 128×128×128 GEMM using the RVV kernel.

### Rectangular GEMM

```bash
"$GEM5_BIN" \
    --outdir=m5out/rvv_1x128x128 \
    configs/gem5/riscv_baseline.py \
    rvv 1 128 128
```

The argument order is:

```text
<kernel> <M> <N> <K>
```

The matrix multiplication is:

```text
A (M × K) × B (K × N) = C (M × N)
```

Supported kernels are `scalar` and `rvv`.

The simulation outputs `stats.txt`, `config.ini`, and `config.json` in the specified output directory.

## 4. Run Automated GEMM Experiments

The experiment runner launches gem5 simulations and automatically parses and archives their statistics.

### Run default square GEMM experiments

```bash
python3 scripts/run_gemm_experiments.py
```

Runs square matrix sizes 32, 64, 128, 256, and 528 with both scalar and RVV kernels.

### Run selected square sizes

```bash
python3 scripts/run_gemm_experiments.py --sizes 32 64 128
```

### Run rectangular GEMM experiments

```bash
python3 scripts/run_gemm_experiments.py \
    --shapes 1x128x128 4x128x128 16x128x128 64x128x128 128x128x128
```

This runs five shapes with both scalar and RVV implementations, resulting in 10 gem5 simulations.

### Run only one implementation

```bash
python3 scripts/run_gemm_experiments.py \
    --shapes 1x128x128 \
    --kernels rvv
```

### Combine square and rectangular workloads

```bash
python3 scripts/run_gemm_experiments.py \
    --sizes 32 64 \
    --shapes 1x128x128 8x128x128 \
    --kernels scalar rvv
```

**Note:** gem5 simulations can take considerable wall-clock time, particularly for larger scalar GEMM workloads.

## 5. Parse gem5 Statistics Manually

The experiment runner invokes the statistics parser automatically. These commands are useful when running gem5 directly or reprocessing an existing simulation.

### Square GEMM

```bash
python3 scripts/parse_gem5_stats.py \
    --implementation rvv \
    --size 128 \
    --stats m5out/rvv_128/stats.txt
```

### Rectangular GEMM

```bash
python3 scripts/parse_gem5_stats.py \
    --implementation rvv \
    --m 1 --n 128 --k 128 \
    --stats m5out/rvv_1x128x128/stats.txt
```

Parsed results are stored in:

```text
results/gemm/gemm_scalar.csv
results/gemm/gemm_rvv.csv
```

Raw statistics and gem5 configuration files are archived under:

```text
results/gemm/raw/<shape>/<implementation>/
```

Repeated experiments with the same operator, implementation, and matrix dimensions replace their previous CSV row.

## 6. Generate Performance Plots

### Square GEMM scaling

```bash
python3 scripts/plot_gemm_metrics.py
```

### Rectangular GEMM scaling

```bash
python3 scripts/plot_gemm_metrics.py \
    --mode rectangular \
    --n 128 \
    --k 128
```

The rectangular plotting mode compares workloads with different `M` values while holding `N` and `K` constant.

Generated plots are saved under:

```text
results/gemm/plots/
```

For example:

```text
gemm_comparison.png
gemm_rectangular_N128_K128.png
```

## 7. Inspect Simulation Statistics

To inspect instruction counts and cycles:

```bash
grep -E "simInsts|numCycles" m5out/rvv_1x128x128/stats.txt
```

To inspect cache statistics:

```bash
grep -E "demandHits|demandMisses|demandAccesses" \
    m5out/rvv_1x128x128/stats.txt
```

To inspect simulated execution time:

```bash
grep "simSeconds" m5out/rvv_1x128x128/stats.txt
```

`simSeconds` measures simulated time, not the wall-clock time spent running gem5 on the host computer.

The statistics are collected within the GEMM Region of Interest, delimited by `m5_work_begin()` and `m5_work_end()`.

## 8. Current Baseline Hardware Configuration

The simulated architecture is defined in:

```text
configs/gem5/riscv_baseline.py
```

| Component | Configuration |
|---|---|
| ISA | RISC-V |
| CPU | TimingSimpleCPU |
| Cores | 1 |
| Clock | 1 GHz |
| L1 instruction cache | 32 KiB |
| L1 data cache | 32 KiB |
| L2 cache | 256 KiB |
| Memory | 1 GiB DDR3-1600 |
| RVV VLEN | 256 bits |
| RVV ELEN | 64 bits |

When comparing workload performance, keep the hardware configuration unchanged unless the experiment explicitly investigates an architectural modification.

## 9. Typical Development Workflow

After modifying a GEMM kernel or benchmark:

1. Recompile `build/gemm_riscv`.
2. Run a small gem5 experiment.
3. Check that the simulation completed and produced `stats.txt`.
4. Run the automated experiments for the desired matrix shapes.
5. Generate the corresponding performance plots.
6. Analyze the results using simulated cycles, instruction counts, cache statistics, and workload dimensions.

For changes to Python analysis or plotting scripts, recompilation of the RISC-V executable is not required.

## 10. Important Notes for Contributors

- The current profiler supports scalar and RVV GEMM workloads.
- Matrix dimensions are specified in `M × N × K` order.
- The simulator runs in gem5 syscall emulation (SE) mode.
- gem5's `simOps` statistic is not equivalent to GEMM floating-point operations (FLOPs).
- For an FP32 GEMM, the approximate arithmetic workload is `2 × M × N × K` FLOPs.
- Cache miss counts alone are not sufficient to establish that a workload is memory-bandwidth-bound.
- Avoid comparing measurements from different hardware configurations without explicitly recording those differences.

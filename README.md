# RISC-V AI Operator Profiling Framework

A reproducible framework for profiling AI operators on RISC-V processors.

This project provides a reproducible workflow for profiling AI operators on RISC-V. It demonstrates how high-level code is translated into RISC-V instructions, how different implementations affect instruction count and memory behavior, and how to reason about whether an operator is **compute-bound**, **memory-bound**, or limited by **instruction overhead**.

Rather than focusing solely on optimization, this repository emphasizes understanding the relationship between software, the compiler, the instruction set architecture (ISA), and the underlying hardware. The methodology is designed to be reusable so that any AI operator can be analyzed using the same process.

---

# Profiling Methodology

Every operator in this repository is analyzed using the same workflow.

```text
Choose AI operator
        ↓
Implement scalar reference
        ↓
Verify correctness
        ↓
Generate RISC-V assembly
        ↓
Study compiler optimizations
        ↓
Implement optimized version (RVV, blocking, etc.)
        ↓
Generate optimized assembly
        ↓
Collect performance metrics
        ↓
Analyze hardware bottlenecks
        ↓
Document findings
```

The goal is to understand **how software decisions translate into hardware behavior**, rather than simply producing a faster implementation.

---

# Current Case Study: General Matrix Multiplication (GEMM)

The first case study in this repository focuses on **General Matrix Multiplication (GEMM)**, one of the most important kernels in machine learning and scientific computing.

GEMM was selected because it exercises nearly every major component of a modern processor:

* Floating-point execution units
* Memory hierarchy
* Cache behavior
* Instruction scheduling
* Register allocation
* Vector execution units

As a result, it provides an excellent foundation for learning how AI workloads interact with computer architecture.

For this case study, we compare a scalar implementation against an implementation using the **RISC-V Vector Extension (RVV)** and investigate how each version behaves from both a software and hardware perspective.

---

# What We Analyze

For each implementation, we aim to answer questions such as:

* How did the compiler transform the original C code?
* Which optimizations were applied automatically?
* How does RVV change the generated instructions?
* How many instructions are executed?
* How much data is moved through memory?
* Which instructions dominate execution?
* Is performance limited by computation, memory accesses, or instruction overhead?
* Which optimizations require compiler improvements, and which require architectural support?

Where available, the analysis will include metrics such as:

* Execution time
* Instruction count
* Instruction mix
* Memory accesses
* FLOPs
* Arithmetic intensity
* IPC (Instructions Per Cycle)
* Cache behavior
* Vector instruction utilization

---

# Repository Structure

```text
.
├── benchmark/        Benchmark harnesses
├── docs/             Methodology and technical documentation
├── experiments/      Small programs exploring individual architecture concepts
├── include/          Shared headers
├── kernels/          AI operator implementations
├── results/          Benchmark results and analysis
├── scripts/          Build and helper scripts
├── tests/            Correctness tests
├── CONTRIBUTING.md   Contribution guide
└── README.md
```

---

# Long-Term Vision

The long-term objective is to build a collection of AI operator case studies that all follow the same profiling methodology.

Future operators may include:

* ReLU
* LayerNorm
* Softmax
* Attention primitives
* Convolution

By keeping the workflow consistent across operators, the repository becomes both a learning resource and a reusable framework for understanding how AI workloads interact with RISC-V hardware.

---

# Contributing

Contributions are welcome.

Whether you want to implement a new operator, improve an existing optimization, extend the benchmarking infrastructure, or contribute to the architectural analysis, please read **CONTRIBUTING.md** before getting started.

The contribution guide explains the purpose of each directory, the expected workflow, and how new case studies should be structured so that they remain consistent with the profiling methodology.

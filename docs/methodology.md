# Profiling Methodology

This repository provides a reproducible workflow for profiling AI operators on RISC-V.

The objective is to understand how software decisions translate into hardware behavior and explain the performance characteristics of an AI operator.

Every operator follows the same workflow:

1. Choose an AI operator.
2. Implement a scalar reference.
3. Verify correctness.
4. Generate RISC-V assembly.
5. Analyze compiler optimizations.
6. Implement an optimized version (RVV, blocking, etc.).
7. Generate optimized assembly.
8. Execute on a RISC-V simulator.
9. Collect architectural metrics.
10. Classify hardware bottlenecks.
11. Document findings.

The methodology is designed to be reusable for any AI operator.
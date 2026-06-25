

riscv64-linux-gnu-gcc \
  -O0 \
  -S \
  -Iinclude \
  kernels/gemm_scalar.c \
  -o build/asm/gemm_scalar_O0.s

riscv64-linux-gnu-gcc -O2 -S -Iinclude kernels/gemm_scalar.c -o build/asm/gemm_scalar_REAL_O2.s

code build/asm/gemm_scalar_O0.s
#ifndef GEMM_H
#define GEMM_H

#include <stddef.h>

void gemm_scalar_f32(
    const float *A,
    const float *B,
    float *C,
    size_t M,
    size_t N,
    size_t K
);

#endif
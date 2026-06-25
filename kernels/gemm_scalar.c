#include <stddef.h>

/*
 * Scalar FP32 GEMM
 *
 * Computes:
 *   C = A x B
 *
 * Matrix layout:
 *   A is M x K, row-major
 *   B is K x N, row-major
 *   C is M x N, row-major
 *
 * This is the baseline implementation.
 * It intentionally does not use RVV intrinsics.
 */
void gemm_scalar_f32(
    const float *A,
    const float *B,
    float *C,
    size_t M,
    size_t N,
    size_t K
) {
    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            float sum = 0.0f;

            for (size_t k = 0; k < K; k++) {
                float a = A[i * K + k]; //Accessed A[] instead of A[][] because 
                //RAM is a huge 1D array that represents the 2D array conceptually
                float b = B[k * N + j];
                sum += a * b;
            }

            C[i * N + j] = sum;
        }
    }
}
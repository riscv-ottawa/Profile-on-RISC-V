#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "../include/gemm.h"

static int test_gemm(size_t M, size_t N, size_t K)
{
    size_t a_size = M * K;
    size_t b_size = K * N;
    size_t c_size = M * N;

    float *A = malloc(a_size * sizeof(float));
    float *B = malloc(b_size * sizeof(float));
    float *C_scalar = malloc(c_size * sizeof(float));
    float *C_rvv = malloc(c_size * sizeof(float));

    if (!A || !B || !C_scalar || !C_rvv) {
        fprintf(stderr, "Memory allocation failed\n");
        free(A);
        free(B);
        free(C_scalar);
        free(C_rvv);
        return 0;
    }

    for (size_t i = 0; i < a_size; i++)
        A[i] = (float)((int)(i % 7) - 3) / 7.0f;

    for (size_t i = 0; i < b_size; i++)
        B[i] = (float)((int)(i % 11) - 5) / 11.0f;

    gemm_scalar_f32(A, B, C_scalar, M, N, K);
    gemm_rvv_f32(A, B, C_rvv, M, N, K);

    int passed = 1;

    for (size_t i = 0; i < c_size; i++) {
        float expected = C_scalar[i];
        float actual = C_rvv[i];

        float tolerance = 1e-5f + 1e-4f * fabsf(expected);

        if (!isfinite(actual) ||
            fabsf(expected - actual) > tolerance) {

            fprintf(stderr,
                    "FAIL: M=%zu N=%zu K=%zu index=%zu "
                    "expected=%f actual=%f\n",
                    M, N, K, i, expected, actual);

            passed = 0;
            break;
        }
    }

    printf("GEMM %zux%zu x %zux%zu: %s\n",
           M, K, K, N, passed ? "PASS" : "FAIL");

    free(A);
    free(B);
    free(C_scalar);
    free(C_rvv);

    return passed;
}

int main(void)
{
    int passed = 1;

    passed &= test_gemm(4, 4, 4);
    passed &= test_gemm(8, 8, 8);
    passed &= test_gemm(7, 13, 5);
    passed &= test_gemm(16, 17, 9);
    passed &= test_gemm(32, 32, 32);

    if (!passed) {
        fprintf(stderr, "RVV GEMM correctness test FAILED\n");
        return EXIT_FAILURE;
    }

    printf("All RVV GEMM correctness tests passed.\n");

    return EXIT_SUCCESS;
}
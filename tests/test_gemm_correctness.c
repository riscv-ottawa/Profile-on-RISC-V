
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>

#include "../include/gemm.h"

static void reference_gemm(
    const float *A,
    const float *B,
    float *C,
    size_t M,
    size_t N,
    size_t K
) {
    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            double sum = 0.0;

            for (size_t k = 0; k < K; k++) {
                sum += (double)A[i * K + k] *
                       (double)B[k * N + j];
            }

            C[i * N + j] = (float)sum;
        }
    }
}

static int compare_matrices(
    const float *reference,
    const float *actual,
    size_t elements,
    const char *kernel
) {
    const float atol = 1e-4f;
    const float rtol = 1e-4f;

    for (size_t i = 0; i < elements; i++) {
        float expected = reference[i];
        float observed = actual[i];

        float error = fabsf(expected - observed);
        float tolerance = atol + rtol * fabsf(expected);

        if (!isfinite(observed) || error > tolerance) {
            printf(
                "FAIL [%s] index=%zu expected=%f actual=%f\n",
                kernel, i, expected, observed
            );
            return 0;
        }
    }

    return 1;
}

static int test_shape(size_t M, size_t N, size_t K) {
    size_t a_count = M * K;
    size_t b_count = K * N;
    size_t c_count = M * N;

    float *A = malloc(a_count * sizeof(float));
    float *B = malloc(b_count * sizeof(float));
    float *reference = malloc(c_count * sizeof(float));
    float *scalar = malloc(c_count * sizeof(float));
    float *rvv = malloc(c_count * sizeof(float));

    if (!A || !B || !reference || !scalar || !rvv) {
        fprintf(stderr, "Allocation failed\n");
        free(A);
        free(B);
        free(reference);
        free(scalar);
        free(rvv);
        return 0;
    }

    for (size_t i = 0; i < a_count; i++)
        A[i] = (float)((int)(i % 17) - 8) / 17.0f;

    for (size_t i = 0; i < b_count; i++)
        B[i] = (float)((int)(i % 19) - 9) / 19.0f;

    reference_gemm(A, B, reference, M, N, K);

    // Initialize outputs before running kernels.
    for (size_t i = 0; i < c_count; i++) {
        scalar[i] = 0.0f;
        rvv[i] = 0.0f;
    }

    gemm_scalar_f32(A, B, scalar, M, N, K);
    gemm_rvv_f32(A, B, rvv, M, N, K);

    int scalar_ok = compare_matrices(
        reference, scalar, c_count, "scalar"
    );

    int rvv_ok = compare_matrices(
        reference, rvv, c_count, "rvv"
    );

    printf(
        "GEMM %zux%zux%zu: scalar=%s, rvv=%s\n",
        M, N, K,
        scalar_ok ? "PASS" : "FAIL",
        rvv_ok ? "PASS" : "FAIL"
    );

    free(A);
    free(B);
    free(reference);
    free(scalar);
    free(rvv);

    return scalar_ok && rvv_ok;
}

int main(void) {
    const size_t shapes[][3] = {
        {1, 1, 1},
        {1, 128, 128},
        {4, 128, 128},
        {16, 128, 128},
        {64, 128, 128},
        {128, 128, 128},
        {3, 13, 17},
        {7, 19, 31}
    };

    int all_passed = 1;

    size_t count = sizeof(shapes) / sizeof(shapes[0]);

    for (size_t i = 0; i < count; i++) {
        if (!test_shape(
            shapes[i][0],
            shapes[i][1],
            shapes[i][2]
        )) {
            all_passed = 0;
        }
    }

    printf(
        "\nOverall correctness: %s\n",
        all_passed ? "PASS" : "FAIL"
    );

    return all_passed ? EXIT_SUCCESS : EXIT_FAILURE;
}

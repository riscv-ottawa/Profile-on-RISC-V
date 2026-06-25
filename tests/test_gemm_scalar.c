#include <stdio.h>
#include <math.h>
#include "../include/gemm.h"

#define EPSILON 1e-5f

static int almost_equal(float a, float b) {
    return fabsf(a - b) < EPSILON;
}

int main(void) {
    /*
     * A is 2 x 3
     *
     * [1 2 3]
     * [4 5 6]
     */
    float A[6] = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f
    };

    /*
     * B is 3 x 2
     *
     * [ 7  8]
     * [ 9 10]
     * [11 12]
     */
    float B[6] = {
        7.0f,  8.0f,
        9.0f, 10.0f,
        11.0f, 12.0f
    };

    /*
     * Expected C is 2 x 2
     *
     * [ 58  64]
     * [139 154]
     */
    float expected_C[4] = {
        58.0f, 64.0f,
        139.0f, 154.0f
    };

    float C[4] = {0.0f};

    gemm_scalar_f32(A, B, C, 2, 2, 3);

    for (size_t i = 0; i < 4; i++) {
        if (!almost_equal(C[i], expected_C[i])) {
            printf("Test failed at index %zu: got %f, expected %f\n",
                   i, C[i], expected_C[i]);
            return 1;
        }
    }

    printf("gemm_scalar_f32 test passed.\n");
    return 0;
}
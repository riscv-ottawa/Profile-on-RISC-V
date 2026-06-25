#include <stdio.h>
#include <stdlib.h>
#define _POSIX_C_SOURCE 199309L 
#include <time.h>
#include "../include/gemm.h"

typedef void (*gemm_f32_fn)(
    const float *A,
    const float *B,
    float *C,
    size_t M,
    size_t N,
    size_t K
);

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void fill_matrix(float *matrix, size_t elements) {
    for (size_t i = 0; i < elements; i++) {
        matrix[i] = (float)((i % 13) + 1) / 13.0f;
    }
}

static void zero_matrix(float *matrix, size_t elements) {
    for (size_t i = 0; i < elements; i++) {
        matrix[i] = 0.0f;
    }
}

static void benchmark_gemm(
    const char *kernel_name,
    gemm_f32_fn kernel,
    size_t size,
    int trials
) {
    size_t M = size;
    size_t N = size;
    size_t K = size;

    size_t a_elements = M * K;
    size_t b_elements = K * N;
    size_t c_elements = M * N;

    float *A = malloc(a_elements * sizeof(float));
    float *B = malloc(b_elements * sizeof(float));
    float *C = malloc(c_elements * sizeof(float));

    if (A == NULL || B == NULL || C == NULL) {
        fprintf(stderr, "Allocation failed for size %zu\n", size);
        free(A);
        free(B);
        free(C);
        exit(1);
    }

    fill_matrix(A, a_elements);
    fill_matrix(B, b_elements);

    double total_time = 0.0;

    for (int t = 0; t < trials; t++) {
        zero_matrix(C, c_elements);

        double start = now_seconds();
        kernel(A, B, C, M, N, K);
        double end = now_seconds();

        total_time += end - start;
    }

    double avg_time = total_time / trials;

    printf("%s,%zu,%d,%.9f\n", kernel_name, size, trials, avg_time);

    free(A);
    free(B);
    free(C);
}

int main(void) {
    size_t sizes[] = {32, 64, 128, 256, 512};
    int trials = 3;

    printf("kernel,size,trials,avg_time_seconds\n");

    for (size_t i = 0; i < sizeof(sizes) / sizeof(sizes[0]); i++) {
        benchmark_gemm("scalar", gemm_scalar_f32, sizes[i], trials);
    }

    return 0;
}
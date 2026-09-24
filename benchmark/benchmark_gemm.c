#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <sys/stat.h>
#include <gem5/m5ops.h>

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

    return (double)ts.tv_sec
         + (double)ts.tv_nsec * 1e-9;
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


static int file_is_empty(FILE *file) {
    if (fseek(file, 0, SEEK_END) != 0) {
        return 1;
    }

    long size = ftell(file);

    return size <= 0;
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
        fprintf(
            stderr,
            "Allocation failed for size %zu\n",
            size
        );

        free(A);
        free(B);
        free(C);

        exit(EXIT_FAILURE);
    }

    /*
     * These operations happen before the ROI.
     * Their instructions and cache activity will not be included
     * after gem5 resets its statistics at WORKBEGIN.
     */
    fill_matrix(A, a_elements);
    fill_matrix(B, b_elements);

    double total_time = 0.0;

    for (int t = 0; t < trials; t++) {
        /*
         * Clear C before starting the ROI, because we want to profile
         * GEMM rather than matrix initialization.
         */
        zero_matrix(C, c_elements);

        double start = now_seconds();

        /*
         * Begin the gem5 Region of Interest.
         *
         * work_id  = 0
         * thread_id = 0
         */
        m5_work_begin(0, 0);

        kernel(A, B, C, M, N, K);

        /*
         * End the gem5 Region of Interest.
         */
        m5_work_end(0, 0);

        double end = now_seconds();

        total_time += end - start;
    }

    double avg_time = total_time / trials;

    char filename[256];

    snprintf(
        filename,
        sizeof(filename),
        "results/gemm/gemm_%zu.csv",
        size
    );

    FILE *file = fopen(filename, "a+");

    if (file == NULL) {
        fprintf(
            stderr,
            "Could not open result file: %s\n",
            filename
        );

        free(A);
        free(B);
        free(C);

        exit(EXIT_FAILURE);
    }

    if (file_is_empty(file)) {
        fprintf(
            file,
            "kernel,size,trials,avg_time_seconds\n"
        );
    }

    fprintf(
        file,
        "%s,%zu,%d,%.9f\n",
        kernel_name,
        size,
        trials,
        avg_time
    );

    fclose(file);

    printf(
        "%s,%zu,%d,%.9f\n",
        kernel_name,
        size,
        trials,
        avg_time
    );

    free(A);
    free(B);
    free(C);
}


int main(int argc, char *argv[]) {
    if (argc != 3) {
        fprintf(
            stderr,
            "Usage: %s <kernel> <matrix_size>\n",
            argv[0]
        );

        return EXIT_FAILURE;
    }

    const char *kernel_name = argv[1];
    gemm_f32_fn kernel = NULL;

    if (strcmp(kernel_name, "scalar") == 0) {
        kernel = gemm_scalar_f32;
    }
    else if (strcmp(kernel_name, "rvv") == 0) {
        kernel = gemm_rvv_f32;
    }
    else {
        fprintf(
            stderr,
            "Unknown kernel: %s\n"
            "Available kernels: scalar, rvv\n",
            kernel_name
        );

        return EXIT_FAILURE;
    }

    char *end = NULL;
    unsigned long long parsed_size = strtoull(
        argv[2],
        &end,
        10
    );

    if (
        end == argv[2]
        || *end != '\0'
        || parsed_size == 0
    ) {
        fprintf(
        stderr,
        "Invalid matrix size: %s\n",
        argv[2]
    );
        return EXIT_FAILURE;
    }

    size_t size = (size_t)parsed_size;

    /*
     * Use one trial while validating ROI handling.
     */
    int trials = 1;

    printf("kernel,size,trials,avg_time_seconds\n");

    benchmark_gemm(
        kernel_name,
        kernel,
        size,
        trials
    );

    return EXIT_SUCCESS;
}
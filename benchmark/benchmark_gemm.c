
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdint.h>
#include <errno.h>
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


/*
 * Parse a strictly positive matrix dimension.
 */
static size_t parse_dimension(const char *text) {
    if (
        text == NULL
        || text[0] == '\0'
        || text[0] == '-'
    ) {
        fprintf(
            stderr,
            "Invalid matrix dimension: %s\n",
            text ? text : "(null)"
        );

        exit(EXIT_FAILURE);
    }

    char *end = NULL;

    errno = 0;

    unsigned long long value = strtoull(
        text,
        &end,
        10
    );

    if (
        errno == ERANGE
        || end == text
        || *end != '\0'
        || value == 0
        || value > SIZE_MAX
    ) {
        fprintf(
            stderr,
            "Invalid matrix dimension: %s\n",
            text
        );

        exit(EXIT_FAILURE);
    }

    return (size_t)value;
}


/*
 * Check multiplication before allocating memory.
 * Returns the number of float elements.
 */
static size_t checked_elements(
    size_t rows,
    size_t columns
) {
    if (
        rows > SIZE_MAX / columns
    ) {
        fprintf(
            stderr,
            "Matrix dimension overflow: %zu x %zu\n",
            rows,
            columns
        );

        exit(EXIT_FAILURE);
    }

    size_t elements = rows * columns;

    if (
        elements > SIZE_MAX / sizeof(float)
    ) {
        fprintf(
            stderr,
            "Matrix allocation size overflow: %zu x %zu\n",
            rows,
            columns
        );

        exit(EXIT_FAILURE);
    }

    return elements;
}


static void benchmark_gemm(
    const char *kernel_name,
    gemm_f32_fn kernel,
    size_t M,
    size_t N,
    size_t K,
    int trials
) {
    size_t a_elements = checked_elements(M, K);
    size_t b_elements = checked_elements(K, N);
    size_t c_elements = checked_elements(M, N);

    float *A = malloc(a_elements * sizeof(float));
    float *B = malloc(b_elements * sizeof(float));
    float *C = malloc(c_elements * sizeof(float));

    if (A == NULL || B == NULL || C == NULL) {
        fprintf(
            stderr,
            "Allocation failed for GEMM M=%zu, N=%zu, K=%zu\n",
            M,
            N,
            K
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
         * work_id   = 0
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
        "results/gemm/gemm_%zux%zux%zu.csv",
        M,
        N,
        K
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
            "kernel,M,N,K,trials,avg_time_seconds\n"
        );
    }

    fprintf(
        file,
        "%s,%zu,%zu,%zu,%d,%.9f\n",
        kernel_name,
        M,
        N,
        K,
        trials,
        avg_time
    );

    fclose(file);

    printf(
        "%s,%zu,%zu,%zu,%d,%.9f\n",
        kernel_name,
        M,
        N,
        K,
        trials,
        avg_time
    );

    free(A);
    free(B);
    free(C);
}


int main(int argc, char *argv[]) {
    if (argc != 3 && argc != 5) {
        fprintf(
            stderr,
            "Usage:\n"
            "  %s <kernel> <size>\n"
            "  %s <kernel> <M> <N> <K>\n",
            argv[0],
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

    size_t M;
    size_t N;
    size_t K;

    if (argc == 3) {
        /*
         * Backward-compatible square GEMM:
         * <kernel> <size>
         */
        size_t size = parse_dimension(argv[2]);

        M = size;
        N = size;
        K = size;
    }
    else {
        /*
         * Rectangular GEMM:
         * <kernel> <M> <N> <K>
         */
        M = parse_dimension(argv[2]);
        N = parse_dimension(argv[3]);
        K = parse_dimension(argv[4]);
    }

    /*
     * Use one trial while validating ROI handling.
     */
    int trials = 1;

    printf(
        "kernel,M,N,K,trials,avg_time_seconds\n"
    );

    benchmark_gemm(
        kernel_name,
        kernel,
        M,
        N,
        K,
        trials
    );

    return EXIT_SUCCESS;
}

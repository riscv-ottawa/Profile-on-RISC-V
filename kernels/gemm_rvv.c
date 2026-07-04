#include <stddef.h>
#include <riscv_vector.h>

void gemm_rvv_f32(
    const float *A,
    const float *B,
    float *C,
    size_t M,
    size_t N,
    size_t K
) {
    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            size_t k = 0;

            vfloat32m1_t vacc = __riscv_vfmv_v_f_f32m1(0.0f, 1);

            while (k < K) {
                size_t remaining = K - k;
                size_t vl = __riscv_vsetvl_e32m1(remaining);

                const float *a_ptr = &A[i * K + k];
                const float *b_ptr = &B[k * N + j];

                vfloat32m1_t va = __riscv_vle32_v_f32m1(a_ptr, vl);

                /*
                 * B is accessed down a column:
                 * B[k][j], B[k+1][j], B[k+2][j], ...
                 *
                 * Since B is row-major, consecutive column elements
                 * are N floats apart.
                 */
                vfloat32m1_t vb = __riscv_vlse32_v_f32m1(
                    b_ptr,
                    N * sizeof(float),
                    vl
                );

                vacc = __riscv_vfmacc_vv_f32m1(vacc, va, vb, vl);

                k += vl;
            }

            vfloat32m1_t zero = __riscv_vfmv_v_f_f32m1(0.0f, 1);
            vfloat32m1_t reduced = __riscv_vfredusum_vs_f32m1_f32m1(
                vacc,
                zero,
                __riscv_vsetvl_e32m1(1)
            );

            C[i * N + j] = __riscv_vfmv_f_s_f32m1_f32(reduced);
        }
    }
}
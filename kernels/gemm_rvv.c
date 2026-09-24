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

        size_t j = 0;

        while (j < N) {

            size_t vl = __riscv_vsetvl_e32m1(N - j);

            vfloat32m1_t vacc =
                __riscv_vfmv_v_f_f32m1(0.0f, vl);

            for (size_t k = 0; k < K; k++) {

                float a = A[i * K + k];

                const float *b_ptr =
                    &B[k * N + j];

                vfloat32m1_t vb =
                    __riscv_vle32_v_f32m1(
                        b_ptr,
                        vl
                    );

                vacc =
                    __riscv_vfmacc_vf_f32m1(
                        vacc,
                        a,
                        vb,
                        vl
                    );
            }

            __riscv_vse32_v_f32m1(
                &C[i * N + j],
                vacc,
                vl
            );

            j += vl;
        }
    }
}
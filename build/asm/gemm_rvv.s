	.file	"gemm_rvv.c"
	.option pic
	.attribute arch, "rv64i2p1_m2p0_a2p1_f2p2_d2p2_c2p0_v1p0_zicsr2p0_zifencei2p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvl128b1p0_zvl32b1p0_zvl64b1p0"
	.attribute unaligned_access, 0
	.attribute stack_align, 16
	.text
	.align	1
	.globl	gemm_rvv_f32
	.type	gemm_rvv_f32, @function
gemm_rvv_f32:
.LFB2:
	.cfi_startproc
	addi	sp,sp,-32
	.cfi_def_cfa_offset 32
	la	t5,__stack_chk_guard
	sd	s0,16(sp)
	sd	ra,24(sp)
	.cfi_offset 8, -16
	.cfi_offset 1, -8
	mv	s0,a3
	ld	a3, 0(t5)
	sd	a3, 8(sp)
	li	a3, 0
	beq	s0,zero,.L1
	slli	t2,a5,2
	mv	t6,a1
	mv	t4,a2
	mv	a6,a4
	mv	t3,a5
	mv	t1,a0
	add	a2,a0,t2
	slli	a1,a4,2
	li	a7,0
	li	t0,0
.L3:
	li	a0,0
	beq	a6,zero,.L7
.L6:
	sub	a3,a6,a0
	vsetvli	a3,a3,e32,m1,ta,ma
	vmv.v.i	v24,0
	beq	t3,zero,.L4
	slli	a4,a0,2
	add	a4,t6,a4
	mv	a5,t1
.L5:
	flw	fa5,0(a5)
	addi	a5,a5,4
	vle32.v	v25,0(a4)
	add	a4,a4,a1
	vfmacc.vf	v24,fa5,v25
	bne	a5,a2,.L5
.L4:
	add	a5,a0,a7
	slli	a5,a5,2
	add	a5,t4,a5
	add	a0,a0,a3
	vse32.v	v24,0(a5)
	bgtu	a6,a0,.L6
.L7:
	addi	t0,t0,1
	add	a7,a7,a6
	add	t1,t1,t2
	add	a2,a2,t2
	bne	s0,t0,.L3
.L1:
	ld	a4, 8(sp)
	ld	a5, 0(t5)
	xor	a5, a4, a5
	li	a4, 0
	bne	a5,zero,.L21
	ld	ra,24(sp)
	.cfi_remember_state
	.cfi_restore 1
	ld	s0,16(sp)
	.cfi_restore 8
	addi	sp,sp,32
	.cfi_def_cfa_offset 0
	jr	ra
.L21:
	.cfi_restore_state
	call	__stack_chk_fail@plt
	.cfi_endproc
.LFE2:
	.size	gemm_rvv_f32, .-gemm_rvv_f32
	.ident	"GCC: (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0"
	.section	.note.GNU-stack,"",@progbits

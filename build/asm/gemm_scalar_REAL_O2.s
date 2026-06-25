	.file	"gemm_scalar.c"
	.option pic
	.attribute arch, "rv64i2p1_m2p0_a2p1_f2p2_d2p2_c2p0_zicsr2p0_zifencei2p0"
	.attribute unaligned_access, 0
	.attribute stack_align, 16
	.text
	.align	1
	.globl	gemm_scalar_f32
	.type	gemm_scalar_f32, @function
gemm_scalar_f32:
.LFB0:
	.cfi_startproc
	beq	a3,zero,.L18
	addi	sp,sp,-16
	.cfi_def_cfa_offset 16
	slli	t6,a5,2
	sd	s0,8(sp)
	.cfi_offset 8, -8
	mv	t0,a3
	mv	s0,a2
	mv	t2,a1
	mv	t1,a4
	mv	t3,a5
	mv	a7,a0
	add	a3,a0,t6
	slli	a2,a4,2
	li	t5,0
	li	t4,0
.L3:
	beq	t1,zero,.L5
	slli	a1,t5,2
	add	a1,s0,a1
	mv	a6,t2
	li	a0,0
.L7:
	fmv.s.x	fa5,zero
	mv	a4,a6
	mv	a5,a7
	beq	t3,zero,.L6
.L4:
	flw	fa3,0(a5)
	flw	fa4,0(a4)
	addi	a5,a5,4
	add	a4,a4,a2
	fmadd.s	fa5,fa3,fa4,fa5
	bne	a5,a3,.L4
.L6:
	fsw	fa5,0(a1)
	addi	a0,a0,1
	addi	a1,a1,4
	addi	a6,a6,4
	bne	t1,a0,.L7
.L5:
	addi	t4,t4,1
	add	t5,t5,t1
	add	a7,a7,t6
	add	a3,a3,t6
	bne	t0,t4,.L3
	ld	s0,8(sp)
	.cfi_restore 8
	addi	sp,sp,16
	.cfi_def_cfa_offset 0
	jr	ra
.L18:
	ret
	.cfi_endproc
.LFE0:
	.size	gemm_scalar_f32, .-gemm_scalar_f32
	.ident	"GCC: (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0"
	.section	.note.GNU-stack,"",@progbits

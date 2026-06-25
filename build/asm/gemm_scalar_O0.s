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
	addi	sp,sp,-112
	.cfi_def_cfa_offset 112
	sd	s0,104(sp)
	.cfi_offset 8, -8
	addi	s0,sp,112
	.cfi_def_cfa 8, 0
	sd	a0,-72(s0)
	sd	a1,-80(s0)
	sd	a2,-88(s0)
	sd	a3,-96(s0)
	sd	a4,-104(s0)
	sd	a5,-112(s0)
	sd	zero,-40(s0)
	j	.L2
.L7:
	sd	zero,-32(s0)
	j	.L3
.L6:
	fmv.s.x	fa5,zero
	fsw	fa5,-52(s0)
	sd	zero,-24(s0)
	j	.L4
.L5:
	ld	a4,-40(s0)
	ld	a5,-112(s0)
	mul	a4,a4,a5
	ld	a5,-24(s0)
	add	a5,a4,a5
	slli	a5,a5,2
	ld	a4,-72(s0)
	add	a5,a4,a5
	flw	fa5,0(a5)
	fsw	fa5,-48(s0)
	ld	a4,-24(s0)
	ld	a5,-104(s0)
	mul	a4,a4,a5
	ld	a5,-32(s0)
	add	a5,a4,a5
	slli	a5,a5,2
	ld	a4,-80(s0)
	add	a5,a4,a5
	flw	fa5,0(a5)
	fsw	fa5,-44(s0)
	flw	fa4,-48(s0)
	flw	fa5,-44(s0)
	fmul.s	fa5,fa4,fa5
	flw	fa4,-52(s0)
	fadd.s	fa5,fa4,fa5
	fsw	fa5,-52(s0)
	ld	a5,-24(s0)
	addi	a5,a5,1
	sd	a5,-24(s0)
.L4:
	ld	a4,-24(s0)
	ld	a5,-112(s0)
	bltu	a4,a5,.L5
	ld	a4,-40(s0)
	ld	a5,-104(s0)
	mul	a4,a4,a5
	ld	a5,-32(s0)
	add	a5,a4,a5
	slli	a5,a5,2
	ld	a4,-88(s0)
	add	a5,a4,a5
	flw	fa5,-52(s0)
	fsw	fa5,0(a5)
	ld	a5,-32(s0)
	addi	a5,a5,1
	sd	a5,-32(s0)
.L3:
	ld	a4,-32(s0)
	ld	a5,-104(s0)
	bltu	a4,a5,.L6
	ld	a5,-40(s0)
	addi	a5,a5,1
	sd	a5,-40(s0)
.L2:
	ld	a4,-40(s0)
	ld	a5,-96(s0)
	bltu	a4,a5,.L7
	nop
	nop
	ld	s0,104(sp)
	.cfi_restore 8
	.cfi_def_cfa 2, 112
	addi	sp,sp,112
	.cfi_def_cfa_offset 0
	jr	ra
	.cfi_endproc
.LFE0:
	.size	gemm_scalar_f32, .-gemm_scalar_f32
	.ident	"GCC: (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0"
	.section	.note.GNU-stack,"",@progbits

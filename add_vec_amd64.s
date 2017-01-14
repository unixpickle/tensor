//+build amd64,!noasm,!appengine

#include "textflag.h"

// func addVecFast32(v1 []float32, v2 []float32)
TEXT ·addVecFast32(SB), NOSPLIT, $0
  MOVQ v1+0(FP), DX
  MOVQ v1+8(FP), AX
  MOVQ v2+24(FP), CX

loopFast32:
  MOVLPS 0(DX), X0
  MOVHPS 8(DX), X0
  MOVLPS 0(CX), X1
  MOVHPS 8(CX), X1
  ADDPS X1, X0
  MOVLPS X0, 0(DX)
  MOVHPS X0, 8(DX)
  ADDQ $16, CX
  ADDQ $16, DX
  SUBQ $4, AX
  JNZ loopFast32

  RET

// func addVecFast64(v1 []float64, v2 []float64)
TEXT ·addVecFast64(SB), NOSPLIT, $0
  MOVQ v1+0(FP), DX
  MOVQ v1+8(FP), AX
  MOVQ v2+24(FP), CX

loopFast64:
  MOVLPD 0(DX), X0
  MOVHPD 8(DX), X0
  MOVLPD 0(CX), X1
  MOVHPD 8(CX), X1
  ADDPD X1, X0
  MOVLPD X0, 0(DX)
  MOVHPD X0, 8(DX)
  ADDQ $16, CX
  ADDQ $16, DX
  SUBQ $2, AX
  JNZ loopFast64

  RET

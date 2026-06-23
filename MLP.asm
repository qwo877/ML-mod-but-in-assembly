global main
extern printf
extern scanf
;nasm -f win64 MLP.asm -o MLP.obj
;gcc MLP.obj -o MLP.exe
;
;
; 暫存器約定（皆為 Win64 volatile，函式間不需保存）：
;   r10 = x_data 基址,  r11 = y_data 基址
;   rcx = 資料 index i,  r8 = hidden index j
;   rax = 取陣列基址的暫存（RIP-relative 不能帶 index，故先 lea 進 rax）
;
; 記憶體佈局：
;   W1[4] : input  -> hidden     b1[4] : hidden bias
;   W2[4] : hidden -> output     b2    : output bias (scalar)
;   z1[4], a1[4] : forward 暫存（backward 要用）

section .data

x_data:
    dq -3.0, -2.8, -2.6, -2.4, -2.2, -2.0, -1.8, -1.6
    dq -1.4, -1.2, -1.0, -0.8, -0.6, -0.4, -0.2, 0.0
    dq 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6
    dq 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0

y_data:
    dq 9.0, 7.84, 6.76, 5.76, 4.84, 4.0, 3.24, 2.56
    dq 1.96, 1.44, 1.0, 0.64, 0.36, 0.16, 0.04, 0.0
    dq 0.04, 0.16, 0.36, 0.64, 1.0, 1.44, 1.96, 2.56
    dq 3.24, 4.0, 4.84, 5.76, 6.76, 7.84, 9.0

; 超參
lr_const:    dq  0.02                  ; learning_rate
neg2_over_n: dq -0.06451612903225806   ; -2 / 31
inv_n:       dq  0.03225806451612903   ;  1 / 31

; 數值保護
clip_hi:   dq  1.0e6        ; gradient clipping 上界
clip_lo:   dq -1.0e6        ; gradient clipping 下界
cost_huge: dq  1.0e300      ; cost 超過此值視為發散 (Inf)

; 固定初始化
W1: dq -1.0, -1.0,  1.0,  1.0          ; input -> hidden
b1: dq -1.0, -2.0, -1.0, -2.0          ; hidden bias
W2: dq  0.1,  0.1,  0.1,  0.1          ; hidden -> output
b2: dq  0.0                            ; output bias

; 格式字串
fmt_start:
    db "Training MLP(1->4->1, ReLU) to fit y=x^2, N=31", 13, 10, 0
fmt_iter:
    db "Iter %5d | cost = %14.6e", 13, 10, 0
fmt_done:
    db 13, 10, "=== Training Done ===", 13, 10, 0
fmt_w1:
    db "  W1 = [%.6f, %.6f, %.6f, %.6f]", 13, 10, 0
fmt_b1:
    db "  b1 = [%.6f, %.6f, %.6f, %.6f]", 13, 10, 0
fmt_w2:
    db "  W2 = [%.6f, %.6f, %.6f, %.6f]", 13, 10, 0
fmt_b2:
    db "  b2 = %.6f", 13, 10, 0
fmt_bp:
    db "  breakpoints (-b1/W1) = [%.4f, %.4f, %.4f, %.4f]", 13, 10, 0
fmt_cost:
    db "  final cost = %.6e", 13, 10, 0
fmt_nan:
    db "[warn] cost = NaN/Inf at iter %d -> training stopped", 13, 10, 0
fmt_prompt:
    db 13, 10, "Enter x: ", 0
fmt_scan:
    db "%lf", 0
fmt_infer:
    db "pred(x=%.2f) = %.4f   (true x^2 = %.4f)", 13, 10, 0

section .bss

z1:        resq 4      ; forward 暫存
a1:        resq 4
gW1:       resq 4      ; 梯度累加
gb1:       resq 4
gW2:       resq 4
gb2:       resq 1
cost_acc:  resq 1
cur_cost:  resq 1
xi_tmp:    resq 1
ypred:     resq 1
err:       resq 1
delta_out: resq 1
dz1_tmp:   resq 1
bpts:      resq 4      ; breakpoints
x_infer:   resq 1
y_infer:   resq 1
sq_infer:  resq 1

section .text

; compute_gradients_and_cost
;   單趟掃過 N=31 筆，累加梯度與 cost（無 call）
compute_gradients_and_cost:
    lea  r10, [rel x_data]
    lea  r11, [rel y_data]

    ; 梯度與 cost 歸零
    fldz
    xor  r8, r8
.zero_loop:
    cmp  r8, 4
    jge  .zero_done
    lea  rax, [rel gW1]
    fst  qword [rax + r8*8]
    lea  rax, [rel gb1]
    fst  qword [rax + r8*8]
    lea  rax, [rel gW2]
    fst  qword [rax + r8*8]
    inc  r8
    jmp  .zero_loop
.zero_done:
    fst  qword [rel gb2]
    fst  qword [rel cost_acc]
    fstp st0                           ; st0=0

    xor  rcx, rcx                       ; i = 0
.data_loop:
    cmp  rcx, 31
    jge  .data_done

    ; xi = x[i]
    fld  qword [r10 + rcx*8]
    fstp qword [rel xi_tmp]

    ; forward: z1[j]=W1[j]*xi+b1[j]; a1[j]=relu(z1[j])
    xor  r8, r8
.fwd_loop:
    cmp  r8, 4
    jge  .fwd_done
    lea  rax, [rel W1]
    fld  qword [rax + r8*8]
    fmul qword [rel xi_tmp]
    lea  rax, [rel b1]
    fadd qword [rax + r8*8]
    lea  rax, [rel z1]
    fstp qword [rax + r8*8]             ; 存 z1（清空 x87 堆疊）
                                        ; relu: a1 = max(z1, 0)
    movsd xmm0, qword [rax + r8*8]      ; rax 仍為 z1 基址
    xorpd xmm1, xmm1                    ; 0.0
    maxsd xmm0, xmm1
    lea  rax, [rel a1]
    movsd qword [rax + r8*8], xmm0
    inc  r8
    jmp  .fwd_loop
.fwd_done:

    ; y_hat = b2 + sum_j W2[j]*a1[j]
    fld  qword [rel b2]                 ; st0 = acc
    xor  r8, r8
.acc_loop:
    cmp  r8, 4
    jge  .acc_done
    lea  rax, [rel W2]
    fld  qword [rax + r8*8]
    lea  rax, [rel a1]
    fmul qword [rax + r8*8]
    faddp                               ; acc += W2[j]*a1[j]
    inc  r8
    jmp  .acc_loop
.acc_done:
    fstp qword [rel ypred]

    ; err = yi - y_hat ; cost += err^2
    fld  qword [r11 + rcx*8]
    fsub qword [rel ypred]
    fst  qword [rel err]                ; st0 = err
    fmul st0, st0                       ; err^2
    fadd qword [rel cost_acc]
    fstp qword [rel cost_acc]

    ; delta_out = (-2/N) * err
    fld  qword [rel err]
    fmul qword [rel neg2_over_n]
    fstp qword [rel delta_out]

    ; gb2 += delta_out
    fld  qword [rel delta_out]
    fadd qword [rel gb2]
    fstp qword [rel gb2]

    ; backward j-loop
    xor  r8, r8
.bwd_loop:
    cmp  r8, 4
    jge  .bwd_done

    ; gW2[j] += delta_out * a1[j]
    fld  qword [rel delta_out]
    lea  rax, [rel a1]
    fmul qword [rax + r8*8]
    lea  rax, [rel gW2]
    fadd qword [rax + r8*8]
    fstp qword [rax + r8*8]

    ; da1 = delta_out * W2[j]
    fld  qword [rel delta_out]
    lea  rax, [rel W2]
    fmul qword [rax + r8*8]
    fstp qword [rel dz1_tmp]            ; dz1_tmp = da1（落 64-bit，清空 x87）
                                        ; dz1 = (z1>0) ? da1 : 0
    lea  rax, [rel z1]
    movsd xmm0, qword [rax + r8*8]      ; z1
    xorpd xmm1, xmm1                    ; 0.0
    cmpltsd xmm1, xmm0                  ; mask = (0 < z1) ? all-ones : 0
    movsd xmm2, qword [rel dz1_tmp]     ; da1
    andpd xmm2, xmm1                    ; dz1 = da1 & mask
    movsd qword [rel dz1_tmp], xmm2

    ; gW1[j] += dz1 * xi
    fld  qword [rel dz1_tmp]
    fmul qword [rel xi_tmp]
    lea  rax, [rel gW1]
    fadd qword [rax + r8*8]
    fstp qword [rax + r8*8]

    ; gb1[j] += dz1
    fld  qword [rel dz1_tmp]
    lea  rax, [rel gb1]
    fadd qword [rax + r8*8]
    fstp qword [rax + r8*8]

    inc  r8
    jmp  .bwd_loop
.bwd_done:

    inc  rcx
    jmp  .data_loop

.data_done:
    ; cost = cost_acc / N
    fld  qword [rel cost_acc]
    fmul qword [rel inv_n]
    fstp qword [rel cur_cost]
    ret


; update_params : W -= lr*gW ; b -= lr*gb
update_params:
    xor  r8, r8
.up_loop:
    cmp  r8, 4
    jge  .up_done

    fld  qword [rel lr_const]
    lea  rax, [rel gW1]
    fmul qword [rax + r8*8]
    lea  rax, [rel W1]
    fld  qword [rax + r8*8]
    fsubrp                              ; W1 - lr*gW1
    fstp qword [rax + r8*8]

    fld  qword [rel lr_const]
    lea  rax, [rel gb1]
    fmul qword [rax + r8*8]
    lea  rax, [rel b1]
    fld  qword [rax + r8*8]
    fsubrp
    fstp qword [rax + r8*8]

    fld  qword [rel lr_const]
    lea  rax, [rel gW2]
    fmul qword [rax + r8*8]
    lea  rax, [rel W2]
    fld  qword [rax + r8*8]
    fsubrp
    fstp qword [rax + r8*8]

    inc  r8
    jmp  .up_loop
.up_done:
    fld  qword [rel lr_const]
    fmul qword [rel gb2]
    fld  qword [rel b2]
    fsubrp
    fstp qword [rel b2]
    ret


;clip_gradients : 將所有梯度夾在 [clip_lo, clip_hi]（SSE2）
;g = max(min(g, clip_hi), clip_lo)
clip_gradients:
    movsd xmm3, qword [rel clip_hi]
    movsd xmm4, qword [rel clip_lo]
    xor  r8, r8
.cg_loop:
    cmp  r8, 4
    jge  .cg_done
    lea  rax, [rel gW1]
    movsd xmm0, qword [rax + r8*8]
    minsd xmm0, xmm3
    maxsd xmm0, xmm4
    movsd qword [rax + r8*8], xmm0
    lea  rax, [rel gb1]
    movsd xmm0, qword [rax + r8*8]
    minsd xmm0, xmm3
    maxsd xmm0, xmm4
    movsd qword [rax + r8*8], xmm0
    lea  rax, [rel gW2]
    movsd xmm0, qword [rax + r8*8]
    minsd xmm0, xmm3
    maxsd xmm0, xmm4
    movsd qword [rax + r8*8], xmm0
    inc  r8
    jmp  .cg_loop
.cg_done:
    movsd xmm0, qword [rel gb2]
    minsd xmm0, xmm3
    maxsd xmm0, xmm4
    movsd qword [rel gb2], xmm0
    ret


; print4 : printf(rcx=fmt, 4 個 double  [r10+0/8/16/24])
print4:
    sub  rsp, 40                        ; shadow(32)+第5引數槽(8)
    movsd xmm1, qword [r10]
    movq  rdx, xmm1
    movsd xmm2, qword [r10+8]
    movq  r8, xmm2
    movsd xmm3, qword [r10+16]
    movq  r9, xmm3
    movsd xmm0, qword [r10+24]
    movsd qword [rsp+32], xmm0
    call printf
    add  rsp, 40
    ret

; 主函式在這裡 ----------------------------------------------------------------------------------------------------- 主函式在這裡
main:
    push rbp
    mov  rbp, rsp
    push rbx                            ; callee-saved: iter 計數器
    sub  rsp, 40                        ; shadow(32)+第5引數槽(8)

    lea  rcx, [rel fmt_start]
    call printf

    xor  rbx, rbx                       ; iter = 0
.train_loop:
    cmp  rbx, 4000
    jge  .train_done

    call compute_gradients_and_cost

    ; 數值保護
    movsd xmm0, qword [rel cur_cost]
    ucomisd xmm0, xmm0                  ; NaN -> unordered -> PF=1
    jp   .nan_stop
    ucomisd xmm0, qword [rel cost_huge] ; cost > 1e300 -> +Inf / 發散
    ja   .nan_stop

    call clip_gradients                 ; gradient clipping（SSE2）
    call update_params

    ; if iter % 500 == 0 -> print
    mov  rax, rbx
    xor  rdx, rdx
    mov  ecx, 500
    div  ecx                            ; edx = rbx % 500
    test edx, edx
    jnz  .skip_print

    lea  rcx, [rel fmt_iter]
    mov  rdx, rbx                       ; iter
    movsd xmm2, qword [rel cur_cost]
    movq  r8, xmm2                      ; cost（第3引數，double）
    call printf

.skip_print:
    inc  rbx
    jmp  .train_loop

.nan_stop:
    lea  rcx, [rel fmt_nan]
    mov  rdx, rbx                       ; 發散iter
    call printf

.train_done:
    lea  rcx, [rel fmt_done]
    call printf

    lea  rcx, [rel fmt_w1]
    lea  r10, [rel W1]
    call print4

    lea  rcx, [rel fmt_b1]
    lea  r10, [rel b1]
    call print4

    lea  rcx, [rel fmt_w2]
    lea  r10, [rel W2]
    call print4

    lea  rcx, [rel fmt_b2]
    movsd xmm1, qword [rel b2]
    movq  rdx, xmm1
    call printf

    ; breakpoints[j] = -b1[j] / W1[j]
    xor  r8, r8
.bp_loop:
    cmp  r8, 4
    jge  .bp_done
    lea  rax, [rel W1]
    movsd xmm0, qword [rax + r8*8]      ; W1[j]
    xorpd xmm1, xmm1
    ucomisd xmm0, xmm1                  ; W1[j] ? 0
    jp   .bp_zero                       ; W1[j] = NaN
    je   .bp_zero                       ; W1[j] = 0  -> 防除零
    lea  rax, [rel b1]
    fld  qword [rax + r8*8]
    fchs
    lea  rax, [rel W1]
    fdiv qword [rax + r8*8]
    lea  rax, [rel bpts]
    fstp qword [rax + r8*8]
    jmp  .bp_next
.bp_zero:
    xorpd xmm0, xmm0
    lea  rax, [rel bpts]
    movsd qword [rax + r8*8], xmm0      ; 無定義 -> 存 0.0
.bp_next:
    inc  r8
    jmp  .bp_loop
.bp_done:
    lea  rcx, [rel fmt_bp]
    lea  r10, [rel bpts]
    call print4

    ; final cost
    call compute_gradients_and_cost
    lea  rcx, [rel fmt_cost]
    movsd xmm1, qword [rel cur_cost]
    movq  rdx, xmm1
    call printf

    ; 推論
    lea  rcx, [rel fmt_prompt]
    call printf

    lea  rcx, [rel fmt_scan]
    lea  rdx, [rel x_infer]
    call scanf

    ; y_infer = b2 + sum_j relu(W1[j]*x + b1[j]) * W2[j]
    fld  qword [rel b2]
    fstp qword [rel y_infer]
    xor  r8, r8
.inf_loop:
    cmp  r8, 4
    jge  .inf_done
    lea  rax, [rel W1]
    fld  qword [rax + r8*8]
    fmul qword [rel x_infer]
    lea  rax, [rel b1]
    fadd qword [rax + r8*8]             ; st0 = z
    fstp qword [rel ypred]              ; 借 ypred 暫存 z（清空 x87）
    ; relu(z) 與累加改用 SSE2
    movsd xmm0, qword [rel ypred]
    xorpd xmm1, xmm1
    maxsd xmm0, xmm1                    ; relu(z)
    lea  rax, [rel W2]
    mulsd xmm0, qword [rax + r8*8]      ; relu(z) * W2[j]
    addsd xmm0, qword [rel y_infer]
    movsd qword [rel y_infer], xmm0
    inc  r8
    jmp  .inf_loop
.inf_done:

    ; true x^2
    fld  qword [rel x_infer]
    fmul qword [rel x_infer]
    fstp qword [rel sq_infer]

    lea  rcx, [rel fmt_infer]
    movsd xmm1, qword [rel x_infer]
    movq  rdx, xmm1
    movsd xmm2, qword [rel y_infer]
    movq  r8, xmm2
    movsd xmm3, qword [rel sq_infer]
    movq  r9, xmm3
    call printf

    xor  eax, eax
    add  rsp, 40
    pop  rbx
    leave
    ret

global main
extern printf
extern scanf

section .data

x_data:
    dq  3.2,  5.0,  4.7,  3.3,  8.8,  2.7,  1.2,  9.6,  7.1,  5.4
    dq  9.7,  4.5,  1.7,  4.2,  2.5,  9.8,  8.9,  5.4,  4.6,  5.1
    dq  7.5,  3.2,  6.6,  2.3,  2.8,  1.7,  9.6,  1.5,  6.4,  7.5
    dq  9.7,  8.4,  5.7,  4.0,  4.2,  1.7,  6.0,  2.5,  3.7,  8.6

y_data:
    dq  25.0,  44.0,  56.0,  32.0,  85.0,  31.0,   8.0,  94.0,  66.0,  55.0
    dq  97.0,  52.0,  16.0,  39.0,  19.0, 100.0,  85.0,  54.0,  43.0,  58.0
    dq  69.0,  24.0,  66.0,  21.0,  26.0,  21.0,  99.0,  17.0,  73.0,  72.0
    dq 100.0,  81.0,  60.0,  26.0,  39.0,  20.0,  65.0,  24.0,  31.0,  91.0

lr_const:    dq  0.001
neg2_over_n: dq -0.05
inv_n:       dq  0.025

w:  dq -100.0
b:  dq -100.0

fmt_iter:
    db "Iter %5d | w=%10.6f  b=%10.6f  cost=%14.6e", 13, 10, 0
fmt_final:
    db 13, 10, "=== Training Done ===", 13, 10
    db "  w = %.6f", 13, 10
    db "  b = %.6f", 13, 10, 0
fmt_prompt:
    db 13, 10, "Enter study hours: ", 0
fmt_scan:
    db "%lf", 0
fmt_infer:
    db "Predicted score for %.2f hrs = %.2f", 13, 10, 0

section .bss


sum_dw:   resq 1
sum_db:   resq 1
cost_acc: resq 1
ypred:    resq 1
err:      resq 1
cur_cost: resq 1
x_infer:  resq 1
y_infer:  resq 1

section .text

;   無 call 不需要 shadow space / stack frame
;   只用 caller-saved 暫存器
;   以 r10=x基址, r11=y基址, rcx=計數器都是 caller-saved
compute_gradient_and_cost:

    lea  r10, [rel x_data]
    lea  r11, [rel y_data]

    fldz
    fstp qword [rel cost_acc]
    fldz
    fstp qword [rel sum_db]
    fldz
    fstp qword [rel sum_dw]

    xor  rcx, rcx

.loop:
    cmp  rcx, 40
    jge  .epilogue

    ; y_hat i = w·x[i] + b
    fld  qword [rel w]
    fld  qword [r10 + rcx*8]
    fmulp
    fadd qword [rel b]
    fstp qword [rel ypred]

    ; ei = y[i] - y_hat i 
    fld  qword [r11 + rcx*8]
    fsub qword [rel ypred]
    fstp qword [rel err]

    ; cost_acc += ei^2
    fld  qword [rel err]
    fmul qword [rel err]
    fadd qword [rel cost_acc]
    fstp qword [rel cost_acc]

    ; sum_db += ei
    fld  qword [rel err]
    fadd qword [rel sum_db]
    fstp qword [rel sum_db]

    ; sum_dw += x[i]·ei
    fld  qword [r10 + rcx*8]
    fmul qword [rel err]
    fadd qword [rel sum_dw]
    fstp qword [rel sum_dw]

    inc  rcx
    jmp  .loop

.epilogue:
    fld  qword [rel cost_acc]
    fmul qword [rel inv_n]
    fstp qword [rel cur_cost]

    fld  qword [rel sum_dw]
    fmul qword [rel neg2_over_n]
    fstp qword [rel sum_dw]

    fld  qword [rel sum_db]
    fmul qword [rel neg2_over_n]
    fstp qword [rel sum_db]

    ret


update_params:

    fld  qword [rel lr_const]
    fmul qword [rel sum_dw]
    fld  qword [rel w]
    fsubrp
    fstp qword [rel w]

    fld  qword [rel lr_const]
    fmul qword [rel sum_db]
    fld  qword [rel b]
    fsubrp
    fstp qword [rel b]

    ret

main: ;對齊
    push rbp
    mov  rbp, rsp
    push rbx            ; callee-saved: iter 計數器
    sub  rsp, 40        ; shadow(32) + 5th arg slot(8)

    xor  rbx, rbx       ; iter = 0

.train_loop:
    cmp  rbx, 3000
    jge  .train_done

    call compute_gradient_and_cost
    call update_params
    ; div 會破壞 rax/rdx，用 ecx 當除數（caller-saved）
    mov  rax, rbx
    xor  rdx, rdx
    mov  ecx, 500
    div  ecx             ; edx = rbx % 500
    test edx, edx
    jnz  .skip_print

    ;printf(fmt_iter, iter, w, b, cost)
    lea  rcx, [rel fmt_iter]
    mov  rdx, rbx                   ; iter → rdx
    movsd xmm2, qword [rel w]
    movq  r8, xmm2                  ; mirror w → r8
    movsd xmm3, qword [rel b]
    movq  r9, xmm3                  ; mirror b → r9
    movsd xmm0, qword [rel cur_cost]
    movsd qword [rsp+32], xmm0      ; cost → 5th arg slot
    call printf

.skip_print:
    inc  rbx
    jmp  .train_loop

.train_done:
    ;printf(fmt_final, w, b)
    lea  rcx, [rel fmt_final]
    movsd xmm1, qword [rel w]
    movq  rdx, xmm1
    movsd xmm2, qword [rel b]
    movq  r8, xmm2
    call printf

    lea  rcx, [rel fmt_prompt]
    call printf

    ;scanf("%lf", &x_infer) 
    lea  rcx, [rel fmt_scan]
    lea  rdx, [rel x_infer]
    call scanf

    ;y_hat = w·x_infer + b  (x87)
    fld  qword [rel w]
    fmul qword [rel x_infer]
    fadd qword [rel b]
    fstp qword [rel y_infer]

    ;printf(fmt_infer, x_infer, y_infer)
    lea  rcx, [rel fmt_infer]
    movsd xmm1, qword [rel x_infer]
    movq  rdx, xmm1
    movsd xmm2, qword [rel y_infer]
    movq  r8, xmm2
    call printf

    xor  eax, eax
    add  rsp, 40
    pop  rbx
    leave
    ret

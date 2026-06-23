# ============================================================
#  MLP 1 -> 4 -> 1   (ReLU hidden, linear output, MSE loss)
#  擬合 y = x^2 , x in [-3, 3]  (
#
# 
#
#  佈局：
#    W1[4]  : input -> hidden
#    b1[4]  : hidden bias
#    W2[4]  : hidden -> output
#    b2     : output bias (scalar)
#    z1[4], a1[4] : forward 暫存 (backward 要用，需存於 .bss)
#
# ============================================================

DATA_X = [
    -3.0, -2.8, -2.6, -2.4, -2.2, -2.0, -1.8, -1.6,
    -1.4, -1.2, -1.0, -0.8, -0.6, -0.4, -0.2, 0.0,
    0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6,
    1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0,
]
DATA_Y = [
    9.0, 7.84, 6.76, 5.76, 4.84, 4.0, 3.24, 2.56,
    1.96, 1.44, 1.0, 0.64, 0.36, 0.16, 0.04, 0.0,
    0.04, 0.16, 0.36, 0.64, 1.0, 1.44, 1.96, 2.56,
    3.24, 4.0, 4.84, 5.76, 6.76, 7.84, 9.0,
]

x = DATA_X
y = DATA_Y

N = 31   
H = 4    # hidden 神經元數

#   兩個負斜率 ReLU 管左半、兩個正斜率管右半，breakpoint 散佈 [-3,3]
W1 = [-1.0, -1.0, 1.0, 1.0]   # input -> hidden
b1 = [-1.0, -2.0, -1.0, -2.0]
W2 = [0.1, 0.1, 0.1, 0.1]     # hidden -> output
b2 = 0.0


def relu(z):
    if z > 0.0:
        return z
    return 0.0


def relu_grad(z):
    if z > 0.0:
        return 1.0
    return 0.0


def forward(xi):
    # 回傳 z1[4], a1[4], y_hat
    z1 = [0.0, 0.0, 0.0, 0.0]
    a1 = [0.0, 0.0, 0.0, 0.0]
    for j in range(H):
        z1[j] = W1[j] * xi + b1[j]
        a1[j] = relu(z1[j])
    z2 = b2
    for j in range(H):
        z2 = z2 + W2[j] * a1[j]
    y_hat = z2          # linear output
    return z1, a1, y_hat


def compute_gradients_and_cost():
    gW1 = [0.0, 0.0, 0.0, 0.0]
    gb1 = [0.0, 0.0, 0.0, 0.0]
    gW2 = [0.0, 0.0, 0.0, 0.0]
    gb2 = 0.0
    cost = 0.0

    for i in range(N):
        xi = x[i]
        yi = y[i]

        # --- forward ---
        z1, a1, y_hat = forward(xi)

        # --- loss (MSE 累加項) ---
        err = yi - y_hat                 # e = y - y_hat
        cost = cost + err * err

        # --- backward ---
        # output linear: dL/dz2 = dL/dy_hat = -(2/N)*err
        delta_out = (-2.0 / N) * err

        # output 層梯度
        for j in range(H):
            gW2[j] = gW2[j] + delta_out * a1[j]
        gb2 = gb2 + delta_out

        # 反傳到 hidden 層
        for j in range(H):
            da1 = delta_out * W2[j]          # dL/da1[j]
            dz1 = da1 * relu_grad(z1[j])     # 經過 ReLU 閘門
            gW1[j] = gW1[j] + dz1 * xi       # dL/dW1[j]
            gb1[j] = gb1[j] + dz1            # dL/db1[j]

    cost = cost / N                          # MSE 平均
    return gW1, gb1, gW2, gb2, cost


def update_params(gW1, gb1, gW2, gb2, lr):
    global b2
    for j in range(H):
        W1[j] = W1[j] - lr * gW1[j]
        b1[j] = b1[j] - lr * gb1[j]
        W2[j] = W2[j] - lr * gW2[j]
    b2 = b2 - lr * gb2


def gradient_descent(lr, run_iter, p_iter=500):
    for i in range(run_iter):
        gW1, gb1, gW2, gb2, cost = compute_gradients_and_cost()
        update_params(gW1, gb1, gW2, gb2, lr)
        if i % p_iter == 0:
            print(f"Iter {i:5d} | cost = {cost: .6e}")


learning_rate = 0.02
run_iter = 4000

print(f"開始訓練 MLP(1->4->1, ReLU) 擬合 y=x^2，資料筆數：{N}")
gradient_descent(learning_rate, run_iter)

print("-" * 40)
print("=== Training Done ===")
print(f"W1 = {[round(v, 6) for v in W1]}")
print(f"b1 = {[round(v, 6) for v in b1]}")
print(f"W2 = {[round(v, 6) for v in W2]}")
print(f"b2 = {b2:.6f}")
print(f"breakpoints (-b1/W1) = {[round(-b1[j]/W1[j], 4) for j in range(H)]}")

_, _, _, _, final_cost = compute_gradients_and_cost()
print(f"final cost = {final_cost:.6e}")

# ---- 推論 ----
try:
    x_in = float(input("\nEnter x: "))
    _, _, y_out = forward(x_in)
    print(f"pred(x={x_in:.2f}) = {y_out:.4f}   (true x^2 = {x_in*x_in:.4f})")
except (EOFError, ValueError):
    pass
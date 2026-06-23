import numpy as np
import pandas as pd

DATA = pd.DataFrame({
    "Study_Hours": [
        3.2, 5.0, 4.7, 3.3, 8.8, 2.7, 1.2, 9.6, 7.1, 5.4,
        9.7, 4.5, 1.7, 4.2, 2.5, 9.8, 8.9, 5.4, 4.6, 5.1,
        7.5, 3.2, 6.6, 2.3, 2.8, 1.7, 9.6, 1.5, 6.4, 7.5,
        9.7, 8.4, 5.7, 4.0, 4.2, 1.7, 6.0, 2.5, 3.7, 8.6
    ],
    "Scores": [
        25.0, 44.0, 56.0, 32.0, 85.0, 31.0, 8.0, 94.0, 66.0, 55.0,
        97.0, 52.0, 16.0, 39.0, 19.0, 100.0, 85.0, 54.0, 43.0, 58.0,
        69.0, 24.0, 66.0, 21.0, 26.0, 21.0, 99.0, 17.0, 73.0, 72.0,
        100.0, 81.0, 60.0, 26.0, 39.0, 20.0, 65.0, 24.0, 31.0, 91.0
    ]
})

x = DATA["Study_Hours"].to_numpy()
y = DATA["Scores"].to_numpy()

def compute_cost(x, y, w, b):
    y_pred = w * x + b
    cost = ((y - y_pred) ** 2).mean()
    return cost

def compute_gradient(x, y, w, b):
    n = len(x)
    y_pred = w * x + b
    w_gradient = (-2.0 / n) * np.sum(x * (y - y_pred))
    b_gradient = (-2.0 / n) * np.sum(y - y_pred)
    return w_gradient, b_gradient

def gradient_descent(x, y, w_init, b_init, learning_rate, run_iter, p_iter=500):
    w, b = w_init, b_init
    
    for i in range(run_iter):
        w_grad, b_grad = compute_gradient(x, y, w, b)
        w -= learning_rate * w_grad
        b -= learning_rate * b_grad
        
        if i % p_iter == 0:
            cost = compute_cost(x, y, w, b)
            print(f"Iteration {i:5} : Cost {cost: .4e}, w: {w: .4f}, b: {b: .4f}")
            
    return w, b

w_init, b_init = -100.0, -100.0
learning_rate = 0.001
run_iter = 3000

print(f"開始執行梯度下降，總資料筆數：{len(x)}")
w_final, b_final = gradient_descent(x, y, w_init, b_init, learning_rate, run_iter)

print("-" * 30)
print(f"梯度下降最終結果: w = {w_final:.6f}, b = {b_final:.6f}")

pf = np.polyfit(x, y, 1)
print(f"Numpy Polyfit 結果: w = {pf[0]:.6f}, b = {pf[1]:.6f}")

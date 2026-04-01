import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

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


OUT_DIR = "/mnt/data"
os.makedirs(OUT_DIR, exist_ok=True)

x = DATA["Study_Hours"].to_numpy()
y = DATA["Scores"].to_numpy()

print("使用資料筆數：", len(x))
print(DATA.head())

def compute_cost(x, y, w, b):
    y_pred = w * x + b
    loss = (y - y_pred) ** 2
    cost = loss.mean()
    return cost

def compute_gradient(x, y, w, b):
    y_pred = w * x + b
    n = len(x)
    w_gradient = (-2.0 / n) * np.sum(x * (y - y_pred))
    b_gradient = (-2.0 / n) * np.sum(y - y_pred)
    return w_gradient, b_gradient


def gradient_descent(x, y, w_init, b_init, learning_rate, cost_function, gradient_function, run_iter, p_iter=100):
    w_hist = [w_init]
    b_hist = [b_init]
    c_hist = [cost_function(x, y, w_init, b_init)]
    w = w_init
    b = b_init
    for i in range(run_iter):
        w_gradient, b_gradient = gradient_function(x, y, w, b)
        w = w - learning_rate * w_gradient
        b = b - learning_rate * b_gradient
        cost = cost_function(x, y, w, b)
        w_hist.append(w)
        b_hist.append(b)
        c_hist.append(cost)
        if i % p_iter == 0:
            print(f"Iteration {i:5} : Cost {cost: .6e}, w: {w: .6e}, b: {b: .6e}, w_grad: {w_gradient: .6e}, b_grad: {b_gradient: .6e}")
    return w, b, w_hist, b_hist, c_hist
w_init = -100.0
b_init = -100.0
learning_rate = 0.001
run_iter = 3000

ws = np.linspace(-10, 10, 201)
bs = np.linspace(-50, 150, 201)

W_grid, B_grid = np.meshgrid(ws, bs, indexing='xy')
n_w = len(ws)
n_b = len(bs)
costs = np.zeros((n_w, n_b))
for i, wv in enumerate(ws):
    y_pred = wv * x[None, :] + bs[:, None]
    loss = (y[None, :] - y_pred) ** 2
    cost_per_b = loss.mean(axis=1)
    costs[i, :] = cost_per_b

w_final, b_final, w_hist, b_hist, c_hist = gradient_descent(x, y, w_init, b_init, learning_rate, compute_cost, compute_gradient, run_iter, p_iter=500)

print(f"\n最終參數 w={w_final:.6f}, b={b_final:.6f}")
pf = np.polyfit(x, y, 1)
print(f"numpy.polyfit 結果 (slope, intercept) = ({pf[0]:.6f}, {pf[1]:.6f})")

plt.figure(figsize=(6,4))
plt.scatter(x, y, label='data')
sort_idx = np.argsort(x)
x_sorted = x[sort_idx]
y_pred_final = w_final * x + b_final
y_pred_sorted = y_pred_final[sort_idx]
plt.plot(x_sorted, y_pred_sorted, label=f'GD fit: w={w_final:.3f}, b={b_final:.3f}')
plt.xlabel("Study Hours")
plt.ylabel("Scores")
plt.xlim([0, 12])
plt.ylim([0, 100])
plt.title("Data and final fitted line")
plt.legend()
path1 = os.path.join(OUT_DIR, 'data_and_fit.png')
plt.tight_layout()
try:
    plt.show()
except Exception:
    pass

plt.figure(figsize=(6,4))
plt.plot(np.arange(len(c_hist)), c_hist)
plt.title("Iteration vs Cost (全程)")
plt.xlabel("Iteration")
plt.ylabel("Cost (MSE)")
plt.yscale("log")
path2 = os.path.join(OUT_DIR, 'cost_vs_iter.png')
plt.tight_layout()
try:
    plt.show()
except Exception:
    pass

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
W_mesh, B_mesh = np.meshgrid(ws, bs, indexing='ij')
ax.plot_surface(W_mesh, B_mesh, costs.T, alpha=0.5)
ax.plot(w_hist, b_hist, c_hist, color='red')
ax.scatter(w_hist[0], b_hist[0], c_hist[0], s=50)
ax.set_xlabel("w")
ax.set_ylabel("b")
ax.set_zlabel("cost")
ax.set_title("Cost surface and gradient descent path")
path3 = os.path.join(OUT_DIR, 'cost_surface.png')
plt.tight_layout()

try:
    plt.show()
except Exception:
    pass

plt.figure(figsize=(5,3))
plt.plot(np.arange(len(c_hist)), c_hist)
plt.title("Cost (saved image)")
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.yscale("log")
plt.tight_layout()
path4 = os.path.join(OUT_DIR, 'cost_curve.png')

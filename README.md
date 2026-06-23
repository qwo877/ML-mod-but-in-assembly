# ML-mod-but-in-assembly

> Machine learning models —  **x86-64 assembly**. No NumPy, no PyTorch, no math libraries. Just NASM, the x87 FPU, and SSE2.  
I don't know what the hell is wrong with my brain. 

Each model is first prototyped in plain Python, then translated line-by-line into assembly. 

The assembly versions reproduce the Python reference **bit-for-bit** across thousands of gradient-descent iterations.

---

## Structure

| File | Description |
|------|-------------|
| `ML.py`   | Linear regression (gradient descent) — Python reference |
| `ML.asm`  | Linear regression in x86-64 assembly |
| `MLP.py`  | 1→4→1 MLP with ReLU — Python reference |
| `MLP.asm` | The MLP in x86-64 assembly |


---

## The models

### 1 · Linear regression

A single-feature linear model `y = w·x + b` fit by batch gradient descent on a small *study-hours → exam-score* dataset, with the result compared against NumPy's `polyfit` as a sanity check. This was the warm-up: a single weight and bias, just enough to get the x87 floating-point pipeline, the MSE gradient, and the Win64 calling convention working.

### 2 · Multilayer perceptron (1 → 4 → 1)

Disappointingly, it's clear that my abilities are insufficient to complete a full MLP model; strictly speaking, this is a half-finished product.

```
        ┌── ReLU ──┐
  x ──▶─┤── ReLU ──┤──▶ Σ + b₂ ──▶ ŷ      (linear output)
        ├── ReLU ──┤
        └── ReLU ──┘
```

- **Architecture:** 1 input → 4 hidden (ReLU) → 1 output (linear)
- **Loss:** mean squared error
- **Training:** full forward pass + manual backpropagation + gradient descent, all in assembly

**Task: fit `y = x²` on `x ∈ [-3, 3]`** — a genuinely *non-linear* target that a single line cannot represent. The four ReLU units learn to bend at different points and stitch together a piecewise-linear approximation of the parabola:

| Metric | Value |
|--------|-------|
| Final MSE | `1.21e-02` |
| Linear-regression baseline on the same data | `8.17e+00` |
| Improvement | **~675×** |
| Learned breakpoints (`-b/w`) | ≈ ±0.675, ±1.899 |

Because the inputs span negative values, the ReLU gate is genuinely exercised — different neurons switch on and off depending on the input — both during training and at inference. (For example, at `x = 0` every hidden unit is off and the prediction collapses to the output bias.) That is the whole point: the non-linearity is doing real work, not sitting idle.

---

## Build & run

Target platform is **Windows x64** (the code follows the Win64 ABI and links against the C runtime for `printf`/`scanf`).

```bash
# Assemble
nasm -f win64 MLP.asm -o MLP.obj

# Link (MinGW-w64 / GCC)
gcc MLP.obj -o MLP.exe

# Run
MLP.exe
```

Substitute `ML.asm` for the linear-regression model. With the MSVC toolchain, link the object file with `cl` / `link` against the CRT instead of GCC.

---

## Implementation notes

A few things that make writing ML in raw assembly interesting:

- **x87 FPU + SSE2.** Scalar arithmetic (multiply-accumulate, the loss, the gradients) runs on the x87 stack; ReLU and gradient clipping use SSE2. The x87 stack is only 8 slots deep, so every per-sample iteration is kept perfectly balanced back to depth 0 — a single leaked register would crash the run after a handful of samples.

- **Branchless ReLU.** The activation is `maxsd` against zero. Its gradient (1 if `z > 0`, else 0) is computed as a `cmpltsd` mask `AND`-ed onto the incoming gradient — no conditional jumps in the inner loop.

- **Win64 variadic glue.** To print a `double` with `printf`, the value is copied into *both* the integer register and the corresponding XMM register, because the Microsoft CRT reads variadic floating-point arguments from the integer registers. The stack is kept 16-byte aligned across every call.

- **Flat memory layout.** Weights, biases, gradient accumulators and the forward-pass scratch buffers (`z1`, `a1`) live in fixed `.data` / `.bss` slots, indexed manually with `reg*8`.

- **Deterministic.** Fixed initial weights and a fixed iteration count mean the program produces the same numbers on every run — which is how the bit-for-bit match against the Python reference was verified.

## Why do this?
 ~~I don't know.~~   
Actually, this was just meant to be an ML-in-assembly practice at first. Then it became this.

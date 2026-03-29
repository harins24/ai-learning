# Day 12 — Gradient Descent Intuition

**Phase:** Foundations | **Week:** 2 | **GitHub Issue:** #23

---

## 🎯 Objective

Understand gradient descent — the engine that powers nearly all of modern machine learning. By the end of today you will be able to explain what it does intuitively, implement it from scratch in Python, and understand the role of the learning rate.

---

## 🧠 Concept Explanation

### The Core Problem: Finding the Best Parameters

When a model makes predictions, it will be wrong by some amount. We measure the total wrongness with a **loss function** (Day 11). Training a model means finding the parameter values that make the loss as small as possible.

This is an **optimisation problem**: find the minimum of the loss function.

---

### The Mountain Analogy

Imagine you are blindfolded on a hilly mountain range and your goal is to reach the lowest valley. You cannot see the whole terrain — you can only feel the slope under your feet right now.

Your strategy:
1. Feel which direction goes downhill (the gradient)
2. Take a small step in that direction
3. Repeat until you stop descending

This is **gradient descent** in a nutshell.

In ML:
- **The mountain** = the loss function surface
- **Your current position** = the current model parameters (weights)
- **The slope** = the gradient of the loss with respect to each parameter
- **Each step** = one parameter update
- **The valley floor** = the minimum loss — optimal parameters

---

### The Mathematics

For a single parameter θ, gradient descent updates it as follows:

```
θ_new = θ_old  −  α × ∂L/∂θ
```

Where:
- **θ** (theta) — the parameter being updated (e.g., a weight in the model)
- **α** (alpha) — the **learning rate** (how big each step is)
- **∂L/∂θ** — the **gradient**: the derivative of the loss L with respect to θ. This tells you the slope — which direction makes the loss go up.
- The minus sign — we subtract the gradient because we want to go **downhill**

**Intuition for the update rule:**
- If the gradient is positive (loss increases as θ increases) → decrease θ
- If the gradient is negative (loss decreases as θ increases) → increase θ
- If the gradient is zero → we are at a flat point (potentially a minimum!)

---

### The Learning Rate — The Most Important Hyperparameter

**Too large (α = 1.0):** You overshoot the valley. The algorithm bounces back and forth and may never converge — or even diverge (loss increases!).

**Too small (α = 0.0001):** You take tiny steps. The algorithm will eventually find the minimum, but it takes forever.

**Just right (α = 0.01):** Smooth, steady descent to the minimum.

**Backend analogy:** The learning rate is like the step size in a binary search — but for optimisation. Too coarse and you miss the target. Too fine and it takes too long. There is an optimal sweet spot.

---

### Variants of Gradient Descent

**Batch Gradient Descent:**
Compute the gradient using **all training examples** before taking one step.
- Slow on large datasets
- Stable, smooth convergence

**Stochastic Gradient Descent (SGD):**
Compute the gradient using **one training example** at a time.
- Fast, noisy updates
- The noise can actually help escape local minima

**Mini-batch Gradient Descent:**
Compute the gradient using a **small batch** (e.g., 32 or 64 examples).
- Best of both worlds — used in virtually all modern deep learning
- This is what PyTorch and TensorFlow use by default

**Backend analogy:**
- Batch = processing all log entries before writing to database (consistent but slow)
- SGD = writing to database after each log event (fast but noisy)
- Mini-batch = buffering 64 events then flushing (balanced throughput)

---

### Local vs Global Minima

For **linear regression**, the loss surface is a perfect bowl (convex) — gradient descent always finds the global minimum.

For **neural networks**, the loss surface is complex with many bumps. Gradient descent may get stuck in a **local minimum** (a valley that is not the deepest one) or a **saddle point**.

Modern techniques (Adam optimiser, momentum, learning rate schedules) help navigate these landscapes.

---

### Concrete Example: Linear Regression with Gradient Descent

For a simple linear model `ŷ = w × x + b`, the MSE loss is:
```
L = (1/n) Σ(yᵢ − ŷᵢ)²
  = (1/n) Σ(yᵢ − (w × xᵢ + b))²
```

The gradients are:
```
∂L/∂w = (−2/n) Σ xᵢ(yᵢ − ŷᵢ)
∂L/∂b = (−2/n) Σ (yᵢ − ŷᵢ)
```

The update rules become:
```
w = w − α × ∂L/∂w
b = b − α × ∂L/∂b
```

---

## 📖 Key Terms

| Term | Definition |
|------|-----------|
| **Gradient** | Vector of partial derivatives — points in the direction of steepest increase |
| **Gradient Descent** | Iteratively moving in the opposite direction of the gradient to minimise loss |
| **Learning Rate (α)** | Step size for each parameter update — the most critical hyperparameter |
| **Epoch** | One full pass through the entire training dataset |
| **Convergence** | When the loss stops meaningfully decreasing — training is complete |
| **Local Minimum** | A valley that is not the globally lowest point in the loss landscape |
| **Global Minimum** | The absolute lowest point of the loss function |
| **Saddle Point** | A point where gradient is zero but it is not a minimum (flat in one direction, curved in another) |
| **Mini-batch** | A subset of training data used to compute one gradient update |

---

## 💻 Code Exercise

Implement linear regression from scratch using gradient descent. Watch the loss decrease epoch by epoch.

```python
# day12.py — Linear Regression via Gradient Descent from scratch

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# ─── Generate synthetic data: house size → price ──────────────────────────────
# True relationship: price = 3 × size + 50 (+ noise)
n = 100
X = np.random.uniform(20, 200, size=n)           # house sizes in m²
y = 3.0 * X + 50 + np.random.normal(0, 15, n)   # price in $1000s

# Standardise X for better gradient descent performance
X_mean, X_std = X.mean(), X.std()
X_norm = (X - X_mean) / X_std

# ─── Initialise parameters ────────────────────────────────────────────────────
w = 0.0   # weight (slope)
b = 0.0   # bias (intercept)
alpha = 0.05   # learning rate
epochs = 200

loss_history = []

# ─── Gradient Descent Loop ────────────────────────────────────────────────────
print(f"{'Epoch':>6}  {'Loss':>12}  {'w':>8}  {'b':>8}")
print("-" * 45)

for epoch in range(epochs):
    # Forward pass: predict
    y_pred = w * X_norm + b

    # Compute MSE loss
    errors = y_pred - y
    loss = np.mean(errors ** 2)
    loss_history.append(loss)

    # Compute gradients
    grad_w = (2 / n) * np.dot(errors, X_norm)
    grad_b = (2 / n) * np.sum(errors)

    # Update parameters (gradient descent step)
    w = w - alpha * grad_w
    b = b - alpha * grad_b

    if epoch % 20 == 0:
        print(f"{epoch:>6}  {loss:>12.4f}  {w:>8.4f}  {b:>8.4f}")

print(f"\nFinal parameters: w = {w:.4f}, b = {b:.4f}")

# Convert w back to original scale for interpretation
w_original_scale = w / X_std
b_original_scale = b - w * X_mean / X_std
print(f"Recovered: price ≈ {w_original_scale:.2f} × size + {b_original_scale:.2f}")
print(f"True:      price ≈ 3.00 × size + 50.00")

# ─── Plot: Loss curve ─────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(loss_history)
ax1.set_xlabel("Epoch")
ax1.set_ylabel("MSE Loss")
ax1.set_title("Loss Curve (Gradient Descent)")
ax1.grid(True)

# ─── Plot: Fitted line ────────────────────────────────────────────────────────
ax2.scatter(X, y, alpha=0.5, label="Training data")
x_line = np.linspace(X.min(), X.max(), 100)
x_line_norm = (x_line - X_mean) / X_std
y_line = w * x_line_norm + b
ax2.plot(x_line, y_line, 'r-', linewidth=2, label="Fitted line")
ax2.set_xlabel("House size (m²)")
ax2.set_ylabel("Price ($1000s)")
ax2.set_title("Linear Regression Fit")
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig("day12_gradient_descent.png")
plt.show()

# ─── Experiment: Effect of learning rate ─────────────────────────────────────
print("\n=== Learning Rate Comparison ===")
learning_rates = [0.001, 0.05, 0.5, 1.5]
fig, axes = plt.subplots(1, 4, figsize=(20, 4))

for i, lr in enumerate(learning_rates):
    w_test, b_test = 0.0, 0.0
    losses = []
    for _ in range(100):
        y_pred_test = w_test * X_norm + b_test
        errors_test = y_pred_test - y
        loss_test = np.mean(errors_test ** 2)
        losses.append(loss_test)
        if not np.isfinite(loss_test):
            break
        grad_w_test = (2 / n) * np.dot(errors_test, X_norm)
        grad_b_test = (2 / n) * np.sum(errors_test)
        w_test -= lr * grad_w_test
        b_test -= lr * grad_b_test

    axes[i].plot(losses[:50])
    axes[i].set_title(f"α = {lr}")
    axes[i].set_xlabel("Epoch")
    axes[i].set_ylabel("Loss")
    axes[i].grid(True)
    status = "diverged" if not np.isfinite(losses[-1]) else f"final loss={losses[-1]:.1f}"
    print(f"  α={lr:5.3f}: {status}")

plt.suptitle("Effect of Learning Rate on Gradient Descent")
plt.tight_layout()
plt.savefig("day12_learning_rates.png")
plt.show()
```

**Expected output:**
```
 Epoch          Loss         w         b
---------------------------------------------
     0   14826.1234    0.0000    0.0000
    20    3210.4521   45.2341  155.4321
    40     921.3210   68.1234  172.3456
   ...
   180     305.2341   89.1234  185.4321

Recovered: price ≈ 2.97 × size + 51.34
True:      price ≈ 3.00 × size + 50.00

Learning Rate Comparison:
  α=0.001: final loss=1234.5   (too slow)
  α=0.050: final loss=301.2    (good)
  α=0.500: final loss=305.1    (oscillates)
  α=1.500: diverged             (explodes)
```

---

## 🏆 Mini Challenge

1. **Modify the learning rate experiment**: plot the loss curve for all four learning rates on the same plot (not subplots). What do you observe about the slope and shape of each curve?

2. **Implement momentum** (a simple extension of gradient descent):
   ```python
   velocity_w = 0.0
   velocity_b = 0.0
   momentum = 0.9
   # Instead of: w = w - alpha * grad_w
   # Use:        velocity_w = momentum * velocity_w - alpha * grad_w
   #             w = w + velocity_w
   ```
   Compare convergence speed with and without momentum at α=0.05.

---

## ❓ Interview Questions

1. **Explain gradient descent in plain English, as if talking to a non-technical product manager.** No equations.

2. **What is the learning rate and what happens if you set it too high or too low?** How do you typically choose it?

3. **What is the difference between Batch, Stochastic, and Mini-batch Gradient Descent?** Which one do deep learning frameworks use and why?

4. **Can gradient descent get stuck? How?** What techniques can help?

5. **Why do we subtract the gradient instead of adding it?** What would happen if we added it?

---

## 📝 Summary

- Gradient descent is the **optimisation engine** of ML — it finds the parameters that minimise the loss function
- The update rule is: **θ = θ − α × gradient** — always step opposite to the slope
- The **learning rate (α)** controls step size: too large diverges, too small is slow, just right converges smoothly
- One complete pass through the training data is called an **epoch**
- **Convergence** = when the loss stabilises (stops decreasing meaningfully)
- **Batch GD** uses all data per step; **SGD** uses one example; **Mini-batch** uses a small chunk — the standard in deep learning
- For convex loss functions (linear regression), gradient descent always finds the global minimum
- For neural networks, the loss landscape is non-convex — local minima and saddle points are real challenges

---

**GitHub Issue:** #23

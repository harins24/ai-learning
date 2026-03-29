# Day 13 — Bias vs Variance

**Phase:** Foundations | **Week:** 2 | **GitHub Issue:** #24

---

## 🎯 Objective

Understand the bias-variance tradeoff — one of the most fundamental concepts in all of machine learning. Learn to diagnose whether your model is suffering from high bias or high variance, and know what to do about each.

---

## 🧠 Concept Explanation

### The Central Question of ML: Why Does a Model Fail?

When an ML model makes poor predictions, there are only two fundamental reasons:
1. The model is **too simple** and misses the underlying pattern (**high bias**)
2. The model is **too complex** and memorises noise instead of learning the pattern (**high variance**)

Understanding this split tells you exactly what to try next when your model is not performing well.

---

### Bias: Systematic Error from Wrong Assumptions

**Bias** is the error introduced by assuming the data follows a simpler pattern than it actually does.

A **high-bias model** is too rigid. It cannot capture the true complexity of the relationship between features and target. No matter how much data you give it, it will never be accurate.

**Mental model:** Imagine asking a junior developer to implement a payment service by only using `if-else` statements. No matter how many edge cases they add, the model is fundamentally limited by its structure.

**High bias characteristics:**
- High training error AND high test error
- The model underperforms on training data itself
- Also called **underfitting**

**Example:** Using a straight line to model data that follows a curve. The line is always wrong, even on training data.

---

### Variance: Error from Sensitivity to Training Data

**Variance** is the error introduced by the model being too sensitive to the specific training data it was given. It learns the noise and random fluctuations as if they were real patterns.

A **high-variance model** performs well on training data but poorly on any new data. Change the training set slightly and the model changes dramatically.

**Mental model:** A developer who "cargo-cults" every pattern they see in the first codebase they worked on. They memorised that codebase perfectly, but their patterns don't transfer to any other system.

**High variance characteristics:**
- Very low training error, but high test error
- Large gap between training and test performance
- Also called **overfitting**

**Example:** A polynomial of degree 10 fitted to 12 data points. It passes through nearly every training point perfectly, but predicts completely wrong values between them.

---

### The Bias-Variance Tradeoff

The total expected error of a model can be decomposed into three components:

```
Total Error = Bias² + Variance + Irreducible Noise
```

- **Bias²** — error from wrong assumptions (can be reduced by using more complex model)
- **Variance** — error from sensitivity to training data (can be reduced by using simpler model or more data)
- **Irreducible Noise** — inherent randomness in the data, cannot be eliminated

**The tradeoff:** As you increase model complexity:
- Bias decreases (the model gets better at capturing the true pattern)
- Variance increases (the model starts memorising noise)

There is a sweet spot — an **optimal complexity** where the sum of bias and variance is minimised. This is the model you want to deploy.

```
              |
    Error     |   Total Error
              |  /
              | / Variance
              |/______
              /\      \_______
             / \           --___
            /   Bias²            --------
           /
          +---------------------------------→ Model Complexity
          Simple                       Complex
          (underfitting)           (overfitting)
```

---

### Diagnosing Bias vs Variance

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| High training error, high test error | High Bias (underfitting) | Use a more complex model, add features, reduce regularisation |
| Low training error, high test error | High Variance (overfitting) | Get more data, simplify model, add regularisation, use dropout |
| Low training error, low test error | Good fit ✅ | Deploy! |
| High training error, low test error | Bug in your code (rare) | Check data pipeline |

---

### How Model Complexity Relates to Bias/Variance

**Decision Trees:**
- Small tree (max depth=2): high bias, low variance
- Huge tree (unlimited depth): low bias, high variance
- Random Forest averages many high-variance trees to reduce overall variance ← this is why ensembles work!

**Polynomial Regression:**
- Degree 1 (linear): high bias if data is curved
- Degree 10: very low bias but catastrophically high variance
- Degree 3-4: often the sweet spot

**K-Nearest Neighbours:**
- k=1: very high variance (every single training point matters)
- k=n (all points): very high bias (predicts the same value for everyone)
- k=5-20: usually a good balance

---

### Backend Analogy: Service Configuration

Think of your microservice configuration:
- **Too simple config (high bias):** Hard-coded timeout of 30s regardless of endpoint. Quick to set up, always wrong for edge cases.
- **Too complex config (high variance):** Different timeout for every single endpoint based on historical logs. Memorises past patterns but breaks on new workloads.
- **Good config (balanced):** Sensible defaults with overrides for known outliers. Generalises well.

---

## 📖 Key Terms

| Term | Definition |
|------|-----------|
| **Bias** | Error from a model being too simple to capture the true pattern |
| **Variance** | Error from a model being too sensitive to the specific training data |
| **Bias-Variance Tradeoff** | As model complexity increases, bias falls but variance rises |
| **Underfitting** | High bias — model fails even on training data |
| **Overfitting** | High variance — model performs well on training data but poorly on new data |
| **Irreducible Error** | Noise inherent in the data that cannot be eliminated regardless of the model |
| **Regularisation** | Techniques that intentionally constrain model complexity to reduce variance |
| **Generalisation** | How well a model performs on unseen data |

---

## 💻 Code Exercise

Visually demonstrate the bias-variance tradeoff using polynomial regression of increasing degree.

```python
# day13.py — Bias-Variance Tradeoff via Polynomial Regression

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# ─── True function: a sine wave (non-linear) ──────────────────────────────────
def true_function(x):
    return np.sin(2 * np.pi * x)

def generate_data(n=15):
    """Generate noisy training data."""
    x = np.sort(np.random.uniform(0, 1, n))
    y = true_function(x) + np.random.normal(0, 0.2, n)
    return x, y

# ─── Fit polynomial of a given degree ─────────────────────────────────────────
def fit_and_predict(x_train, y_train, x_test, degree):
    coeffs = np.polyfit(x_train, y_train, degree)
    poly   = np.poly1d(coeffs)
    return poly(x_test)

# ─── Visualise underfitting vs good fit vs overfitting ────────────────────────
x_train, y_train = generate_data(n=15)
x_plot = np.linspace(0, 1, 300)
y_true = true_function(x_plot)

degrees = [1, 4, 13]   # underfitting, good fit, overfitting
labels  = ["Degree 1 (Underfitting)", "Degree 4 (Good fit)", "Degree 13 (Overfitting)"]
colours = ["red", "green", "purple"]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, degree, label, colour in zip(axes, degrees, labels, colours):
    y_pred_train = fit_and_predict(x_train, y_train, x_train, degree)
    y_pred_plot  = fit_and_predict(x_train, y_train, x_plot,  degree)

    train_mse = np.mean((y_train - y_pred_train) ** 2)

    ax.scatter(x_train, y_train, color='black', zorder=5, label="Training data")
    ax.plot(x_plot, y_true,      'b--', linewidth=2,               label="True function")
    ax.plot(x_plot, y_pred_plot, colour, linewidth=2,              label=f"Fitted ({label})")
    ax.set_ylim(-2.5, 2.5)
    ax.set_title(f"{label}\nTrain MSE: {train_mse:.4f}")
    ax.legend(fontsize=8)
    ax.grid(True)

plt.suptitle("Bias-Variance Tradeoff: Polynomial Regression", fontsize=14)
plt.tight_layout()
plt.savefig("day13_bias_variance.png")
plt.show()

# ─── Quantify bias and variance across polynomial degrees ─────────────────────
print("\n=== Bias² vs Variance vs Test MSE by Model Complexity ===")
print(f"{'Degree':>6}  {'Train MSE':>10}  {'Test MSE':>10}  {'Diagnosis':>20}")
print("-" * 55)

n_experiments = 50
x_test, y_test = generate_data(n=200)   # large test set

for degree in [1, 2, 3, 4, 6, 8, 12]:
    train_errors, test_errors, predictions_at_test = [], [], []

    for _ in range(n_experiments):
        x_tr, y_tr = generate_data(n=15)

        train_pred = fit_and_predict(x_tr, y_tr, x_tr, degree)
        test_pred  = fit_and_predict(x_tr, y_tr, x_test, degree)

        train_mse = np.mean((y_tr - train_pred) ** 2)
        test_mse  = np.mean((y_test - test_pred) ** 2)

        train_errors.append(train_mse)
        test_errors.append(test_mse)
        predictions_at_test.append(test_pred)

    avg_train = np.mean(train_errors)
    avg_test  = np.mean(test_errors)

    # Diagnosis
    if avg_train > 0.15 and avg_test > 0.15:
        diagnosis = "HIGH BIAS (underfitting)"
    elif avg_train < 0.05 and avg_test > 0.3:
        diagnosis = "HIGH VARIANCE (overfitting)"
    elif avg_train < 0.08 and avg_test < 0.15:
        diagnosis = "GOOD FIT ✅"
    else:
        diagnosis = "moderate"

    print(f"{degree:>6}  {avg_train:>10.4f}  {avg_test:>10.4f}  {diagnosis:>20}")

# ─── Plot train vs test error by degree ───────────────────────────────────────
degrees_list = list(range(1, 14))
train_mses, test_mses = [], []

for degree in degrees_list:
    tr_errs, te_errs = [], []
    for _ in range(50):
        x_tr, y_tr = generate_data(n=15)
        tr_errs.append(np.mean((y_tr - fit_and_predict(x_tr, y_tr, x_tr, degree))**2))
        te_errs.append(np.mean((y_test - fit_and_predict(x_tr, y_tr, x_test, degree))**2))
    train_mses.append(np.mean(tr_errs))
    test_mses.append(np.mean(te_errs))

plt.figure(figsize=(10, 6))
plt.plot(degrees_list, train_mses, 'b-o', label="Train MSE")
plt.plot(degrees_list, test_mses,  'r-o', label="Test MSE")
plt.xlabel("Polynomial Degree (Model Complexity)")
plt.ylabel("Mean Squared Error")
plt.title("Bias-Variance Tradeoff: Train vs Test Error")
plt.legend()
plt.grid(True)
plt.axvline(x=4, color='green', linestyle='--', label="Sweet spot")
plt.savefig("day13_train_vs_test.png")
plt.show()
```

**What to observe:**
- Degree 1: train MSE is high AND test MSE is high → high bias
- Degree 4: train MSE is moderate, test MSE is low → good balance
- Degree 13: train MSE is near-zero but test MSE explodes → high variance
- The train/test error curves cross-over at the "sweet spot" model complexity

---

## 🏆 Mini Challenge

1. **More data cures variance:** Repeat the experiment but with `n=100` training points instead of 15. Does degree-13 still overfit as badly? Why does more data reduce variance?

2. **Diagnose a real scenario:** Your team deployed a fraud detection model. Training accuracy = 99.8%, test accuracy = 72%. What is wrong? What are three things you should try?

3. **Write a function** `diagnose_model(train_error, test_error)` that returns `"high bias"`, `"high variance"`, or `"well-fitted"` based on thresholds you define.

---

## ❓ Interview Questions

1. **What is the bias-variance tradeoff?** Can you draw a diagram showing how total error, bias, and variance change as model complexity increases?

2. **Your model has 95% training accuracy but only 62% test accuracy. What is the problem and what are your top 3 remedies?**

3. **Your model has 65% training accuracy and 63% test accuracy on a dataset where the baseline is 90%. What is the problem?**

4. **Why does a Random Forest have lower variance than a single Decision Tree?** What principle does this exploit?

5. **What is regularisation and how does it address high variance?** Name two common regularisation techniques.

---

## 📝 Summary

- **Bias** = error from model being too simple to capture reality. Manifests as high training error.
- **Variance** = error from model being too sensitive to training data. Manifests as a large gap between training and test error.
- Total error = Bias² + Variance + Irreducible Noise — you can minimise bias + variance but never the irreducible noise
- As model complexity increases, bias falls and variance rises — there is a sweet spot
- **Diagnosing:** both errors high → high bias. Training low, test high → high variance.
- **Cures for high bias:** more complex model, more features, less regularisation
- **Cures for high variance:** more training data, simpler model, regularisation, ensemble methods
- This framework is the lens through which you should evaluate every model you build

---

**GitHub Issue:** #24

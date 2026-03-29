# Day 14 — Overfitting vs Underfitting

**Phase:** Foundations | **Week:** 2 | **GitHub Issue:** #25

---

## 🎯 Objective

Go deeper on overfitting and underfitting — the two failure modes of ML models. Learn the concrete techniques used to prevent overfitting (regularisation, dropout, cross-validation, early stopping) and understand why generalisation is the ultimate goal of ML.

---

## 🧠 Concept Explanation

### Recap: The Two Ways a Model Can Fail

From Day 13, you know the conceptual difference. Today you will operationalise this knowledge into practical tools and techniques.

- **Underfitting (High Bias):** The model is too simple. It performs poorly even on training data because it cannot capture the underlying pattern.
- **Overfitting (High Variance):** The model is too complex. It memorises the training data — including noise — and fails on new data.

The goal of ML is not to minimise training error. The goal is **generalisation**: performing well on data the model has never seen.

---

### Why Overfitting Happens

Consider a model trained on 100 examples. If the model has 200 parameters (weights), it has more parameters than data points — it can literally solve for an exact mapping that passes through every training point. This tells you nothing about the true pattern.

**Signs of overfitting:**
- Training loss is near-zero; validation loss is high and climbing
- Model performance degrades significantly on the test set
- The validation loss starts increasing while training loss continues to decrease (the classic "divergence" curve)

**Signs of underfitting:**
- Both training and validation loss are high
- Adding more training data does not help (the model cannot use it)
- Performance is barely better than a naive baseline

---

### Technique 1: Regularisation

Regularisation adds a **penalty to the loss function** for model complexity. This discourages the model from fitting noise.

**L2 Regularisation (Ridge):**
```
Loss_regularised = MSE + λ × Σwᵢ²
```
- Penalises large weights quadratically
- Encourages weights to be small but non-zero
- The hyperparameter λ (lambda) controls regularisation strength: larger λ = more regularisation = more bias

**L1 Regularisation (Lasso):**
```
Loss_regularised = MSE + λ × Σ|wᵢ|
```
- Penalises large weights linearly
- Drives some weights exactly to zero → automatic feature selection
- Produces sparse models

**Elastic Net:** A combination of L1 and L2.

**Backend analogy:** Regularisation is like a complexity budget for your service. You can add features (weights) but each one costs something. You only keep features that earn more than they cost.

---

### Technique 2: Cross-Validation

Instead of using a single train/test split, **k-fold cross-validation** trains and evaluates the model k times, each time on a different fold of the data.

**Process:**
1. Split data into k equal parts (folds)
2. For each fold i: train on the other k−1 folds, evaluate on fold i
3. Average the k evaluation scores

**Why it is better:** It uses all your data for both training and evaluation. The performance estimate is more reliable and has less variance.

**Standard in ML:** k=5 or k=10 is standard practice.

**Backend analogy:** Instead of testing your service under one traffic pattern, you test it under 5 different traffic patterns and report the average — a much more reliable performance characterisation.

---

### Technique 3: Early Stopping

During training, monitor the **validation loss** after every epoch. When the validation loss stops improving (or starts getting worse), **stop training**.

```
Epoch  Train Loss  Val Loss
  10     0.8500    0.9100
  20     0.5200    0.6800
  30     0.3100    0.5100
  40     0.1800    0.4900  ← best validation
  50     0.0900    0.5300  ← getting worse
  60     0.0400    0.6100  ← overfitting
         ↑ STOP HERE (restore weights from epoch 40)
```

Early stopping is free regularisation — it requires no modification to the model architecture, just monitoring.

---

### Technique 4: Dropout (Neural Networks)

During training, **randomly set a fraction of neurons to zero** on each forward pass. This prevents neurons from co-adapting (learning to rely on specific other neurons) and forces the network to learn redundant representations.

```
Normal forward pass: x → [n1, n2, n3, n4, n5] → output
Dropout (rate=0.5):  x → [n1,  0, n3,  0, n5] → output  (random mask)
```

At inference time, dropout is turned off. All neurons participate, but their weights are scaled to account for the dropout rate used during training.

---

### Technique 5: Data Augmentation

If you have limited data, artificially create more training examples by **transforming existing ones**.

- Images: flip, rotate, crop, add noise
- Text: synonym replacement, back-translation
- Tabular: SMOTE (Synthetic Minority Over-sampling)

More diverse training data reduces overfitting naturally.

---

### Technique 6: More Training Data

The most effective cure for overfitting is simply **more data**. With enough data, even a complex model cannot memorise all of it and is forced to learn the true pattern.

**Rule of thumb:** If your model overfits, try collecting more data before reaching for regularisation. Regularisation is a substitute for data you don't have.

---

### Learning Curves — Your Diagnostic Tool

A **learning curve** plots training and validation error as a function of training set size (or epochs). Reading them tells you exactly what is wrong.

**High Bias (Underfitting):**
```
Error
  |   train ────────────────
  |                          ← large gap is not the problem
  |   val ─────────────────
  |
  +────────────────────────→ Training set size
  (both curves plateau HIGH)
```

**High Variance (Overfitting):**
```
Error
  |   val  \─────────────
  |          ─────────────  ← large gap = problem
  |   train      ──────────
  +────────────────────────→ Training set size
  (large gap between curves)
```

**Good Fit:**
```
Error
  |   val  \────
  |               ───────    ← small gap = healthy
  |   train   ───────────
  +────────────────────────→ Training set size
  (curves converge to low error)
```

---

## 📖 Key Terms

| Term | Definition |
|------|-----------|
| **Overfitting** | Model memorises training data including noise; fails on unseen data |
| **Underfitting** | Model is too simple to capture the true pattern; fails on training data too |
| **Generalisation** | Model performs well on new, unseen data — the ultimate goal |
| **Regularisation** | Penalty on model complexity added to the loss function |
| **L1 (Lasso)** | Regularisation that drives some weights exactly to zero (feature selection) |
| **L2 (Ridge)** | Regularisation that shrinks all weights toward zero (smooth decay) |
| **λ (lambda)** | Regularisation strength hyperparameter — larger = stronger penalty |
| **Cross-Validation** | Evaluating model on multiple folds of the data for robust performance estimation |
| **Early Stopping** | Stop training when validation loss stops improving |
| **Dropout** | Randomly deactivating neurons during training to prevent co-adaptation |
| **Learning Curve** | Plot of training and validation error — your primary diagnostic tool |

---

## 💻 Code Exercise

Demonstrate overfitting and the effect of L2 regularisation and cross-validation from scratch.

```python
# day14.py — Overfitting, Regularisation, Cross-Validation from scratch

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# ─── Part 1: Demonstrate Overfitting with Polynomial Regression ───────────────

def poly_features(x, degree):
    """Create polynomial feature matrix up to given degree."""
    return np.column_stack([x ** d for d in range(degree + 1)])

def ridge_regression(X, y, lam=0.0):
    """Closed-form Ridge Regression: w = (X^T X + λI)^(-1) X^T y"""
    n_features = X.shape[1]
    I = np.eye(n_features)
    I[0, 0] = 0  # Don't regularise the bias term
    return np.linalg.inv(X.T @ X + lam * I) @ X.T @ y

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Generate data: true function sin(2πx) + noise
def true_fn(x):
    return np.sin(2 * np.pi * x)

n_train = 20
x_train = np.sort(np.random.uniform(0, 1, n_train))
y_train = true_fn(x_train) + np.random.normal(0, 0.25, n_train)

x_test = np.linspace(0, 1, 200)
y_test_true = true_fn(x_test)

# Fit models of various degrees
degrees = [1, 3, 9, 15]
fig, axes = plt.subplots(1, 4, figsize=(20, 5))

for ax, deg in zip(axes, degrees):
    X_tr = poly_features(x_train, deg)
    X_te = poly_features(x_test, deg)

    w = ridge_regression(X_tr, y_train, lam=0.0)  # no regularisation

    y_pred_train = X_tr @ w
    y_pred_test  = X_te @ w

    train_mse = mse(y_train, y_pred_train)
    test_mse  = mse(y_test_true, y_pred_test)

    ax.scatter(x_train, y_train, color='black', zorder=5, s=30)
    ax.plot(x_test, y_test_true, 'b--', linewidth=1.5, label="True")
    ax.plot(x_test, np.clip(y_pred_test, -3, 3), 'r-', linewidth=2, label=f"Degree {deg}")
    ax.set_ylim(-2.5, 2.5)
    ax.set_title(f"Degree {deg}\nTrain MSE: {train_mse:.4f}\nTest MSE: {test_mse:.4f}")
    ax.legend(fontsize=8)
    ax.grid(True)

plt.suptitle("Overfitting: Increasing Polynomial Degree (No Regularisation)")
plt.tight_layout()
plt.savefig("day14_overfitting.png")
plt.show()

# ─── Part 2: Effect of L2 Regularisation ──────────────────────────────────────
print("\n=== Effect of L2 Regularisation (Degree 9 polynomial) ===")
print(f"{'Lambda':>10}  {'Train MSE':>12}  {'Test MSE':>12}  {'Status':>20}")
print("-" * 60)

X_tr_9 = poly_features(x_train, 9)
X_te_9 = poly_features(x_test, 9)

lambdas = [0.0, 1e-6, 1e-4, 1e-2, 1.0, 10.0, 100.0]

best_lam, best_test_mse = 0, float('inf')
for lam in lambdas:
    w = ridge_regression(X_tr_9, y_train, lam=lam)
    tr_mse = mse(y_train, X_tr_9 @ w)
    te_mse = mse(y_test_true, X_te_9 @ w)
    if te_mse < best_test_mse:
        best_lam, best_test_mse = lam, te_mse
    status = "overfitting" if tr_mse < 0.05 and te_mse > 0.5 else \
             "underfitting" if tr_mse > 0.2 else "good"
    print(f"{lam:>10.0e}  {tr_mse:>12.4f}  {te_mse:>12.4f}  {status:>20}")

print(f"\nBest lambda: {best_lam} (test MSE: {best_test_mse:.4f})")

# ─── Part 3: Cross-Validation from scratch ────────────────────────────────────
print("\n=== 5-Fold Cross-Validation ===")

def k_fold_cv(X_all, y_all, degree, lam, k=5):
    """Run k-fold cross-validation and return mean CV score."""
    n = len(X_all)
    indices = np.random.permutation(n)
    fold_size = n // k
    cv_scores = []

    for fold in range(k):
        val_idx   = indices[fold * fold_size : (fold + 1) * fold_size]
        train_idx = np.concatenate([indices[:fold * fold_size],
                                    indices[(fold + 1) * fold_size:]])

        X_cv_tr, y_cv_tr = X_all[train_idx], y_all[train_idx]
        X_cv_val, y_cv_val = X_all[val_idx], y_all[val_idx]

        # Fit polynomial features
        x_cv_tr_raw = x_train[train_idx % len(x_train)]  # map back to x for poly
        x_cv_val_raw = x_train[val_idx % len(x_train)]

        Xf_tr  = poly_features(x_cv_tr_raw,  degree)
        Xf_val = poly_features(x_cv_val_raw, degree)

        w = ridge_regression(Xf_tr, y_cv_tr, lam=lam)
        val_mse = mse(y_cv_val, Xf_val @ w)
        cv_scores.append(val_mse)

    return np.mean(cv_scores), np.std(cv_scores)

# Compare degrees with lambda=0.01
print(f"{'Degree':>8}  {'CV Mean MSE':>14}  {'CV Std':>10}")
print("-" * 40)
for deg in [1, 2, 3, 4, 6, 9]:
    mean_cv, std_cv = k_fold_cv(
        poly_features(x_train, deg), y_train,
        degree=deg, lam=0.01
    )
    print(f"{deg:>8}  {mean_cv:>14.4f}  {std_cv:>10.4f}")

# ─── Part 4: Learning Curves ──────────────────────────────────────────────────
print("\n=== Learning Curves ===")
train_sizes = range(5, 20)
lc_train_mse, lc_val_mse = [], []

x_full = np.sort(np.random.uniform(0, 1, 100))
y_full = true_fn(x_full) + np.random.normal(0, 0.25, 100)

degree = 9
lam = 0.01

for size in train_sizes:
    x_s = x_full[:size]
    y_s = y_full[:size]
    Xf_s  = poly_features(x_s, degree)
    Xf_te = poly_features(x_test, degree)

    w = ridge_regression(Xf_s, y_s, lam=lam)
    lc_train_mse.append(mse(y_s, Xf_s @ w))
    lc_val_mse.append(mse(y_test_true, Xf_te @ w))

plt.figure(figsize=(8, 5))
plt.plot(list(train_sizes), lc_train_mse, 'b-o', label="Train MSE")
plt.plot(list(train_sizes), lc_val_mse,   'r-o', label="Validation MSE")
plt.xlabel("Training Set Size")
plt.ylabel("MSE")
plt.title("Learning Curves (Degree-9, λ=0.01)")
plt.legend()
plt.grid(True)
plt.savefig("day14_learning_curves.png")
plt.show()
```

---

## 🏆 Mini Challenge

You are building a customer churn prediction model for your company's SaaS product.

1. Your model achieves 97% training accuracy and 68% test accuracy. **Diagnose the problem and list 4 concrete things you would try.**

2. You apply L2 regularisation with λ=100 and now get 71% training accuracy and 70% test accuracy. **Has this helped, or have you introduced a new problem?**

3. **Implement early stopping** by training a polynomial model for 200 epochs and plotting the training and validation loss. Add logic to stop training and restore the best weights when validation loss has not improved for 10 consecutive epochs.

---

## ❓ Interview Questions

1. **Explain overfitting to a non-technical stakeholder** (e.g., your product manager). What is it and why does it matter for the product?

2. **What is the difference between L1 and L2 regularisation?** When would you prefer L1 over L2?

3. **Walk me through how you would use cross-validation to select the best model.** What is the risk if you tune hyperparameters using the test set?

4. **What is early stopping?** Why is it particularly important for neural networks?

5. **Your validation loss is going down but your training loss has plateaued at a high value. What does this indicate?** (Hint: this is a rare but important case — think carefully.)

---

## 📝 Summary

- **Overfitting** = memorising training data including noise → fails on unseen data (high variance)
- **Underfitting** = too simple to capture the real pattern → fails on training data (high bias)
- **Generalisation** is the true measure of ML model quality — never optimise for training accuracy alone
- **L2 regularisation (Ridge)** adds a squared weight penalty → all weights shrink smoothly
- **L1 regularisation (Lasso)** adds an absolute weight penalty → some weights become exactly zero (sparse models)
- **Cross-validation** provides a robust, unbiased performance estimate by training on multiple data folds
- **Early stopping** halts training when validation performance stops improving → free regularisation
- **Dropout** randomly disables neurons during training → robust, non-redundant feature learning
- **Learning curves** are your primary diagnostic: plot train vs val error over training size or epochs to instantly see whether you have a bias or variance problem

---

**GitHub Issue:** #25

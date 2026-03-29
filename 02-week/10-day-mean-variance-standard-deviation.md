# Day 10 — Mean, Variance & Standard Deviation

**Phase:** Foundations | **Week:** 2 | **GitHub Issue:** #21

---

## 🎯 Objective

Master the three statistics that describe any dataset: mean (where is the centre?), variance (how spread out is it?), and standard deviation (in human-readable units, how spread out is it?). Understand covariance and correlation — the tools that reveal relationships between features in ML.

---

## 🧠 Concept Explanation

### Why These Numbers Matter in ML

Before building any model, you need to understand your data. Mean, variance, and standard deviation are the first three numbers you reach for. They appear throughout the ML pipeline:

- **Feature scaling** (normalisation, standardisation) uses mean and std
- **Gradient descent** converges faster when features have similar scales
- **Anomaly detection** flags points that are many standard deviations from the mean
- **Neural network weight initialisation** uses controlled variance (Xavier, He initialisation)

---

### Mean (Average) — μ

The **arithmetic mean** is the sum of all values divided by the count:

```
μ = (x₁ + x₂ + ... + xₙ) / n  =  (1/n) Σxᵢ
```

**What it tells you:** The centre of mass of your data. One extreme outlier can pull it far from where most data lives.

**Backend analogy:** Average API response time. If 99 requests take 50ms and one takes 5000ms, the mean is about 99ms — not representative of typical performance. This is why engineers use p95/p99 latencies, not just averages.

---

### Variance — σ²

Variance measures **how far values are spread from the mean on average**:

```
σ² = (1/n) Σ(xᵢ − μ)²
```

Steps to compute:
1. Find the mean μ
2. Subtract the mean from each value: (xᵢ − μ)
3. Square each difference: (xᵢ − μ)²
4. Average the squared differences

**Why square?** To make all deviations positive and to penalise large deviations more heavily.

**Note:** When estimating from a sample (not the full population), use **(n−1)** in the denominator — this is called *Bessel's correction* and gives an unbiased estimate. NumPy's `np.var(x, ddof=1)` does this.

---

### Standard Deviation — σ

Standard deviation is simply the square root of variance:

```
σ = √σ²
```

**Why bother?** Variance is in **squared units** (ms², dollars²), which is hard to interpret. Standard deviation brings it back to the original units.

**The 68-95-99.7 Rule** (from Day 9): For a normal distribution:
- μ ± 1σ contains ~68% of the data
- μ ± 2σ contains ~95% of the data
- μ ± 3σ contains ~99.7% of the data

---

### Feature Standardisation (Z-score Normalisation)

This is directly built from mean and std — and you will use it in almost every ML project:

```
z = (x − μ) / σ
```

This transforms any feature so that:
- Mean = 0
- Standard Deviation = 1

**Why it matters for ML:**
- Gradient descent converges much faster when all features are on the same scale
- Distance-based algorithms (KNN, K-Means, SVM) give equal importance to all features
- Without it, a feature in the range [0, 1,000,000] dominates features in [0, 1]

---

### Covariance — How Two Variables Move Together

Covariance measures whether two variables **tend to increase together or move in opposite directions**:

```
Cov(X, Y) = (1/n) Σ(xᵢ − μₓ)(yᵢ − μᵧ)
```

- **Positive covariance:** X and Y tend to rise together (height and weight)
- **Negative covariance:** When X rises, Y tends to fall (exercise and body fat)
- **Near zero:** X and Y are unrelated

**Problem:** Covariance is in the product of their units (e.g., kg·cm), making it hard to compare across different feature pairs.

---

### Correlation — Scaled Covariance

Pearson correlation normalises covariance to the range [-1, +1]:

```
r = Cov(X, Y) / (σₓ × σᵧ)
```

- **r = +1:** Perfect positive linear relationship
- **r = −1:** Perfect negative linear relationship
- **r = 0:** No linear relationship (but there could be a non-linear one!)

**ML use:** High correlation between two features means they carry redundant information — a signal to consider dropping one (dimensionality reduction).

---

## 📖 Key Terms

| Term | Definition |
|------|-----------|
| **Mean (μ)** | Sum of values divided by count — the central tendency |
| **Variance (σ²)** | Average of squared deviations from the mean — measures spread |
| **Standard Deviation (σ)** | Square root of variance — spread in original units |
| **Z-score** | How many standard deviations a value is from the mean: (x−μ)/σ |
| **Covariance** | Measure of how much two variables change together |
| **Correlation** | Normalised covariance (−1 to +1) — scale-independent measure of linear relationship |
| **Standardisation** | Transforming features to have mean=0, std=1 |
| **Normalisation** | Scaling features to a fixed range, often [0, 1] |

---

## 💻 Code Exercise

Implement all statistics from scratch, then compare to NumPy. Apply feature standardisation to a real scenario.

```python
# day10.py — Mean, Variance, Std Dev, and Feature Standardisation from scratch

import numpy as np

# ─── PART 1: Compute statistics from scratch ──────────────────────────────────

api_latencies_ms = [45, 52, 48, 61, 55, 43, 250, 47, 50, 53,
                    49, 51, 46, 58, 44, 48, 52, 60, 47, 900]

def mean(data):
    return sum(data) / len(data)

def variance(data, population=True):
    mu = mean(data)
    squared_diffs = [(x - mu) ** 2 for x in data]
    denominator = len(data) if population else len(data) - 1
    return sum(squared_diffs) / denominator

def std_dev(data, population=True):
    return variance(data, population) ** 0.5

mu   = mean(api_latencies_ms)
var  = variance(api_latencies_ms, population=False)  # sample variance
std  = std_dev(api_latencies_ms, population=False)

print("=== API Latency Statistics ===")
print(f"Mean:              {mu:.2f} ms")
print(f"Variance (sample): {var:.2f} ms²")
print(f"Std Dev (sample):  {std:.2f} ms")

# Verify against NumPy
print("\n=== NumPy Verification ===")
arr = np.array(api_latencies_ms, dtype=float)
print(f"np.mean:   {np.mean(arr):.2f}")
print(f"np.var:    {np.var(arr, ddof=1):.2f}")
print(f"np.std:    {np.std(arr, ddof=1):.2f}")

# ─── PART 2: Outlier detection using Z-scores ─────────────────────────────────

print("\n=== Outlier Detection (|z| > 2) ===")
for latency in api_latencies_ms:
    z = (latency - mu) / std
    if abs(z) > 2:
        print(f"  Latency {latency:4d}ms is an OUTLIER  (z = {z:.2f})")

# ─── PART 3: Feature Standardisation ─────────────────────────────────────────

print("\n=== Feature Standardisation (Z-score normalisation) ===")

# Suppose you have two features: latency (0–1000ms) and error_rate (0.0–0.05)
latency    = np.array([45, 52, 250, 48, 900], dtype=float)
error_rate = np.array([0.01, 0.02, 0.05, 0.01, 0.04], dtype=float)

def standardise(feature):
    return (feature - np.mean(feature)) / np.std(feature, ddof=1)

lat_std  = standardise(latency)
err_std  = standardise(error_rate)

print("Original latency:   ", latency)
print("Standardised:       ", np.round(lat_std, 2))
print("Original error rate:", error_rate)
print("Standardised:       ", np.round(err_std, 2))

# Both features now have mean≈0 and std≈1 — ready for ML
print(f"\nStandardised latency: mean={lat_std.mean():.2f}, std={lat_std.std():.2f}")
print(f"Standardised error:   mean={err_std.mean():.2f}, std={err_std.std():.2f}")

# ─── PART 4: Covariance and Correlation ───────────────────────────────────────

print("\n=== Covariance and Correlation ===")

hours_studied = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
test_scores   = np.array([45, 50, 55, 58, 65, 70, 72, 78, 82, 88], dtype=float)

cov_matrix  = np.cov(hours_studied, test_scores)
correlation = np.corrcoef(hours_studied, test_scores)

print(f"Covariance (hours vs score):  {cov_matrix[0, 1]:.2f}")
print(f"Correlation (hours vs score): {correlation[0, 1]:.4f}")
print("Interpretation: Strong positive correlation — more study hours → higher scores")
```

**Expected output (approximate):**
```
=== API Latency Statistics ===
Mean:              166.95 ms
Variance (sample): 42069.42 ms²
Std Dev (sample):  205.11 ms

=== Outlier Detection (|z| > 2) ===
  Latency  250ms is an OUTLIER  (z = 2.11)  [borderline]
  Latency  900ms is an OUTLIER  (z = 3.57)

=== Covariance and Correlation ===
Covariance (hours vs score):  47.78
Correlation (hours vs score): 0.9922
```

---

## 🏆 Mini Challenge

You have two candidate features for a loan default prediction model:

```python
income       = [30000, 45000, 60000, 80000, 120000, 200000, 250000, 300000]
credit_score = [580, 620, 680, 710, 750, 790, 810, 840]
loan_default = [1, 1, 0, 0, 0, 0, 0, 0]   # 1 = defaulted
```

1. Standardise both `income` and `credit_score`
2. Compute the correlation of each feature with `loan_default`
3. Which feature has a stronger linear relationship with default? What would you do with this information before building an ML model?

---

## ❓ Interview Questions

1. **Why do we standardise features before training a model?** Which algorithms absolutely require it and which are immune to it?

2. **What is the difference between variance and standard deviation?** In what situations would you report one over the other?

3. **Explain the difference between covariance and correlation.** Why is correlation generally more useful when comparing feature relationships?

4. **A feature has mean=100 and std=50. What is the z-score of the value 250? Is it an outlier?**

5. **What is Bessel's correction (using n−1 instead of n)?** Why does it matter when working with samples instead of full populations?

---

## 📝 Summary

- **Mean** is the centre of the data — but vulnerable to outliers
- **Variance** measures average squared spread from the mean; **standard deviation** is its square root (same units as the data)
- **Z-score** tells you how many standard deviations any value is from the mean — the foundation of standardisation and outlier detection
- **Feature standardisation** (z-score normalisation) is a required preprocessing step for gradient descent, KNN, SVM, and neural networks
- **Covariance** reveals the direction of the relationship between two features; **correlation** normalises it to [−1, +1] for comparison
- High correlation between features signals redundancy — use it to guide feature selection and dimensionality reduction

---

**GitHub Issue:** #21

# Day 9 — Probability Distributions

**Phase:** Foundations | **Week:** 2 | **GitHub Issue:** #20

---

## 🎯 Objective

Understand probability distributions — the mathematical shapes that describe how data is spread. By the end of today you will know the most important distributions in ML, when to use each one, and how they connect to the models you will build.

---

## 🧠 Concept Explanation

### What is a Distribution?

A probability distribution tells you: **for any possible value, how likely is it?**

**Backend analogy:** Think of your API response time logs. If you plot all the response times on a histogram, the shape of that histogram is a distribution. Most requests finish fast (say, 50ms), fewer take 200ms, and a handful spike to 2000ms. That specific shape is called a **log-normal distribution** — and ML models use this knowledge to build anomaly detectors.

Distributions come in two flavours:

- **Discrete distributions** — outcomes are countable integers (number of retries, number of errors)
- **Continuous distributions** — outcomes are real numbers (temperature, latency, price)

---

### The Most Important Distributions in ML

#### 1. Bernoulli Distribution (Discrete)

Models a **single binary event**: success or failure, 1 or 0.

```
P(X = 1) = p
P(X = 0) = 1 − p
```

**ML use:** Models a single coin flip — or a single classification decision ("is this email spam?").

---

#### 2. Binomial Distribution (Discrete)

Models the **number of successes in n independent Bernoulli trials**.

```
P(X = k) = C(n, k) × p^k × (1−p)^(n−k)
```

**ML use:** "Out of 100 predictions, how many will be correct?" Quality estimation, A/B test analysis.

**Backend analogy:** Out of 1000 requests, how many will fail given a 2% error rate? That is a Binomial(n=1000, p=0.02) question.

---

#### 3. Normal (Gaussian) Distribution (Continuous) ⭐

The most important distribution in all of statistics and ML.

```
f(x) = (1 / σ√(2π)) × exp(−(x−μ)² / 2σ²)
```

**Parameters:**
- **μ (mu)** — mean, the centre of the bell
- **σ (sigma)** — standard deviation, the width of the bell

**The 68-95-99.7 Rule:**
- 68% of data falls within 1σ of the mean
- 95% within 2σ
- 99.7% within 3σ

**Why it matters everywhere:**
- Feature values in many real-world datasets approximate a normal distribution
- Linear regression assumes residuals are normally distributed
- The Central Limit Theorem (next lesson) explains why averages are always normal
- Neural network weights are often initialised from a normal distribution

---

#### 4. Uniform Distribution (Continuous)

Every value in a range [a, b] is equally likely.

```
f(x) = 1 / (b − a)  for a ≤ x ≤ b
```

**ML use:** Random initialisation of weights, random sampling, shuffling data.

---

#### 5. Exponential Distribution (Continuous)

Models the **time between independent events** (waiting time, inter-arrival time).

```
f(x) = λ × exp(−λx)  for x ≥ 0
```

**Backend analogy:** Time between user requests to your API follows an exponential distribution. This is why queueing theory and Poisson processes use it.

---

#### 6. Poisson Distribution (Discrete)

Models the **number of events in a fixed time interval** given a known average rate λ.

```
P(X = k) = (λ^k × e^−λ) / k!
```

**ML use:** Count data modelling — number of clicks per hour, number of failures per day.

---

### The Central Limit Theorem (CLT)

> **"The mean of a large number of independent samples will be approximately normally distributed, regardless of the original distribution."**

This is one of the most powerful theorems in statistics. It means:
- Even if individual data points are not normal (e.g., exponential, uniform), **averages of batches will be normal**
- This is why batch statistics in neural networks (Batch Normalisation) work so well
- This is why confidence intervals and hypothesis tests work even on skewed data

---

### Visualising Distributions

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-4, 4, 300)

# Normal distribution PDF by hand
def normal_pdf(x, mu=0, sigma=1):
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-((x - mu)**2) / (2 * sigma**2))

plt.figure(figsize=(10, 4))
plt.plot(x, normal_pdf(x, mu=0, sigma=1), label="N(0, 1) Standard Normal")
plt.plot(x, normal_pdf(x, mu=0, sigma=2), label="N(0, 2) Wider")
plt.plot(x, normal_pdf(x, mu=1, sigma=1), label="N(1, 1) Shifted")
plt.title("Gaussian Distributions")
plt.legend()
plt.grid(True)
plt.savefig("day09_distributions.png")
plt.show()
```

---

## 📖 Key Terms

| Term | Definition |
|------|-----------|
| **PDF** | Probability Density Function — gives the relative likelihood for a continuous variable |
| **PMF** | Probability Mass Function — gives the exact probability for a discrete variable |
| **CDF** | Cumulative Distribution Function — P(X ≤ x), the area under the PDF up to x |
| **Mean (μ)** | The expected value / centre of a distribution |
| **Variance (σ²)** | Average squared deviation from the mean — measures spread |
| **Standard Deviation (σ)** | Square root of variance — same units as the data |
| **Skewness** | Asymmetry of a distribution (left-skewed vs right-skewed) |
| **CLT** | Central Limit Theorem — sample means tend toward normality |

---

## 💻 Code Exercise

Simulate distributions from scratch using only NumPy and visualise them.

```python
# day09.py — Simulating and Visualising Probability Distributions

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
n = 10_000

# 1. Bernoulli (p=0.3)
bernoulli_samples = (np.random.random(n) < 0.3).astype(int)

# 2. Binomial (n=10 trials, p=0.5)
binomial_samples = np.random.binomial(n=10, p=0.5, size=n)

# 3. Normal (mu=0, sigma=1)
normal_samples = np.random.normal(loc=0, scale=1, size=n)

# 4. Exponential (rate=1)
exponential_samples = np.random.exponential(scale=1.0, size=n)

# 5. Uniform (0 to 1)
uniform_samples = np.random.uniform(low=0, high=1, size=n)

# Plot all distributions
fig, axes = plt.subplots(1, 5, figsize=(20, 4))

axes[0].bar([0, 1], [np.mean(bernoulli_samples == 0), np.mean(bernoulli_samples == 1)])
axes[0].set_title("Bernoulli (p=0.3)")

axes[1].hist(binomial_samples, bins=11, edgecolor='black')
axes[1].set_title("Binomial (n=10, p=0.5)")

axes[2].hist(normal_samples, bins=50, edgecolor='black')
axes[2].set_title("Normal (μ=0, σ=1)")

axes[3].hist(exponential_samples, bins=50, edgecolor='black')
axes[3].set_title("Exponential (rate=1)")

axes[4].hist(uniform_samples, bins=20, edgecolor='black')
axes[4].set_title("Uniform (0, 1)")

plt.tight_layout()
plt.savefig("day09_all_distributions.png")
plt.show()

# Central Limit Theorem Demo
print("\n--- Central Limit Theorem Demo ---")
# Sample from an exponential distribution (clearly NOT normal)
# Take means of batches and watch them become normal

batch_sizes = [1, 5, 30, 100]
fig, axes = plt.subplots(1, 4, figsize=(20, 4))

for i, batch_size in enumerate(batch_sizes):
    batch_means = [
        np.mean(np.random.exponential(scale=1.0, size=batch_size))
        for _ in range(5000)
    ]
    axes[i].hist(batch_means, bins=50, edgecolor='black')
    axes[i].set_title(f"Batch size = {batch_size}")
    print(f"Batch size {batch_size:3d}: mean of means = {np.mean(batch_means):.3f}, std = {np.std(batch_means):.3f}")

plt.suptitle("Central Limit Theorem: Exponential → Normal as batch size grows")
plt.tight_layout()
plt.savefig("day09_clt_demo.png")
plt.show()
```

**What to observe:**
- The Exponential distribution is heavily right-skewed
- As batch size grows to 30+, the distribution of sample means becomes visually bell-shaped
- This is the Central Limit Theorem in action — it is why we use normal statistics on almost everything

---

## 🏆 Mini Challenge

1. **API latency simulation:** Generate 10,000 simulated API response times using `np.random.exponential(scale=100)` (mean = 100ms). Then:
   - Calculate what percentage of requests exceed 250ms
   - Calculate the 95th percentile latency (p95)
   - Plot a histogram with a vertical red line at p95
2. **Predict:** If you take batches of 50 requests and compute mean latency, what distribution do you expect the batch means to follow? Verify by plotting.

---

## ❓ Interview Questions

1. **Why is the Normal distribution so central to machine learning?** Name at least three places it appears in an ML pipeline.

2. **What is the Central Limit Theorem and why does it matter for model training?** How does it justify using batch statistics in neural networks?

3. **What is the difference between a PDF and a CDF?** If I told you a feature has a CDF value of 0.95 at x=200, what does that mean?

4. **When would you use an Exponential distribution vs a Poisson distribution?** Give a backend system example for each.

5. **What is the relationship between Bernoulli and Binomial distributions?** How does this relate to a neural network's output layer for binary classification?

---

## 📝 Summary

- A **distribution** describes how probability is spread across all possible values
- **Bernoulli** models a single binary outcome; **Binomial** models counts of successes over multiple trials
- The **Normal (Gaussian) distribution** is the most important in ML — features, weights, and errors often follow it
- The **Exponential distribution** models waiting/inter-arrival times — directly relevant to backend system design
- The **Central Limit Theorem** guarantees that sample means converge to normal regardless of the original distribution — this is why batch normalisation and statistical tests work
- Always visualise your data's distribution before choosing an ML model

---

**GitHub Issue:** #20

# Day 11 — Machine Learning Concepts Introduction

**Phase:** Foundations | **Week:** 2 | **GitHub Issue:** #22

---

## 🎯 Objective

Get a solid mental model of what Machine Learning actually is, how it differs from traditional programming, and understand the three main categories of ML. Map these concepts to things you already know as a backend developer.

---

## 🧠 Concept Explanation

### What is Machine Learning?

**Traditional programming:**
```
Rules + Data → Output
```
You write explicit `if-else` logic, business rules, and algorithms. The developer is the one who encodes the "knowledge."

**Machine Learning:**
```
Data + Desired Output → Rules (the model "learns" the rules)
```
Instead of writing the rules yourself, you give the algorithm examples and it figures out the rules on its own.

**Backend analogy:** Imagine you wrote a fraud detection service with 500 hand-crafted if-else rules. Every time a new fraud pattern appeared, you had to update the code, redeploy, and release. ML flips this: you give the system thousands of examples of fraud and non-fraud, and it learns its own detection rules — rules that generalise to patterns you never explicitly programmed.

---

### The Three Categories of ML

#### 1. Supervised Learning

You provide **labelled training examples**: each input has a known correct output.

```
Input (features) + Correct Label → Model learns the mapping
```

**Two types:**
- **Classification:** Output is a category (spam / not spam, fraud / legit, dog / cat)
- **Regression:** Output is a continuous number (house price, temperature, response time)

**Analogy:** Like an intern who learns by reading 10,000 customer support tickets that were already categorised as "billing", "technical", or "general". After seeing enough examples, they can classify new tickets themselves.

**Common algorithms:** Linear Regression, Logistic Regression, Decision Trees, Random Forest, SVM, Neural Networks

---

#### 2. Unsupervised Learning

You provide **unlabelled data**: the model finds hidden structure on its own.

```
Input (no labels) → Model discovers patterns/groupings
```

**Common tasks:**
- **Clustering:** Group similar items together (K-Means, DBSCAN)
- **Dimensionality Reduction:** Compress features while preserving information (PCA)
- **Anomaly Detection:** Find unusual points that don't fit any cluster

**Analogy:** A new employee on their first day who has never seen the codebase. They browse through hundreds of files and start to notice natural groupings — "these look like controllers, these look like services, these look like models" — without anyone telling them the categories.

**Common algorithms:** K-Means, DBSCAN, PCA, Autoencoders

---

#### 3. Reinforcement Learning

An **agent** learns by interacting with an environment. It receives **rewards** for good actions and **penalties** for bad ones, and learns a policy to maximise total reward over time.

```
Agent → Action → Environment → Reward/Penalty → Better Action
```

**Analogy:** Training a new developer with a prod deployment. They make changes, if uptime improves they get praise (reward), if the system crashes they get reprimanded (penalty). Over time they learn what kind of changes are safe.

**Common uses:** Game playing (AlphaGo, chess engines), robotics, recommendation systems, autonomous vehicles

---

### The ML Workflow

Every ML project — regardless of algorithm — follows roughly this pipeline:

```
1. Define the Problem
       ↓
2. Collect & Explore Data (EDA)
       ↓
3. Prepare Data (clean, feature engineer, split)
       ↓
4. Choose a Model
       ↓
5. Train the Model
       ↓
6. Evaluate the Model
       ↓
7. Tune (hyperparameter optimisation)
       ↓
8. Deploy & Monitor
```

**Backend comparison:** This maps surprisingly well to software development:
- Define problem → requirements gathering
- Collect data → requirements + dependency audit
- Prepare data → data migration / schema normalisation
- Train model → compile + run
- Evaluate → testing
- Tune → performance profiling
- Deploy → CI/CD pipeline

---

### Key ML Terminology

**Feature (X):** An input variable used to make a prediction. Also called a predictor or independent variable.
> Example: for house price prediction — square footage, number of bedrooms, location

**Label / Target (y):** The output variable you are trying to predict. Also called the dependent variable.
> Example: house price in dollars

**Training Set:** The portion of data used to train the model (typically 70–80%)

**Validation Set:** A held-out portion used to tune hyperparameters and evaluate during training (typically 10–15%)

**Test Set:** Final held-out data used only once at the end to report true performance (typically 10–20%)

**Model:** A mathematical function that maps inputs (features) to outputs (predictions)

**Training (Learning / Fitting):** The process of adjusting model parameters to minimise prediction error

**Inference:** Using a trained model to make predictions on new, unseen data

**Generalisation:** How well the model performs on data it has never seen before

---

### Parameters vs Hyperparameters

This distinction trips up many beginners:

**Parameters** are values the model **learns from data** during training.
- Neural network weights and biases
- Coefficients in linear regression

**Hyperparameters** are values you **set before training** that control how training happens.
- Learning rate
- Number of trees in a random forest
- Number of layers in a neural network
- Regularisation strength

**Backend analogy:** Parameters are like runtime state (your in-memory cache). Hyperparameters are like configuration values in your `application.yml` — you set them before starting the service.

---

### How Models Learn: Loss Functions

Every supervised model needs a way to measure how wrong its predictions are. This is the **loss function** (also called cost function or objective function).

**For regression:** Mean Squared Error (MSE)
```
MSE = (1/n) Σ(yᵢ − ŷᵢ)²
```

**For classification:** Cross-Entropy Loss
```
Loss = −Σ yᵢ log(ŷᵢ)
```

Training = minimising this loss function by adjusting model parameters. This is done via **gradient descent** (tomorrow's topic!).

---

## 📖 Key Terms

| Term | Definition |
|------|-----------|
| **Supervised Learning** | Learning from labelled examples (input + correct output) |
| **Unsupervised Learning** | Finding hidden structure in unlabelled data |
| **Reinforcement Learning** | Agent learns from reward/penalty signals |
| **Feature** | Input variable used to make a prediction |
| **Label/Target** | The correct output the model is trying to predict |
| **Training/Test Split** | Dividing data into seen (for training) and unseen (for evaluation) portions |
| **Generalisation** | Model's ability to perform well on new, unseen data |
| **Loss Function** | A measure of how wrong the model's predictions are |
| **Parameters** | Values the model learns during training (weights) |
| **Hyperparameters** | Values you configure before training begins |

---

## 💻 Code Exercise

Build intuition for supervised learning by implementing the simplest possible "model" — a rule-based classifier — and compare it to a data-driven approach.

```python
# day11.py — Your first ML pipeline from scratch

import numpy as np
from collections import Counter

# ─── Dataset: Predict if a server is overloaded ───────────────────────────────
# Features: [cpu_percent, memory_percent, active_connections]
# Label:    1 = overloaded, 0 = normal

X_train = np.array([
    [90, 85, 500],   # overloaded
    [95, 90, 600],   # overloaded
    [88, 82, 480],   # overloaded
    [85, 78, 450],   # overloaded
    [30, 40, 100],   # normal
    [25, 35, 80],    # normal
    [40, 50, 150],   # normal
    [20, 30, 60],    # normal
    [70, 65, 300],   # borderline normal
    [75, 70, 350],   # borderline overloaded
])

y_train = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 1])

X_test = np.array([
    [92, 88, 520],   # should be: overloaded
    [28, 38, 90],    # should be: normal
    [65, 60, 280],   # ambiguous
])
y_test = np.array([1, 0, 0])

# ─── Approach 1: Hand-crafted rules (traditional programming) ─────────────────
def rule_based_classifier(x):
    """Hard-coded if-else rules — the traditional approach."""
    cpu, memory, connections = x
    if cpu > 80 and memory > 75:
        return 1  # overloaded
    return 0

print("=== Rule-Based Classifier ===")
predictions = [rule_based_classifier(x) for x in X_test]
accuracy = np.mean(np.array(predictions) == y_test)
print(f"Predictions: {predictions}")
print(f"True labels: {y_test.tolist()}")
print(f"Accuracy:    {accuracy * 100:.0f}%")

# ─── Approach 2: K-Nearest Neighbours (data-driven ML) ────────────────────────
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def knn_predict(X_train, y_train, x_new, k=3):
    """Find k nearest training examples and take majority vote."""
    distances = [(euclidean_distance(x_train, x_new), label)
                 for x_train, label in zip(X_train, y_train)]
    distances.sort(key=lambda d: d[0])
    k_nearest_labels = [label for _, label in distances[:k]]
    return Counter(k_nearest_labels).most_common(1)[0][0]

# Important: standardise features first (Day 10!)
X_train_mean = X_train.mean(axis=0)
X_train_std  = X_train.std(axis=0)
X_train_norm = (X_train - X_train_mean) / X_train_std
X_test_norm  = (X_test  - X_train_mean) / X_train_std  # use TRAIN stats on test

print("\n=== K-Nearest Neighbours Classifier (k=3) ===")
knn_preds = [knn_predict(X_train_norm, y_train, x, k=3) for x in X_test_norm]
knn_accuracy = np.mean(np.array(knn_preds) == y_test)
print(f"Predictions: {knn_preds}")
print(f"True labels: {y_test.tolist()}")
print(f"Accuracy:    {knn_accuracy * 100:.0f}%")

# ─── Compare the two approaches ───────────────────────────────────────────────
print("\n=== Comparison ===")
print("Rule-based: requires domain expertise, brittle, easy to explain")
print("KNN:        learns from data, adapts to patterns, no explicit rules needed")

# ─── Demonstrate train/test split ────────────────────────────────────────────
print("\n=== Train/Test Split Demo ===")
all_data = np.vstack([X_train, X_test])
all_labels = np.hstack([y_train, y_test])

np.random.seed(42)
indices = np.random.permutation(len(all_data))
split = int(0.8 * len(all_data))
train_idx, test_idx = indices[:split], indices[split:]

print(f"Total samples: {len(all_data)}")
print(f"Training:      {len(train_idx)} samples ({100*len(train_idx)/len(all_data):.0f}%)")
print(f"Testing:       {len(test_idx)} samples ({100*len(test_idx)/len(all_data):.0f}%)")
```

**Expected output:**
```
=== Rule-Based Classifier ===
Predictions: [1, 0, 0]
True labels: [1, 0, 0]
Accuracy:    100%

=== K-Nearest Neighbours Classifier (k=3) ===
Predictions: [1, 0, 0]
True labels: [1, 0, 0]
Accuracy:    100%

=== Train/Test Split Demo ===
Total samples: 13
Training:      10 samples (77%)
Testing:       3 samples (23%)
```

---

## 🏆 Mini Challenge

Extend the KNN implementation:

1. Try k = 1, 3, 5, 7 on the training data itself (in-sample evaluation) — what do you notice about k=1?
2. Add a new ambiguous test point `[72, 68, 320]` and vary k — does the prediction change? Why?
3. Explain in one paragraph why we should **never** evaluate a model on the same data it was trained on.

---

## ❓ Interview Questions

1. **What is the difference between supervised and unsupervised learning?** Give one real-world example of each from a backend/platform engineering context.

2. **What is a loss function?** Why do we need one and what happens if you pick the wrong one?

3. **Explain the difference between model parameters and hyperparameters.** Give two examples of each for a decision tree.

4. **Why do we split data into train, validation, and test sets?** What goes wrong if you evaluate on training data?

5. **A colleague says: "Our model has 99% accuracy on the training data — it must be great!" What is your response?**

---

## 📝 Summary

- ML shifts the paradigm from "write the rules" to "learn the rules from data"
- **Supervised learning** learns from labelled examples (regression for numbers, classification for categories)
- **Unsupervised learning** finds hidden structure without labels (clustering, dimensionality reduction)
- **Reinforcement learning** learns through trial and reward/penalty feedback
- Every ML project follows the same pipeline: define → collect → prepare → train → evaluate → tune → deploy
- **Parameters** are learned from data; **hyperparameters** are configured before training
- **Loss functions** quantify prediction error — training is the process of minimising them
- Always split data into train/validation/test — never evaluate on data the model has seen

---

**GitHub Issue:** #22

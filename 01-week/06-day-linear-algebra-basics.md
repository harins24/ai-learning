# Day 6 — Linear Algebra Basics

**Phase:** Foundations | **Week:** 1 | **GitHub Issue:** #17

---

## 🎯 Objective
Understand the linear algebra concepts that underpin every ML algorithm — vectors, matrices, dot products, and matrix multiplication. You don't need to be a mathematician; you need to understand the shapes and operations that ML code performs.

---

## 🧠 Concept Explanation

### Why Linear Algebra for ML?

Every ML model is fundamentally a sequence of matrix operations:
- A dataset of 1000 rows × 10 features = a matrix of shape `(1000, 10)`
- A neural network layer = a matrix multiplication + activation
- A recommendation system = a dot product between user and item vectors

**Backend analogy:** Linear algebra is to ML what SQL is to databases. You don't need to implement a query parser, but you need to understand `JOIN`, `GROUP BY`, and indexes. Same here — you need to understand shapes and products, not derive theorems.

---

### 1. Scalars, Vectors, Matrices, Tensors

```
Scalar  →  a single number           → 5.0
Vector  →  a 1D array of numbers     → [1, 2, 3]         shape: (3,)
Matrix  →  a 2D grid of numbers      → [[1,2],[3,4]]     shape: (2, 2)
Tensor  →  3D+ array                 → shape: (32,28,28) (batch of images)
```

```python
import numpy as np

scalar = 5.0
vector = np.array([1, 2, 3])               # shape (3,)
matrix = np.array([[1, 2], [3, 4]])        # shape (2, 2)
tensor = np.zeros((32, 28, 28))            # shape (32, 28, 28)
```

**ML meaning of each dimension:**
```
(1000, 10)      → 1000 training samples, 10 features each
(512, 256)      → weight matrix of a neural net layer (512 inputs, 256 outputs)
(32, 28, 28, 1) → batch of 32 grayscale images, 28x28 pixels
```

---

### 2. Dot Product — The Core ML Operation

The dot product multiplies element-wise and sums:

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

dot = np.dot(a, b)   # 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
```

**Why it matters:** A linear regression prediction is literally a dot product:
```
prediction = w[0]*x[0] + w[1]*x[1] + ... + w[n]*x[n] + bias
           = np.dot(weights, features) + bias
```

**Geometric meaning:** The dot product measures **similarity** between two vectors. If `a · b` is large, the vectors point in the same direction. This is how recommendation systems work — user vector dotted with item vector gives a similarity score.

---

### 3. Matrix Multiplication

```python
A = np.array([[1, 2],
              [3, 4],
              [5, 6]])   # shape (3, 2)

B = np.array([[7, 8, 9],
              [10, 11, 12]])   # shape (2, 3)

C = A @ B    # shape (3, 3)  — use @ operator in Python
# or: np.matmul(A, B)
# or: np.dot(A, B)  for 2D
```

**Shape rule:** `(m, n) @ (n, p) → (m, p)` — inner dimensions must match.

**ML meaning:**
```
X @ W + b
(100, 10) @ (10, 64) + (64,) → (100, 64)
100 samples, 10 features → 100 samples, 64 hidden units
```
Every neural network forward pass is exactly this.

---

### 4. Transpose

```python
A = np.array([[1, 2, 3],
              [4, 5, 6]])    # shape (2, 3)

A.T          # shape (3, 2) — rows become columns
```

Used constantly to fix shape mismatches:
```python
# If shapes don't align for @, try transposing
X @ W        # (100, 10) @ (10, 64) ✓
X @ W.T      # (100, 10) @ (64, 10).T = (100, 10) @ (10, 64) ✓
```

---

### 5. Norms — Measuring Vector Size

```python
v = np.array([3, 4])

np.linalg.norm(v)           # L2 norm = sqrt(3² + 4²) = 5.0
np.linalg.norm(v, ord=1)    # L1 norm = |3| + |4| = 7
```

**ML uses:**
- L2 norm of weights → L2 regularization (Ridge) — penalizes large weights
- L1 norm of weights → L1 regularization (Lasso) — drives weights to 0 (feature selection)
- Cosine similarity uses norms to compare document embeddings

---

### 6. Key Matrix Types

```python
# Identity matrix — like 1 in multiplication (A @ I = A)
np.eye(3)

# Diagonal matrix
np.diag([2, 5, 8])

# Symmetric matrix — A == A.T (common in covariance matrices)

# Inverse — like dividing by a matrix (A @ A_inv = I)
A = np.array([[2, 1], [1, 3]])
A_inv = np.linalg.inv(A)

# Determinant — tells if matrix is invertible (det != 0)
np.linalg.det(A)
```

---

### 7. Eigenvalues & Eigenvectors (Preview — needed for PCA on Day 24)

```python
A = np.array([[3, 1],
              [0, 2]])

eigenvalues, eigenvectors = np.linalg.eig(A)
# eigenvalues:  [3, 2]
# eigenvectors: directions that don't rotate, only scale
```

**Intuition:** When you multiply a matrix by its eigenvector, the result is just the vector scaled by the eigenvalue. PCA uses this to find the directions of maximum variance in your data.

---

## 🔑 Key Terms

| Term | Meaning |
|---|---|
| **Scalar** | Single number |
| **Vector** | 1D array — represents a point or direction in space |
| **Matrix** | 2D array — represents a linear transformation |
| **Tensor** | n-dimensional array (generalisation of matrix) |
| **Dot product** | Element-wise multiply + sum → measures similarity |
| **Matrix multiplication** | `A @ B` — combines linear transformations |
| **Transpose** | Swap rows and columns |
| **Norm** | Measure of a vector's magnitude/length |
| **Eigenvalue/vector** | Directions a matrix "stretches" — core of PCA |

---

## 💻 Code Exercise

Create `day06.ipynb`:

```python
import numpy as np

# Task 1 — Shapes
# Create the following and print their shapes:
# a) A vector of 5 elements
# b) A 4x3 matrix of ones
# c) A 2x3x4 tensor of zeros
# d) An identity matrix of size 5
```

```python
# Task 2 — Dot product
weights   = np.array([0.5, -0.3, 0.8, 0.2])
features  = np.array([1.2, 0.5, -0.4, 2.0])

# a) Compute the dot product manually (element-wise multiply + sum)
# b) Compute using np.dot()
# c) Verify both match
# d) This simulates a single neuron: add bias=0.1 and compute final prediction
```

```python
# Task 3 — Matrix multiplication (neural network forward pass)
X = np.random.randn(5, 3)   # 5 samples, 3 features
W = np.random.randn(3, 4)   # weight matrix: 3 input → 4 output neurons
b = np.random.randn(4)      # bias for 4 neurons

# a) Compute Z = X @ W + b
# b) What is the shape of Z? What does each dimension mean?
# c) Apply ReLU activation: Z_activated = np.maximum(0, Z)
# d) Print Z and Z_activated — notice negative values become 0
```

```python
# Task 4 — Transpose & shape fixing
A = np.array([[1, 2, 3, 4],
              [5, 6, 7, 8]])   # shape (2, 4)

# a) Compute A.T and print its shape
# b) Compute A @ A.T — what shape do you get?
# c) Compute A.T @ A — what shape do you get?
# d) Why can you compute A @ A.T but not A @ A directly?
```

```python
# Task 5 — Cosine similarity (used in NLP and recommendation systems)
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# User and item vectors (like in a recommendation system)
user_profile  = np.array([0.9, 0.1, 0.8, 0.2])  # loves action, hates romance
movie_action  = np.array([0.95, 0.05, 0.85, 0.1])
movie_romance = np.array([0.1, 0.9, 0.15, 0.85])

# a) Compute similarity between user and action movie
# b) Compute similarity between user and romance movie
# c) Which movie should be recommended?
```

---

## 🔥 Mini Challenge

```python
# Implement linear regression prediction using matrix operations only
# No sklearn, no loops

np.random.seed(42)
n_samples, n_features = 100, 3

X = np.random.randn(n_samples, n_features)  # feature matrix
true_weights = np.array([2.5, -1.0, 0.8])
true_bias = 3.0

# 1. Generate true labels: y = X @ weights + bias + noise
y = X @ true_weights + true_bias + np.random.randn(n_samples) * 0.1

# 2. Add bias column to X (column of ones) to absorb bias into weight vector
X_b = np.hstack([np.ones((n_samples, 1)), X])   # shape becomes (100, 4)

# 3. Use the Normal Equation to find optimal weights:
#    w = (X_b.T @ X_b)^(-1) @ X_b.T @ y
#    Hint: np.linalg.inv() and @
w_hat = ...

# 4. Print the recovered weights — should be close to [3.0, 2.5, -1.0, 0.8]
# 5. Compute predictions and mean squared error
#    MSE = mean((y_pred - y)^2)
```

---

## ❓ Interview Questions

1. **You have a feature matrix X of shape (1000, 20) and a weight matrix W. What shape must W be for the operation X @ W to produce a (1000, 64) output?**
2. **What is the dot product measuring geometrically? How is this used in recommendation systems?**
3. **What is the shape rule for matrix multiplication? Why does `(3,4) @ (5,3)` fail?**
4. **What is the difference between L1 and L2 norms, and how do they affect regularization in ML models?**
5. **You have vectors for 10,000 documents as a matrix of shape (10000, 300). How do you compute all pairwise cosine similarities in one operation? What shape is the result?**

---

## 📝 Summary

- Every ML operation reduces to matrix math: `y = X @ W + b`
- Dot product = element-wise multiply + sum = similarity measure
- Shape rule: `(m, n) @ (n, p) → (m, p)` — commit this to memory
- Transpose flips shape: `(2, 3).T → (3, 2)` — used to fix shape mismatches
- L2 norm measures vector length; used in regularization and cosine similarity
- NumPy `@` operator = matrix multiply; understand shapes before computing

---

## ✅ Done Checklist
- [ ] All 5 tasks completed in `day06.ipynb`
- [ ] Normal equation implemented and weights recovered
- [ ] Can state the matrix multiplication shape rule from memory
- [ ] Understand what each shape dimension means in an ML context
- [ ] Close GitHub issue #17 when done

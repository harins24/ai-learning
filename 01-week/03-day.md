# Day 3 — NumPy Basics

**Phase:** Foundations | **Week:** 1 | **GitHub Issue:** #14

---

## 🎯 Objective
Understand NumPy arrays — the core data structure behind every ML framework. Learn to create, shape, index, and operate on arrays efficiently using vectorized operations.

---

## 🧠 Concept Explanation

### What is NumPy?
NumPy (Numerical Python) provides an `ndarray` — an n-dimensional array that is:
- Stored in contiguous memory (like a C array, not a Java ArrayList)
- Operated on in bulk without Python loops (vectorized)
- The foundation for Pandas, scikit-learn, PyTorch, TensorFlow

**Backend analogy:** Think of a Python `list` as a Java `ArrayList<Object>` — flexible but slow. A NumPy array is like a typed `int[]` stored in off-heap memory — fast, fixed-type, bulk-operable.

### Why not just use Python lists?

```python
# Python list — element-by-element, slow
a = [1, 2, 3, 4, 5]
b = [x * 2 for x in a]   # loop required

# NumPy array — whole array at once, fast
import numpy as np
a = np.array([1, 2, 3, 4, 5])
b = a * 2   # [2, 4, 6, 8, 10] — no loop needed
```

For 1 million elements, NumPy is ~100x faster than a Python loop.

---

### Creating Arrays

```python
import numpy as np

# From list
a = np.array([1, 2, 3, 4, 5])

# Ranges
np.arange(0, 10, 2)        # [0, 2, 4, 6, 8]  — like range() but returns array
np.linspace(0, 1, 5)       # [0.0, 0.25, 0.5, 0.75, 1.0] — evenly spaced floats

# Special arrays
np.zeros((3, 4))           # 3x4 matrix of 0.0
np.ones((2, 3))            # 2x3 matrix of 1.0
np.eye(3)                  # 3x3 identity matrix
np.random.rand(3, 3)       # 3x3 random floats [0,1)
np.random.randint(0, 10, size=(3, 3))  # 3x3 random ints
```

---

### Array Properties

```python
a = np.array([[1, 2, 3],
              [4, 5, 6]])

a.shape     # (2, 3)  — 2 rows, 3 cols
a.ndim      # 2       — number of dimensions
a.size      # 6       — total elements
a.dtype     # int64   — data type
```

**Shape is critical in ML.** An image of 28×28 pixels with 1 colour channel is shape `(28, 28, 1)`. A batch of 32 such images is `(32, 28, 28, 1)`. You will read and debug shapes constantly.

---

### Indexing & Slicing

```python
a = np.array([[10, 20, 30],
              [40, 50, 60],
              [70, 80, 90]])

a[0, 1]       # 20   — row 0, col 1
a[1, :]       # [40, 50, 60]  — entire row 1
a[:, 2]       # [30, 60, 90]  — entire col 2
a[0:2, 1:3]   # [[20,30],[50,60]]  — submatrix

# Boolean indexing — extremely common in ML
scores = np.array([45, 82, 67, 91, 55, 78])
scores[scores > 70]    # [82, 91, 78]  — filter passing scores
scores[scores < 60] = 0  # zero out failing scores
```

---

### Vectorized Operations

```python
a = np.array([1, 2, 3, 4])
b = np.array([10, 20, 30, 40])

a + b        # [11, 22, 33, 44]
a * b        # [10, 40, 90, 160]
a ** 2       # [1, 4, 9, 16]
np.sqrt(a)   # [1.0, 1.414, 1.732, 2.0]

# Aggregate operations
a.sum()      # 10
a.mean()     # 2.5
a.std()      # standard deviation
a.min(), a.max()
np.dot(a, b) # dot product: 1*10 + 2*20 + 3*30 + 4*40 = 300
```

---

### Reshaping

```python
a = np.arange(12)        # [0,1,2,...,11]

a.reshape(3, 4)          # 3x4 matrix
a.reshape(2, 6)          # 2x6 matrix
a.reshape(2, 2, 3)       # 3D: 2 blocks of 2x3

# Flatten back to 1D
matrix = np.array([[1,2,3],[4,5,6]])
matrix.flatten()         # [1, 2, 3, 4, 5, 6]
matrix.reshape(-1)       # same — -1 means "infer this dimension"
```

`reshape(-1, 1)` converts a 1D array into a column vector — you'll use this constantly when feeding data into sklearn models.

---

## 🔑 Key Terms

| Term | Meaning |
|---|---|
| **ndarray** | NumPy's n-dimensional array |
| **shape** | Tuple describing dimensions, e.g. `(100, 28, 28)` |
| **dtype** | Data type of array elements (`float32`, `int64`, etc.) |
| **vectorized** | Operation applied to whole array without Python loop |
| **broadcasting** | NumPy's rule for operating on arrays of different shapes |
| **axis** | The dimension along which an operation is applied (axis=0 → rows, axis=1 → cols) |

---

## 💻 Code Exercise

Create `day03.ipynb`:

```python
import numpy as np

# Task 1 — Array creation
# a) Create array of even numbers from 2 to 20
# b) Create a 4x4 identity matrix
# c) Create a 3x3 matrix of random floats between 0 and 1
# d) Create 6 evenly spaced values between 0 and 5
```

```python
# Task 2 — Shape & indexing
matrix = np.array([
    [1,  2,  3,  4],
    [5,  6,  7,  8],
    [9, 10, 11, 12]
])

# a) Print shape, ndim, size
# b) Get element at row 2, col 3
# c) Get the entire second row
# d) Get the last two columns
# e) Get the 2x2 submatrix from top-right corner
```

```python
# Task 3 — Vectorized ops
prices = np.array([100, 250, 80, 320, 150, 90])

# a) Apply 10% discount to all prices
# b) Find prices above 150
# c) Replace prices below 100 with 100 (price floor)
# d) Compute mean, std, min, max
```

```python
# Task 4 — Reshape (critical for ML)
data = np.arange(24)

# a) Reshape to 6x4
# b) Reshape to 2x3x4 (3D)
# c) Flatten it back to 1D
# d) Reshape to a column vector (24x1)
```

---

## 🔥 Mini Challenge

```python
# Simulate a tiny ML dataset
np.random.seed(42)
X = np.random.randn(100, 5)   # 100 samples, 5 features

# 1. What is the shape of X?
# 2. Compute the mean of each feature (column) — result should be shape (5,)
#    Hint: use axis=0
# 3. Compute the std of each feature
# 4. Normalize X: subtract mean, divide by std (this is called standardization)
#    Result: each feature should have mean≈0 and std≈1
# 5. Verify by printing mean and std of normalized X
```

This normalization step is done in **every** ML preprocessing pipeline.

---

## ❓ Interview Questions

1. **What is the difference between `np.array([1,2,3])` and a Python list `[1,2,3]`? Why does it matter for ML?**
2. **What does `array.reshape(-1, 1)` do, and when would you need it?**
3. **You have a 2D NumPy array of shape `(1000, 10)`. How do you compute the mean of each of the 10 features?**
4. **What is broadcasting in NumPy? Give a simple example.**
5. **Why is `float32` often preferred over `float64` in deep learning?**

---

## 📝 Summary

- NumPy's `ndarray` is the building block of all ML frameworks
- `shape`, `dtype`, `ndim` describe an array — always check these when debugging
- Slicing works like Python lists but extends to multiple dimensions: `a[row, col]`
- Vectorized operations eliminate loops — `a * 2` multiplies every element
- `reshape` transforms array dimensions — mastering it prevents shape errors in models
- Boolean indexing (`array[array > 0]`) is how you filter data without loops

---

## ✅ Done Checklist
- [ ] All 4 tasks completed in `day03.ipynb`
- [ ] Mini challenge: data normalized and verified
- [ ] Can explain shape `(32, 28, 28, 1)` in terms of a batch of images
- [ ] Close GitHub issue #14 when done

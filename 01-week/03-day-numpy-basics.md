# Day 3 — NumPy Basics

**Phase:** Foundations | **Week:** 1 | **GitHub Issue:** #14

---

## Objective
Understand NumPy arrays — the core data structure behind every ML framework. Learn to create, shape, index, and operate on arrays efficiently using vectorized operations.

---

## Concept Explanation

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

**Shape is critical in ML.** An image of 28x28 pixels with 1 colour channel is shape `(28, 28, 1)`. A batch of 32 such images is `(32, 28, 28, 1)`. You will read and debug shapes constantly.

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

## Key Terms

| Term | Meaning |
|---|---|
| **ndarray** | NumPy's n-dimensional array |
| **shape** | Tuple describing dimensions, e.g. `(100, 28, 28)` |
| **dtype** | Data type of array elements (`float32`, `int64`, etc.) |
| **vectorized** | Operation applied to whole array without Python loop |
| **broadcasting** | NumPy's rule for operating on arrays of different shapes |
| **axis** | The dimension along which an operation is applied (axis=0 = rows, axis=1 = cols) |

---

## Code Exercise

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

## Mini Challenge

```python
# Simulate a tiny ML dataset
np.random.seed(42)
X = np.random.randn(100, 5)   # 100 samples, 5 features

# 1. What is the shape of X?
# 2. Compute the mean of each feature (column) — result should be shape (5,)
#    Hint: use axis=0
# 3. Compute the std of each feature
# 4. Normalize X: subtract mean, divide by std (this is called standardization)
#    Result: each feature should have mean~0 and std~1
# 5. Verify by printing mean and std of normalized X
```

This normalization step is done in **every** ML preprocessing pipeline.

---

## Interview Questions

1. **What is the difference between `np.array([1,2,3])` and a Python list `[1,2,3]`? Why does it matter for ML?**
2. **What does `array.reshape(-1, 1)` do, and when would you need it?**
3. **You have a 2D NumPy array of shape `(1000, 10)`. How do you compute the mean of each of the 10 features?**
4. **What is broadcasting in NumPy? Give a simple example.**
5. **Why is `float32` often preferred over `float64` in deep learning?**

---

## Summary

- NumPy's `ndarray` is the building block of all ML frameworks
- `shape`, `dtype`, `ndim` describe an array — always check these when debugging
- Slicing works like Python lists but extends to multiple dimensions: `a[row, col]`
- Vectorized operations eliminate loops — `a * 2` multiplies every element
- `reshape` transforms array dimensions — mastering it prevents shape errors in models
- Boolean indexing (`array[array > 0]`) is how you filter data without loops

---

## Done Checklist
- [ ] All 4 tasks completed in `day03.ipynb`
- [ ] Mini challenge: data normalized and verified
- [ ] Can explain shape `(32, 28, 28, 1)` in terms of a batch of images
- [ ] Close GitHub issue #14 when done

---

---

# Deep Dive: Complete Reference Guide

This section is a full, exhaustive breakdown of every concept, line of code, and idea above. Think of it as your reference manual + teaching guide combined.

---

## Table of Contents

1. [What is NumPy — The Full Story](#1-what-is-numpy)
2. [Memory Model — How NumPy Arrays Actually Work](#2-memory-model)
3. [Why Not Python Lists — A Deep Comparison](#3-why-not-python-lists)
4. [Creating Arrays — Every Method Explained](#4-creating-arrays)
5. [Array Properties — Shape, dtype, ndim, size](#5-array-properties)
6. [Indexing and Slicing — Every Form](#6-indexing-and-slicing)
7. [Boolean Indexing — The ML Workhorse](#7-boolean-indexing)
8. [Vectorized Operations — How and Why They're Fast](#8-vectorized-operations)
9. [Aggregate Operations](#9-aggregate-operations)
10. [Broadcasting — NumPy's Most Powerful Rule](#10-broadcasting)
11. [Reshaping — The Shape-Shifting Superpower](#11-reshaping)
12. [Code Exercise — Line-by-Line Walkthrough](#12-code-exercise-walkthrough)
13. [Mini Challenge — Full Solution + Explanation](#13-mini-challenge-solution)
14. [Interview Questions — Complete Answers](#14-interview-questions)
15. [Performance Internals — Why NumPy is Fast](#15-performance-internals)
16. [NumPy in the ML Pipeline — The Big Picture](#16-numpy-in-ml)
17. [Common Mistakes and How to Avoid Them](#17-common-mistakes)
18. [Advanced NumPy Concepts to Know About](#18-advanced-concepts)
19. [Summary and Connections](#19-summary)

---

## 1. What is NumPy

### 1.1 The Name and Origin

**NumPy** stands for **Numerical Python**. It was created by Travis Oliphant in 2005 by merging two earlier projects — **Numeric** and **Numarray**. The entire scientific Python ecosystem (Pandas, scikit-learn, SciPy, Matplotlib, TensorFlow, PyTorch) is built on top of NumPy as the foundational layer.

When people say "Python is slow," they mean *pure Python*. When people say "Python is fast enough for ML," they mean *NumPy-backed Python*. The difference is enormous.

### 1.2 What Problem NumPy Solves

In scientific computing and ML, you constantly need to:

- Store thousands or millions of numbers
- Apply the same mathematical operation to every number
- Combine datasets through matrix multiplication
- Reshape data into different dimensional forms

Python's built-in types are not designed for this. NumPy provides the `ndarray` — an n-dimensional array — that solves all of these problems with C-level speed.

### 1.3 NumPy's Place in the Ecosystem

```
Your Python Code
      |
    NumPy (ndarray, vectorized ops)
      |
    BLAS/LAPACK (highly optimized C/Fortran math libraries)
      |
    CPU hardware (cache-optimized SIMD instructions)
```

**BLAS** = Basic Linear Algebra Subprograms. These are battle-tested C and Fortran routines, decades old, incredibly optimized for matrix operations. NumPy calls these under the hood.

**SIMD** = Single Instruction, Multiple Data. A modern CPU can add 8 numbers simultaneously using special vector registers (AVX2 on most modern CPUs). NumPy triggers these automatically.

### 1.4 Java Analogy — The Full Picture

| Concept | Java Equivalent | NumPy Equivalent |
|---|---|---|
| Python `list` | `ArrayList<Object>` | — |
| NumPy `ndarray` | `int[]` / `double[]` | `np.array` |
| Off-heap memory | `DirectByteBuffer` | NumPy buffer |
| Vectorized op | SIMD via JVM JIT | BLAS routines |
| Shape | Array dimensions in Java | `.shape` tuple |
| dtype | Generic type `T` in arrays | `.dtype` |

In Java, when you use `int[]`, the JVM stores numbers contiguously in memory as raw bytes. No object header per element. NumPy does the exact same thing in Python.

---

## 2. Memory Model

### 2.1 Python List Memory Layout

When you write `a = [1, 2, 3, 4, 5]` in Python:

```
Python list object:
  - Header (refcount, type pointer, size)
  - Array of POINTERS to Python int objects

Pointer[0] ──► PyIntObject(1)  [header + value]
Pointer[1] ──► PyIntObject(2)  [header + value]
Pointer[2] ──► PyIntObject(3)  [header + value]
...
```

Each Python integer is a full object. A Python integer takes **28 bytes** on a 64-bit system. A pointer to it takes **8 bytes**. So a list of 5 integers actually consumes roughly `5 * (28 + 8) = 180 bytes`.

Worse: these objects are scattered all over heap memory. To iterate, the CPU must follow each pointer to a random memory location, causing **cache misses** — the most expensive operation a CPU can do (100–300 CPU cycles per miss).

### 2.2 NumPy Array Memory Layout

When you write `a = np.array([1, 2, 3, 4, 5])`:

```
NumPy ndarray:
  - Metadata block (shape, strides, dtype, pointer to data buffer)
  - Contiguous data buffer:
    [1][2][3][4][5]  <- raw bytes, no headers, packed tightly
```

Each integer is stored as a raw 8-byte (`int64`) value. The total data is `5 * 8 = 40 bytes`. No pointers, no headers.

More importantly: all values are stored **next to each other** in memory. When the CPU loads one number, it automatically prefetches the neighboring ones into its L1/L2 cache. Iterating becomes blazing fast — near zero cache misses.

### 2.3 Strides — The Secret of ndarray

Every NumPy array has a `strides` attribute alongside `shape`. Strides tell NumPy how many bytes to jump to reach the next element in each dimension.

```python
import numpy as np

a = np.array([[1, 2, 3],
              [4, 5, 6]])

print(a.strides)   # (24, 8) — 24 bytes to jump to next row, 8 bytes to next column
print(a.shape)     # (2, 3)
```

With `dtype=int64` (8 bytes per element):
- Moving to the next **column**: jump 8 bytes (one element)
- Moving to the next **row**: jump `3 * 8 = 24` bytes (three elements)

This is why reshaping is essentially "free" in NumPy — it just changes the strides and shape metadata without moving any data.

### 2.4 C-Order vs Fortran-Order

By default, NumPy stores arrays in **C-order** (row-major): elements in the same row are adjacent in memory.

```
C-order (default):   [1, 2, 3, 4, 5, 6]  <- row 0 then row 1
Fortran-order:       [1, 4, 2, 5, 3, 6]  <- col 0 then col 1
```

For iterating row by row, C-order is cache-friendly. For column operations, Fortran-order would be better. Most ML frameworks default to C-order and deal with column operations through BLAS routines that are optimized either way.

---

## 3. Why Not Python Lists

### 3.1 Speed Benchmark

```python
import numpy as np
import time

n = 1_000_000

# Python list approach
a = list(range(n))
start = time.time()
b = [x * 2 for x in a]
python_time = time.time() - start

# NumPy approach
c = np.arange(n)
start = time.time()
d = c * 2
numpy_time = time.time() - start

print(f"Python list: {python_time:.4f}s")
print(f"NumPy:       {numpy_time:.4f}s")
print(f"NumPy is {python_time / numpy_time:.0f}x faster")
```

Typical output:
```
Python list: 0.0832s
NumPy:       0.0008s
NumPy is 104x faster
```

### 3.2 Feature Comparison

| Feature | Python List | NumPy Array |
|---|---|---|
| Element type | Mixed (any object) | Fixed (all same dtype) |
| Memory | Non-contiguous, objects | Contiguous raw bytes |
| Arithmetic | Requires explicit loop | Element-wise by default |
| Mathematical functions | Need `math` module per element | Built-in, whole-array |
| Multi-dimensional | List of lists (ugly) | Native n-dimensional |
| Size fixed? | No (dynamic) | Yes (fixed after creation) |
| Broadcasting | No | Yes |
| Slicing | Returns copy | Returns view (default) |

### 3.3 The List-of-Lists Problem

In Python, a 2D array is a list of lists:

```python
matrix = [[1, 2, 3],
          [4, 5, 6]]

# To get element at row 1, col 2:
matrix[1][2]   # 6 — double indexing required

# To multiply all elements by 2:
result = [[x * 2 for x in row] for row in matrix]  # nested loop!
```

With NumPy:
```python
matrix = np.array([[1, 2, 3],
                   [4, 5, 6]])

matrix[1, 2]   # 6 — single indexing with tuple
matrix * 2     # whole array multiplied — one operation
```

The nested loop in Python is not just ugly — it's slow because it calls the Python bytecode interpreter once per element.

### 3.4 Type Coercion

NumPy enforces a single dtype per array. This is a feature:

```python
# If you mix types, NumPy upcasts to fit all values
a = np.array([1, 2.5, 3])   # dtype becomes float64
print(a)        # [1.  2.5  3. ]
print(a.dtype)  # float64

# You can explicitly specify dtype
b = np.array([1, 2, 3], dtype=np.float32)
c = np.array([1, 2, 3], dtype=np.int16)
```

In ML, `float32` is almost always used instead of `float64` because:
- GPUs are optimized for 32-bit floats
- Takes half the memory (GPUs have limited VRAM)
- Precision difference is negligible for training

---

## 4. Creating Arrays — Every Method Explained

### 4.1 `np.array()` — From Existing Data

```python
import numpy as np

# From a Python list
a = np.array([1, 2, 3, 4, 5])
print(a)        # [1 2 3 4 5]
print(a.dtype)  # int64

# From a list of lists (creates 2D array)
b = np.array([[1, 2, 3],
              [4, 5, 6]])
print(b.shape)  # (2, 3)

# From a tuple
c = np.array((10, 20, 30))

# With explicit dtype
d = np.array([1, 2, 3], dtype=np.float32)
print(d)        # [1. 2. 3.]
print(d.dtype)  # float32
```

**What happens internally:** `np.array()` allocates a contiguous block of memory, determines the dtype (or uses the one you specified), and copies each value from the Python list into the buffer as raw bytes.

### 4.2 `np.arange()` — Numeric Ranges

```python
# np.arange(stop)
np.arange(5)           # [0 1 2 3 4]  — note: stop is EXCLUSIVE

# np.arange(start, stop)
np.arange(2, 8)        # [2 3 4 5 6 7]

# np.arange(start, stop, step)
np.arange(0, 10, 2)    # [0 2 4 6 8]   — even numbers
np.arange(10, 0, -1)   # [10 9 8 7 6 5 4 3 2 1]  — reversed
np.arange(0, 1, 0.1)   # [0.  0.1  0.2  ...  0.9]  — float step

# Java analogy:
# IntStream.range(0, 10).toArray()  ~=  np.arange(10)
# IntStream.iterate(0, x -> x + 2).limit(5).toArray()  ~=  np.arange(0, 10, 2)
```

**Important:** `np.arange` with float steps can have floating-point precision issues. Prefer `np.linspace` when you need exact endpoint control with floats.

### 4.3 `np.linspace()` — Evenly Spaced Values

```python
# np.linspace(start, stop, num)
# NOTE: stop IS inclusive by default (unlike arange)

np.linspace(0, 1, 5)       # [0.   0.25  0.5   0.75  1.  ]
np.linspace(0, 10, 11)     # [0. 1. 2. ... 10.]
np.linspace(0, 2*np.pi, 100)  # 100 points along a full circle

# endpoint=False makes it behave like arange
np.linspace(0, 1, 5, endpoint=False)   # [0.  0.2  0.4  0.6  0.8]
```

**When to use linspace vs arange:**
- `arange`: when you know the **step size**
- `linspace`: when you know the **number of points** you want

**ML use case:** When plotting a learned decision boundary, you use `linspace` to create a fine grid of x-values to evaluate a function on.

### 4.4 `np.zeros()` — Array of Zeros

```python
np.zeros(5)          # [0. 0. 0. 0. 0.]  — 1D, 5 elements, float64
np.zeros((3, 4))     # 3x4 matrix of 0.0
np.zeros((2, 3, 4))  # 3D array: shape (2, 3, 4)

# With specific dtype
np.zeros((3, 3), dtype=int)           # integer zeros
np.zeros((3, 3), dtype=np.float32)    # 32-bit float zeros
```

**Why zeros?** You often initialize weight matrices or bias vectors to zero before training. You might also initialize an output array to zero and fill it incrementally.

### 4.5 `np.ones()` — Array of Ones

```python
np.ones(5)           # [1. 1. 1. 1. 1.]
np.ones((2, 3))      # 2x3 matrix of 1.0
np.ones_like(a)      # same shape and dtype as array `a`, filled with 1s
np.zeros_like(a)     # same shape and dtype as array `a`, filled with 0s
```

`ones_like` and `zeros_like` are very useful when you want to create an array matching another array's shape, which is common in loss calculations.

### 4.6 `np.eye()` — Identity Matrix

```python
np.eye(3)
# [[1. 0. 0.]
#  [0. 1. 0.]
#  [0. 0. 1.]]

np.eye(3, 4)   # Non-square identity-like matrix (3 rows, 4 cols)
# [[1. 0. 0. 0.]
#  [0. 1. 0. 0.]
#  [0. 0. 1. 0.]]

np.eye(4, k=1)   # k shifts the diagonal off-center
# [[0. 1. 0. 0.]
#  [0. 0. 1. 0.]
#  [0. 0. 0. 1.]
#  [0. 0. 0. 0.]]
```

**Why identity matrices matter in ML:**
- Matrix multiplication by identity = original matrix (like multiplying by 1)
- Used in regularization (L2 adds `lambda * I` to a matrix)
- Used in attention mechanisms in Transformers as initialization

### 4.7 `np.random` — Random Arrays

```python
# Uniform distribution [0.0, 1.0)
np.random.rand(3, 3)          # 3x3 random floats

# Standard normal distribution (mean=0, std=1)
np.random.randn(100, 5)       # 100 samples, 5 features — Gaussian noise

# Random integers
np.random.randint(0, 10, size=(3, 3))   # ints in [0, 10)
np.random.randint(0, 2, size=100)       # binary labels: 0 or 1

# Reproducibility — CRITICAL in ML!
np.random.seed(42)            # set seed before any random ops
np.random.rand(3)             # always produces same output when seed=42

# Modern preferred way (NumPy 1.17+)
rng = np.random.default_rng(42)
rng.normal(0, 1, size=(100, 5))   # 100x5 from N(0,1)
```

**Why `seed(42)` appears everywhere:** Random seeds make experiments reproducible. If two researchers use the same seed, they get the same random numbers, so they can compare results fairly. 42 is just a convention (from *The Hitchhiker's Guide to the Galaxy*).

### 4.8 Other Creation Functions

```python
np.full((3, 3), 7)        # 3x3 array filled with 7
np.full((2, 4), np.pi)    # 2x4 filled with pi

np.empty((3, 3))          # 3x3 UNINITIALIZED array — contains garbage values
                          # Faster than zeros because it skips the zeroing step
                          # Use only when you're about to fill every element

np.diag([1, 2, 3])        # [[1,0,0],[0,2,0],[0,0,3]] — diagonal matrix
np.diag(matrix)           # Extract diagonal elements from a 2D array

np.tile(a, (2, 3))        # Tile array a in a 2x3 grid pattern
np.repeat(a, 3)           # Repeat each element 3 times
```

---

## 5. Array Properties

### 5.1 Shape

```python
a = np.array([[1, 2, 3],
              [4, 5, 6]])

print(a.shape)   # (2, 3)
```

`shape` returns a **tuple** where each element represents the size of one dimension.

- `(2, 3)` means: 2 rows, 3 columns
- `(100,)` means: 1D array with 100 elements (notice the trailing comma — it's a 1-tuple)
- `(32, 28, 28, 3)` means: 4D array — batch of 32 images, each 28x28 pixels with 3 color channels

**Shape is the most important thing to check when debugging ML code.** Shape mismatches are the #1 cause of errors in ML code.

```python
# 1D array
a = np.array([1, 2, 3])
print(a.shape)    # (3,)  — tuple with one element

# 2D array (matrix)
b = np.zeros((4, 5))
print(b.shape)    # (4, 5)

# 3D array (e.g., an image with color channels)
c = np.zeros((28, 28, 3))
print(c.shape)    # (28, 28, 3)

# 4D array (e.g., a batch of images)
d = np.zeros((32, 28, 28, 3))
print(d.shape)    # (32, 28, 28, 3)
```

Understanding the shape `(32, 28, 28, 1)` from the lesson:

```
(32, 28, 28, 1)
  |   |   |  +-- 1 color channel (grayscale)
  |   |   +───── 28 pixels wide
  |   +───────── 28 pixels tall
  +───────────── 32 images in this batch
```

For comparison, `(32, 28, 28, 3)` would be a batch of 32 RGB images (3 channels: red, green, blue).

### 5.2 ndim — Number of Dimensions

```python
a = np.array([1, 2, 3])
print(a.ndim)    # 1

b = np.array([[1, 2], [3, 4]])
print(b.ndim)    # 2

c = np.zeros((2, 3, 4))
print(c.ndim)    # 3
```

A quick shortcut: `a.ndim == len(a.shape)` is always True.

In ML terminology:
- **1D array** = vector (e.g., one data sample with features)
- **2D array** = matrix (e.g., dataset with samples x features)
- **3D array** = tensor (e.g., sequence data: time_steps x batch x features)
- **4D array** = tensor (e.g., image batch: batch x height x width x channels)

### 5.3 size — Total Number of Elements

```python
a = np.zeros((3, 4, 5))
print(a.size)     # 60  (= 3 * 4 * 5)
print(a.shape)    # (3, 4, 5)

# Shortcut: a.size == np.prod(a.shape)
```

### 5.4 dtype — Data Type

```python
a = np.array([1, 2, 3])
print(a.dtype)    # int64  (on most systems)

b = np.array([1.0, 2.0])
print(b.dtype)    # float64

c = np.array([True, False, True])
print(c.dtype)    # bool
```

**Memory sizes:**

| dtype | Bytes per element | Use case |
|---|---|---|
| `bool` | 1 | Masks, flags |
| `int8` | 1 | Quantized models |
| `uint8` | 1 | Image pixels (0-255) |
| `int16` | 2 | Compact integer data |
| `int32` | 4 | General integers |
| `int64` | 8 | Default integer |
| `float16` | 2 | GPU training (half-precision) |
| `float32` | 4 | GPU training (standard) |
| `float64` | 8 | Default float, high-precision |

**Casting dtypes:**

```python
a = np.array([1, 2, 3])    # int64
b = a.astype(np.float32)   # cast to float32
c = a.astype(np.uint8)     # cast to uint8

# Careful with overflow!
d = np.array([200, 300], dtype=np.uint8)
print(d)    # [200, 44]  — 300 overflows uint8 (max 255), wraps to 44
```

### 5.5 itemsize and nbytes

```python
a = np.zeros((100, 100), dtype=np.float64)
print(a.itemsize)   # 8  — bytes per element
print(a.nbytes)     # 80000  — total bytes (100 * 100 * 8)

b = np.zeros((100, 100), dtype=np.float32)
print(b.nbytes)     # 40000  — half the memory

# For GPU ML: a 1000x1000 float32 matrix = 4MB
# float64 would be 8MB — twice the GPU VRAM consumption
```

---

## 6. Indexing and Slicing

### 6.1 1D Array Indexing

```python
a = np.array([10, 20, 30, 40, 50])

# Positive indexing (0-based)
a[0]    # 10  — first element
a[4]    # 50  — last element
a[2]    # 30  — third element

# Negative indexing (from end)
a[-1]   # 50  — last element
a[-2]   # 40  — second to last
a[-5]   # 10  — first element

# Slicing: a[start:stop:step]
a[1:4]      # [20, 30, 40]  — indices 1,2,3 (stop is EXCLUSIVE)
a[::2]      # [10, 30, 50]  — every other element
a[::-1]     # [50, 40, 30, 20, 10]  — reversed
a[1::2]     # [20, 40]  — odd indices
```

### 6.2 2D Array Indexing

```python
a = np.array([[10, 20, 30],
              [40, 50, 60],
              [70, 80, 90]])

# Element access: a[row, col]
a[0, 1]     # 20  — row 0, column 1
a[2, 2]     # 90  — last row, last col
a[-1, -1]   # 90  — same thing with negative indices

# Row access
a[1, :]     # [40, 50, 60]  — entire row 1
a[1]        # [40, 50, 60]  — same (: is implicit)

# Column access
a[:, 2]     # [30, 60, 90]  — entire column 2
a[:, 0]     # [10, 40, 70]  — first column

# Submatrix (slicing)
a[0:2, 1:3]     # [[20, 30],  — rows 0-1, cols 1-2
                #  [50, 60]]
a[:2, :2]       # [[10, 20],  — top-left 2x2
                #  [40, 50]]
a[1:, 1:]       # [[50, 60],  — bottom-right 2x2
                #  [80, 90]]
```

### 6.3 The Comma Syntax Explained

`a[0, 1]` is equivalent to `a[(0, 1)]` — you're passing a tuple of indices.

- `a[0, 1]` — row 0, col 1
- `a[0:2, 1:3]` — rows 0 to 1, cols 1 to 2 (slices)
- `a[:, 2]` — all rows (`:` = `0:end`), col 2

**Java analogy:** In Java, `matrix[1][2]` accesses a 2D array. NumPy uses a single `[]` with comma-separated indices: `matrix[1, 2]`. Same concept, different syntax.

### 6.4 Specific Example from the Lesson

```python
a = np.array([[10, 20, 30],
              [40, 50, 60],
              [70, 80, 90]])

# a[0, 1] — row 0, col 1 = 20
# Visualization:
#     col0  col1  col2
# row0  10   [20]   30
# row1  40    50    60
# row2  70    80    90
print(a[0, 1])   # 20

# a[1, :] — entire row 1
# row0  10    20    30
# row1 [40   50    60]
# row2  70    80    90
print(a[1, :])   # [40 50 60]

# a[:, 2] — entire column 2
# col2: [30, 60, 90]
print(a[:, 2])   # [30 60 90]

# a[0:2, 1:3] — rows 0-1, cols 1-2
# row0: cols 1,2 = [20, 30]
# row1: cols 1,2 = [50, 60]
print(a[0:2, 1:3])   # [[20 30]
                      #  [50 60]]
```

### 6.5 Views vs Copies — CRITICAL CONCEPT

```python
a = np.array([1, 2, 3, 4, 5])

# Slicing returns a VIEW (not a copy)
b = a[1:4]      # b is a view of a
b[0] = 999      # modifying b also modifies a!
print(a)        # [1, 999, 3, 4, 5]  <- a was changed!

# To get an independent copy, use .copy()
c = a[1:4].copy()
c[0] = 0        # modifying c does NOT affect a
print(a)        # unchanged
```

**Why views?** Memory efficiency. If you have a 1GB array and slice a portion of it, you don't want NumPy to copy 1GB of data. The view just points into the same memory buffer with different strides/shape.

**When this bites you in ML:**
```python
X_train = X[0:800]      # this is a VIEW of X
X_train[:] = 0          # accidentally zeros out the original X!

# Safe approach:
X_train = X[0:800].copy()   # independent copy
```

### 6.6 Fancy Indexing

```python
a = np.array([10, 20, 30, 40, 50])

# Index with a list of indices
idx = [0, 2, 4]
print(a[idx])    # [10, 30, 50]

# Works in 2D too
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

rows = [0, 2]
print(matrix[rows])    # [[1, 2, 3],  — rows 0 and 2
                       #  [7, 8, 9]]

# Select specific (row, col) pairs
print(matrix[[0, 1, 2], [0, 1, 2]])  # [1, 5, 9]  — diagonal elements
```

**Fancy indexing returns a copy**, unlike basic slicing which returns a view.

---

## 7. Boolean Indexing

### 7.1 How It Works

```python
scores = np.array([45, 82, 67, 91, 55, 78])

# Step 1: Create a boolean mask
mask = scores > 70
print(mask)    # [False  True  False  True  False  True]

# Step 2: Use mask to index
print(scores[mask])    # [82 91 78]

# One-liner (most common form)
print(scores[scores > 70])    # [82 91 78]
```

**How it works internally:** The boolean array acts as a mask. NumPy collects all elements where the mask is `True` and returns them as a new array.

### 7.2 Modifying with Boolean Indexing

```python
scores = np.array([45, 82, 67, 91, 55, 78])

# Set all failing scores to 0
scores[scores < 60] = 0
print(scores)    # [0, 82, 67, 91, 0, 78]

# Apply a cap (maximum value)
scores[scores > 80] = 80
print(scores)    # [0, 80, 67, 80, 0, 78]

# More complex: clip between min and max
np.clip(scores, 50, 90)    # all values between 50 and 90
```

### 7.3 Combining Conditions

```python
a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# AND condition (use & not 'and')
a[(a > 3) & (a < 8)]     # [4 5 6 7]

# OR condition (use | not 'or')
a[(a < 3) | (a > 8)]     # [1 2 9 10]

# NOT condition (use ~ not 'not')
a[~(a > 5)]              # [1 2 3 4 5]

# WRONG! Don't use Python's 'and' / 'or' with NumPy arrays:
# a[(a > 3) and (a < 8)]    # ValueError!
```

The parentheses around each condition are **mandatory** due to Python operator precedence.

### 7.4 np.where — Conditional Selection

```python
a = np.array([1, -2, 3, -4, 5])

# np.where(condition, value_if_true, value_if_false)
result = np.where(a > 0, a, 0)
print(result)    # [1  0  3  0  5]  — negative values replaced with 0

# Get indices where condition is true
indices = np.where(a > 0)
print(indices)   # (array([0, 2, 4]),)
```

`np.where` is extremely common in ML for creating target masks or applying conditional transforms to data.

---

## 8. Vectorized Operations

### 8.1 Element-wise Arithmetic

```python
a = np.array([1, 2, 3, 4])
b = np.array([10, 20, 30, 40])

# These all operate element by element
a + b        # [11, 22, 33, 44]
a - b        # [-9, -18, -27, -36]
a * b        # [10, 40, 90, 160]
a / b        # [0.1, 0.1, 0.1, 0.1]
a // b       # [0, 0, 0, 0]  — floor division
a % b        # [1, 2, 3, 4]  — modulo
a ** 2       # [1, 4, 9, 16]  — power
```

### 8.2 Scalar Operations (Broadcasting Preview)

```python
a = np.array([1, 2, 3, 4])

a + 10      # [11, 12, 13, 14]  — adds 10 to every element
a * 2       # [2, 4, 6, 8]
a / 2       # [0.5, 1.0, 1.5, 2.0]
a ** 2      # [1, 4, 9, 16]
a - 1       # [0, 1, 2, 3]
10 / a      # [10, 5, 3.33, 2.5]
```

The scalar is "broadcast" to match the array's shape. The number `2` effectively becomes `[2, 2, 2, 2]` for the operation.

### 8.3 Mathematical Functions

```python
a = np.array([1, 4, 9, 16])

np.sqrt(a)        # [1. 2. 3. 4.]
np.square(a)      # [1. 16. 81. 256.]
np.log(a)         # natural log: [0. 1.386 2.197 2.773]
np.log2(a)        # log base 2
np.log10(a)       # log base 10
np.exp(a)         # e^x: [2.718, 54.6, 8103.1, ...]
np.abs(a)         # absolute value
np.sign(a)        # -1, 0, or 1

# Trigonometric
angles = np.linspace(0, 2*np.pi, 100)
np.sin(angles)
np.cos(angles)
np.tan(angles)
```

### 8.4 The Dot Product — Most Important Operation in ML

```python
a = np.array([1, 2, 3, 4])
b = np.array([10, 20, 30, 40])

# Dot product: sum of element-wise products
dot = np.dot(a, b)    # 1*10 + 2*20 + 3*30 + 4*40 = 10+40+90+160 = 300
print(dot)    # 300

# Alternative syntax
dot2 = a @ b          # @ is the matrix multiplication operator (Python 3.5+)
print(dot2)   # 300

# 2D: matrix multiplication
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

C = np.dot(A, B)    # [[19, 22], [43, 50]]
D = A @ B           # same
```

**Why dot product is fundamental to ML:**

A neural network layer computes: `output = W @ x + b`

Where:
- `W` = weight matrix (shape: `output_size x input_size`)
- `x` = input vector (shape: `input_size,`)
- `b` = bias vector (shape: `output_size,`)
- `output` = result vector (shape: `output_size,`)

This dot product IS the core of every neural network computation.

**Real-world meaning:** In linear regression, the prediction is `y_pred = sum(weight_i * feature_i)` — that's a dot product!

### 8.5 How Vectorization Eliminates Loops

Python loop approach:
```python
def predict_prices_loop(weights, features):
    result = []
    for sample in features:
        price = 0
        for i in range(len(weights)):
            price += weights[i] * sample[i]
        result.append(price)
    return result
```

NumPy vectorized approach:
```python
def predict_prices_numpy(weights, features):
    return features @ weights    # one matrix multiply — single BLAS call
```

The NumPy version runs a single optimized BLAS routine in C. For 10,000 samples and 100 features, the loop takes seconds; the NumPy version takes milliseconds.

---

## 9. Aggregate Operations

### 9.1 Basic Aggregates

```python
a = np.array([3, 1, 4, 1, 5, 9, 2, 6])

a.sum()      # 31  — sum of all elements
a.mean()     # 3.875  — arithmetic mean
a.std()      # 2.588  — standard deviation
a.var()      # 6.734  — variance (std squared)
a.min()      # 1
a.max()      # 9
a.argmin()   # 1  — INDEX of minimum value
a.argmax()   # 5  — INDEX of maximum value (position of 9)

np.median(a)     # 3.5  — middle value
np.cumsum(a)     # [3, 4, 8, 9, 14, 23, 25, 31]  — running total
np.cumprod(a)    # cumulative product
np.diff(a)       # difference between consecutive elements
```

### 9.2 Axis-Based Aggregation — CRITICAL FOR ML

```python
data = np.array([[1, 2, 3],
                 [4, 5, 6],
                 [7, 8, 9]])

# axis=0: operate along rows (collapse rows, keep columns)
data.sum(axis=0)    # [12, 15, 18]  — column sums
data.mean(axis=0)   # [4., 5., 6.]  — column means
data.max(axis=0)    # [7, 8, 9]     — column max

# axis=1: operate along columns (collapse columns, keep rows)
data.sum(axis=1)    # [6, 15, 24]   — row sums
data.mean(axis=1)   # [2., 5., 8.]  — row means
data.max(axis=1)    # [3, 6, 9]     — row max
```

**Memorization trick:**
- `axis=0` — result has same number of **columns** (rows were summed/collapsed)
- `axis=1` — result has same number of **rows** (columns were summed/collapsed)

**ML context:** If `data` is a dataset with shape `(n_samples, n_features)`:
- `data.mean(axis=0)` — mean of each feature across all samples — shape `(n_features,)`
- `data.mean(axis=1)` — mean of all features for each sample — shape `(n_samples,)`

You almost always want `axis=0` for feature statistics.

### 9.3 keepdims

```python
data = np.array([[1, 2, 3],
                 [4, 5, 6]])

# Without keepdims
data.mean(axis=1)                     # shape (2,)  — 1D

# With keepdims
data.mean(axis=1, keepdims=True)      # shape (2, 1)  — 2D column vector
# [[2.],
#  [5.]]
```

`keepdims=True` is crucial for broadcasting. If you compute column means and want to subtract them from each column, the shapes must be compatible.

---

## 10. Broadcasting

### 10.1 What Broadcasting Is

Broadcasting is NumPy's set of rules for performing operations on arrays with **different shapes**. Instead of requiring arrays to have exactly the same shape, NumPy automatically "stretches" smaller arrays to match larger ones.

### 10.2 Broadcasting Rules (Step by Step)

When operating on two arrays with different shapes, NumPy aligns their shapes from the **right** and applies these rules:

1. If the arrays have different numbers of dimensions, **prepend 1s** to the shape of the smaller one
2. Arrays with size 1 in a dimension are **stretched** to match the other array in that dimension
3. If sizes don't match and neither is 1, raise an error

### 10.3 Examples

**Scalar + Array:**
```python
a = np.array([[1, 2, 3],    # shape (2, 3)
              [4, 5, 6]])
b = 10                      # scalar = shape ()

result = a + b
# [[11, 12, 13],
#  [14, 15, 16]]
```

**1D array + 2D array:**
```python
a = np.array([[1, 2, 3],    # shape (2, 3)
              [4, 5, 6]])
b = np.array([10, 20, 30])  # shape (3,)

# Alignment (from right):
# a: (2, 3)
# b:    (3,)  -> padded to (1, 3) -> stretched to (2, 3)

result = a + b
# b effectively becomes: [[10, 20, 30],
#                         [10, 20, 30]]
# result: [[11, 22, 33],
#          [14, 25, 36]]
```

**Column vector + Row vector:**
```python
col = np.array([[1],    # shape (3, 1)
                [2],
                [3]])

row = np.array([10, 20, 30])    # shape (3,) -> treated as (1, 3)

result = col + row
# result: [[11, 21, 31],
#          [12, 22, 32],
#          [13, 23, 33]]
```

### 10.4 Broadcasting in Standardization

```python
X = np.random.randn(100, 5)   # shape (100, 5) — 100 samples, 5 features

mean = X.mean(axis=0)   # shape (5,)   — mean of each feature
std = X.std(axis=0)     # shape (5,)   — std of each feature

# Standardization:
X_normalized = (X - mean) / std
#   X:     shape (100, 5)
#   mean:  shape      (5,)  -> broadcast to (100, 5)
#   std:   shape      (5,)  -> broadcast to (100, 5)
```

The `mean` array is effectively repeated 100 times (one per sample) to match `X`'s shape. No explicit loops needed.

### 10.5 When Broadcasting Fails

```python
a = np.array([[1, 2, 3],   # shape (2, 3)
              [4, 5, 6]])
b = np.array([1, 2])       # shape (2,)

a + b   # ValueError!
# Alignment:
# a: (2, 3)
# b:    (2,)  -> padded to (1, 2) -> cannot stretch (3 != 2 and neither is 1)
```

---

## 11. Reshaping

### 11.1 reshape() — The Full Picture

```python
a = np.arange(12)    # [0, 1, 2, ..., 11]  — shape (12,)

# Total elements must stay the same (12 = 3*4 = 2*6 = 2*2*3)
a.reshape(3, 4)       # shape (3, 4)  — 3 rows, 4 cols
a.reshape(4, 3)       # shape (4, 3)
a.reshape(2, 6)       # shape (2, 6)
a.reshape(2, 2, 3)    # shape (2, 2, 3)  — 3D
a.reshape(1, 12)      # shape (1, 12)  — 1 row, 12 columns (2D row vector)
a.reshape(12, 1)      # shape (12, 1)  — column vector
```

**reshape returns a view** (when possible), so no data is copied. Only the shape and strides metadata changes.

### 11.2 The -1 Wildcard

```python
a = np.arange(24)

a.reshape(4, -1)     # -1 means "infer this dimension"  -> shape (4, 6)
a.reshape(-1, 8)     # -> shape (3, 8)
a.reshape(2, 3, -1)  # -> shape (2, 3, 4)
a.reshape(-1)        # -> shape (24,)  — flatten to 1D

# Rule: -1 = total_elements / product_of_other_dimensions
# a.reshape(4, -1)  -> 24 / 4 = 6  -> shape (4, 6)
```

`reshape(-1)` is equivalent to `flatten()`, converting any array to 1D.

### 11.3 flatten() vs ravel()

```python
a = np.array([[1, 2, 3],
              [4, 5, 6]])

# flatten() — always returns a COPY
flat1 = a.flatten()
flat1[0] = 999
print(a[0, 0])   # 1  — a is unchanged

# ravel() — returns a VIEW when possible
flat2 = a.ravel()
flat2[0] = 999
print(a[0, 0])   # 999  — a was changed!
```

Use `flatten()` when you need an independent copy. Use `ravel()` for memory efficiency when you won't modify the result.

### 11.4 reshape(-1, 1) — The Column Vector Pattern

This is used **constantly** in scikit-learn:

```python
# Many sklearn functions require 2D input
from sklearn.linear_model import LinearRegression

# If you have a 1D feature array:
x = np.array([1, 2, 3, 4, 5])   # shape (5,) — 1D

# sklearn's fit() requires shape (n_samples, n_features) — 2D
x_2d = x.reshape(-1, 1)  # shape (5, 1) — column vector
# model.fit(x_2d, y)      # Works!

# Explanation of (-1, 1):
# -1 = infer = 5 (total elements / 1)
# 1 = each sample has 1 feature
```

### 11.5 expand_dims and squeeze

```python
a = np.array([1, 2, 3])   # shape (3,)

# Add a dimension at position 0
b = np.expand_dims(a, axis=0)   # shape (1, 3)
# [[1, 2, 3]]

# Add a dimension at position 1
c = np.expand_dims(a, axis=1)   # shape (3, 1)
# [[1], [2], [3]]

# Remove dimensions of size 1
d = np.array([[[1, 2, 3]]])   # shape (1, 1, 3)
e = np.squeeze(d)              # shape (3,)
f = np.squeeze(d, axis=0)      # shape (1, 3)
```

`expand_dims` and `squeeze` are frequently used when adding or removing batch dimensions in neural network code.

### 11.6 Transposing

```python
a = np.array([[1, 2, 3],
              [4, 5, 6]])   # shape (2, 3)

a.T          # shape (3, 2)  — rows become columns
# [[1, 4],
#  [2, 5],
#  [3, 6]]

# transpose with specific axis order
b = np.zeros((2, 3, 4))
b.transpose(0, 2, 1).shape   # (2, 4, 3)  — swap last two axes
```

---

## 12. Code Exercise Walkthrough

### Task 1 — Array Creation

```python
import numpy as np

# a) Create array of even numbers from 2 to 20
evens = np.arange(2, 21, 2)
# arange(start, stop, step) — stop=21 because stop is exclusive
# [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
print(evens)

# Alternative approaches:
# np.arange(10) * 2 + 2
# np.linspace(2, 20, 10, dtype=int)
```

```python
# b) Create a 4x4 identity matrix
identity = np.eye(4)
print(identity)
# [[1. 0. 0. 0.]
#  [0. 1. 0. 0.]
#  [0. 0. 1. 0.]
#  [0. 0. 0. 1.]]

# The diagonal is 1, everything else is 0
# Identity element for matrix multiplication: A @ I = A
```

```python
# c) Create a 3x3 matrix of random floats between 0 and 1
np.random.seed(0)   # For reproducibility
random_matrix = np.random.rand(3, 3)
# Each element independently drawn from Uniform(0, 1)
```

```python
# d) Create 6 evenly spaced values between 0 and 5
spaced = np.linspace(0, 5, 6)
print(spaced)   # [0., 1., 2., 3., 4., 5.]

# linspace(start, stop, num) — num includes BOTH endpoints by default
# Step size = (5-0)/(6-1) = 1.0
```

### Task 2 — Shape and Indexing

```python
matrix = np.array([
    [1,  2,  3,  4],
    [5,  6,  7,  8],
    [9, 10, 11, 12]
])

# a) Print shape, ndim, size
print(matrix.shape)   # (3, 4)  — 3 rows, 4 columns
print(matrix.ndim)    # 2       — 2-dimensional
print(matrix.size)    # 12      — 3 * 4 = 12 total elements
```

```python
# b) Get element at row 2, col 3
# Row index 2 = third row: [9, 10, 11, 12]
# Col index 3 = fourth column
element = matrix[2, 3]
print(element)    # 12

# Visualization:
#        col0  col1  col2  col3
# row0:    1     2     3     4
# row1:    5     6     7     8
# row2:    9    10    11   [12]  <- this is [2, 3]
```

```python
# c) Get the entire second row
# Row index 1 (0-indexed) = [5, 6, 7, 8]
second_row = matrix[1, :]    # or simply matrix[1]
print(second_row)    # [5 6 7 8]
```

```python
# d) Get the last two columns
# Columns at index 2 and 3
last_two_cols = matrix[:, 2:]    # all rows, columns 2 onwards
print(last_two_cols)
# [[ 3  4]
#  [ 7  8]
#  [11 12]]
```

```python
# e) Get the 2x2 submatrix from top-right corner
# Top-right corner = rows 0-1, columns 2-3
top_right = matrix[0:2, 2:4]    # or matrix[:2, 2:]
print(top_right)
# [[3  4]
#  [7  8]]
```

### Task 3 — Vectorized Operations

```python
prices = np.array([100, 250, 80, 320, 150, 90])

# a) Apply 10% discount to all prices
discounted = prices * 0.90
print(discounted)    # [90.  225.   72.  288.  135.   81.]
```

```python
# b) Find prices above 150 (boolean indexing)
expensive = prices[prices > 150]
print(expensive)    # [250, 320]

# Find WHERE prices are above 150 (get indices)
expensive_idx = np.where(prices > 150)
print(expensive_idx)    # (array([1, 3]),)  — indices 1 and 3
```

```python
# c) Replace prices below 100 with 100 (price floor)
prices_copy = prices.copy()
prices_copy[prices_copy < 100] = 100
print(prices_copy)    # [100, 250, 100, 320, 150, 100]
# original 80 and 90 become 100

# Equivalent using np.maximum:
np.maximum(prices, 100)    # element-wise max of prices and 100
```

```python
# d) Compute mean, std, min, max
print(f"Mean:   {prices.mean():.2f}")    # 165.00
print(f"Std:    {prices.std():.2f}")     # 89.38
print(f"Min:    {prices.min()}")         # 80
print(f"Max:    {prices.max()}")         # 320
```

### Task 4 — Reshape

```python
data = np.arange(24)    # [0, 1, 2, ..., 23]  — shape (24,)

# a) Reshape to 6x4
a = data.reshape(6, 4)
print(a.shape)    # (6, 4)
print(a)
# [[ 0  1  2  3]
#  [ 4  5  6  7]
#  [ 8  9 10 11]
#  [12 13 14 15]
#  [16 17 18 19]
#  [20 21 22 23]]
```

```python
# b) Reshape to 2x3x4 (3D)
b = data.reshape(2, 3, 4)
print(b.shape)    # (2, 3, 4)
# Think of it as: 2 "layers", each with 3 rows and 4 columns
```

```python
# c) Flatten back to 1D
c = b.flatten()
print(c.shape)    # (24,)
print(c)          # [0, 1, 2, ..., 23]
```

```python
# d) Reshape to a column vector (24x1)
d = data.reshape(24, 1)    # or data.reshape(-1, 1)
print(d.shape)    # (24, 1)
# [[ 0]
#  [ 1]
#  [ 2]
#  ...
#  [23]]

# This is required for sklearn:
# sklearn.fit() expects shape (n_samples, n_features)
# If you have one feature, it must be (n_samples, 1), not (n_samples,)
```

---

## 13. Mini Challenge Solution

### Full Solution

```python
import numpy as np

np.random.seed(42)
X = np.random.randn(100, 5)   # 100 samples, 5 features — from standard normal N(0,1)

# 1. Shape of X
print("Shape of X:", X.shape)    # (100, 5)
# X has 100 rows (samples) and 5 columns (features)
# Each row is one data point; each column is one measurement/feature
```

```python
# 2. Mean of each feature (column mean)
feature_means = X.mean(axis=0)
print("Feature means:", feature_means)    # shape (5,)
# axis=0: collapse across rows, giving one value per column
# For X ~ N(0,1), we expect values close to 0
```

```python
# 3. Std of each feature
feature_stds = X.std(axis=0)
print("Feature stds:", feature_stds)    # shape (5,)
# Standard deviation measures how spread out values are
# For X ~ N(0,1), we expect values close to 1
```

```python
# 4. Standardization (Z-score normalization)
X_normalized = (X - feature_means) / feature_stds

# Broadcasting at work:
# X:              shape (100, 5)
# feature_means:  shape      (5,)  -> broadcast to (100, 5)
# (X - means):    shape (100, 5)
# feature_stds:   shape      (5,)  -> broadcast to (100, 5)
# X_normalized:   shape (100, 5)
```

```python
# 5. Verify
print("Normalized means:", X_normalized.mean(axis=0))
# Should be very close to [0, 0, 0, 0, 0]

print("Normalized stds:", X_normalized.std(axis=0))
# Should be very close to [1, 1, 1, 1, 1]
```

### Why Standardization Matters in ML

This technique — subtracting mean and dividing by standard deviation — is called **Z-score normalization** or **standardization**.

**Without standardization:**
- Feature A ranges 0-1 (e.g., age normalized)
- Feature B ranges 0-100,000 (e.g., income in dollars)
- ML models will over-weight Feature B just because its values are bigger
- Gradient descent converges slowly with mismatched feature scales

**With standardization:**
- Every feature has mean ~0 and std ~1
- All features contribute equally
- Gradient descent converges faster (symmetric loss surface)
- Works better with regularization

```python
# In sklearn, this is done with:
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)   # does exactly what we did manually
```

---

## 14. Interview Questions — Complete Answers

### Question 1: np.array([1,2,3]) vs Python list [1,2,3]

A Python list `[1,2,3]` stores Python objects (integers) in scattered heap memory, connected by pointers. Each Python integer takes ~28 bytes. Iterating requires following a pointer for each element (cache-unfriendly).

A NumPy array `np.array([1,2,3])` stores raw numbers in a contiguous block of memory as C-style values (8 bytes each for `int64`). All elements are adjacent, so the CPU can prefetch them efficiently. Operations like `a * 2` run in optimized C code (BLAS), not Python loops.

**Why it matters for ML:**
- A dataset of 100,000 samples x 50 features = 5,000,000 numbers
- As Python list: ~140MB, iteration takes seconds
- As NumPy array: ~40MB, operations take milliseconds
- ML training loops run thousands of iterations — Python lists would be prohibitively slow

**Java analogy:** Python list ~= `ArrayList<Integer>` (boxed ints, object overhead). NumPy array ~= `int[]` (primitive, compact, no boxing).

---

### Question 2: What does array.reshape(-1, 1) do?

`reshape(-1, 1)` converts an array to a 2D column vector:
- `-1` means "infer this dimension automatically" = total elements / 1 = n
- Result shape: `(n, 1)` — n rows, 1 column

```python
a = np.array([1, 2, 3, 4, 5])   # shape (5,)
b = a.reshape(-1, 1)             # shape (5, 1)
# [[1],
#  [2],
#  [3],
#  [4],
#  [5]]
```

**When you need it:**

1. **scikit-learn expects 2D arrays.** If you have one feature, you must reshape:
   ```python
   X = np.array([1, 2, 3, 4, 5])
   model.fit(X.reshape(-1, 1), y)    # required!
   ```

2. **Broadcasting with a 2D matrix.** To subtract a column-wise vector from a 2D array:
   ```python
   data = np.zeros((3, 4))
   col_vector = np.array([1, 2, 3]).reshape(-1, 1)   # shape (3, 1)
   result = data - col_vector   # broadcast: (3, 4) - (3, 1)
   ```

---

### Question 3: Compute mean of each of 10 features from (1000, 10) array

```python
X = np.random.randn(1000, 10)   # 1000 samples, 10 features

feature_means = X.mean(axis=0)   # shape (10,)
```

`axis=0` collapses the first dimension (the 1000 samples), leaving one value per feature (column). For each of the 10 columns, it averages all 1000 values.

**Mnemonic:** "axis=0 removes axis 0 (rows), leaving axis 1 (columns) intact."

**Wrong approaches:**
```python
X.mean()           # scalar — mean of ALL 10,000 values (ignores feature structure)
X.mean(axis=1)     # shape (1000,) — mean per sample, not per feature
```

---

### Question 4: What is broadcasting?

Broadcasting allows NumPy to perform operations on arrays with different shapes by implicitly stretching the smaller array to match the larger one, **without copying data**.

**Rules:**
1. Align shapes from the right
2. Dimensions with size 1 are stretched to match
3. Missing dimensions are treated as size 1

**Simple example:**
```python
X = np.ones((100, 3))                    # shape (100, 3)
means = np.array([1.0, 2.0, 3.0])       # shape (3,)

result = X - means
# means is treated as shape (1, 3) -> stretched to (100, 3)
# X[i, j] - means[j] for all i,j simultaneously
```

Without broadcasting, you'd need to tile:
```python
means_tiled = np.tile(means, (100, 1))   # shape (100, 3) — copies data!
result = X - means_tiled
```

Broadcasting avoids this memory allocation and copy.

---

### Question 5: Why is float32 preferred over float64 in deep learning?

Three main reasons:

**1. Memory:** `float32` uses 4 bytes; `float64` uses 8 bytes.
- A layer with 1 million weights: 4MB vs 8MB
- A modern GPU has 8-24GB VRAM — cutting memory in half means larger models or bigger batches

**2. GPU hardware is optimized for `float32`:**
- NVIDIA GPUs have dedicated `float32` arithmetic units (CUDA cores)
- `float32` throughput is typically 2x `float64` throughput on consumer GPUs
- Tensor cores (NVIDIA Volta+) operate on `float16` and `bfloat16` — even faster

**3. Precision is sufficient:**
- Neural network training uses stochastic gradient descent — inherent randomness
- 7 significant digits (`float32`) is enough; 15 digits (`float64`) is overkill
- Noise from stochastic batches dominates over floating-point rounding errors

**Modern trend — mixed precision (`float16`/`bfloat16`):**
```python
# In PyTorch:
model = model.half()       # convert to float16
# or
from torch.cuda.amp import autocast
with autocast():
    output = model(x)      # automatic mixed precision
```

---

## 15. Performance Internals

### 15.1 The Bytecode Problem

Python is interpreted. Every Python operation — even `a + 1` — goes through:
1. Look up the `+` operator for the type of `a`
2. Call the `__add__` method
3. Check for `NotImplemented`, handle exceptions
4. Create a new Python object for the result
5. Manage reference counts for garbage collection

For a list of 1,000,000 elements, this happens 1,000,000 times.

NumPy sidesteps this entirely: `a + 1` calls a C function once, which loops in C over raw bytes.

### 15.2 SIMD Instructions

Modern CPUs have SIMD (Single Instruction, Multiple Data) registers. An AVX2 register is 256 bits wide — it can hold 4 `float64` values or 8 `float32` values.

```
Without SIMD:  ADD a[0]+b[0], ADD a[1]+b[1], ...   1 addition per instruction
With AVX2:     ADD [a[0..3]] + [b[0..3]]            4 additions per instruction (float64)
With AVX2:     ADD [a[0..7]] + [b[0..7]]            8 additions per instruction (float32)
```

NumPy's underlying BLAS library uses these SIMD instructions automatically.

### 15.3 Cache Effects

CPU cache hierarchy (typical):
- L1 cache: ~32KB, ~4 CPU cycles to access
- L2 cache: ~256KB, ~12 CPU cycles
- L3 cache: ~8MB, ~40 CPU cycles
- RAM: ~GBs, ~100-300 CPU cycles

A NumPy array of 1,000 `float64` values = 8,000 bytes = fits in L1 cache. Once loaded, all subsequent operations are L1-speed.

A Python list of 1,000 objects: each `PyIntObject` is elsewhere in the heap. Every access causes a cache miss.

This is why contiguous memory layout is so important for performance.

---

## 16. NumPy in the ML Pipeline

### 16.1 The Data Flow

```
Raw Data (CSV, images, text)
      |
NumPy arrays (data loading and cleaning)
      |
NumPy operations (feature engineering, preprocessing)
      |
scikit-learn / PyTorch / TensorFlow (model training)
      |
NumPy arrays (predictions, evaluation)
      |
Matplotlib / Pandas (visualization, reporting)
```

### 16.2 Data Preprocessing Pipeline Example

```python
import numpy as np

data = np.array([
    [25, 50000, 0],
    [35, 80000, 1],
    [45, 120000, 1],
    [22, 35000, 0],
])
# shape (4, 3): 4 samples, columns = [age, income, label]

# Separate features and labels
X = data[:, :-1]      # all rows, all but last column -> shape (4, 2)
y = data[:, -1]       # all rows, last column -> shape (4,)

# Train/test split
n = len(X)
split = int(n * 0.75)   # 75% train

X_train = X[:split]     # shape (3, 2)
X_test  = X[split:]     # shape (1, 2)
y_train = y[:split]
y_test  = y[split:]

# Standardize features
mean = X_train.mean(axis=0)    # shape (2,)
std  = X_train.std(axis=0)     # shape (2,)

# IMPORTANT: fit scaler on training data, apply to both train and test
# Never let test data influence your scaling parameters
X_train_scaled = (X_train - mean) / std
X_test_scaled  = (X_test  - mean) / std    # use TRAIN mean/std
```

### 16.3 Manual Linear Regression with NumPy

```python
import numpy as np

# Generate fake data: y = 2x + 1 + noise
np.random.seed(42)
X = np.random.rand(100, 1)          # 100 samples, 1 feature
y = 2 * X.flatten() + 1 + 0.1 * np.random.randn(100)

# Add bias term (column of 1s)
X_b = np.hstack([np.ones((100, 1)), X])   # shape (100, 2)

# Ordinary Least Squares (closed-form solution):
# theta = (X^T @ X)^-1 @ X^T @ y
theta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y

print(f"Intercept: {theta[0]:.3f}")   # ~= 1.0
print(f"Slope:     {theta[1]:.3f}")   # ~= 2.0

# Predict
y_pred = X_b @ theta    # shape (100,)

# Mean Squared Error
mse = np.mean((y - y_pred) ** 2)
print(f"MSE: {mse:.4f}")
```

This is actual linear regression, implemented entirely in NumPy. This is what scikit-learn's `LinearRegression` does internally.

### 16.4 Understanding Batch Processing

In deep learning, you never process one sample at a time. You process a **batch** of samples simultaneously:

```python
# Batch forward pass (32 samples):
X_batch = np.random.randn(32, 3)        # shape (32, 3)
W = np.random.randn(3, 5)               # shape (3, 5) — weight matrix
b = np.zeros(5)                          # shape (5,) — bias

output_batch = X_batch @ W + b          # shape (32, 5)
# Broadcasting: b shape (5,) -> (1, 5) -> (32, 5)
```

The batch operation is one matrix multiplication — a single BLAS call that processes all 32 samples simultaneously and efficiently.

---

## 17. Common Mistakes and How to Avoid Them

### 17.1 Shape Confusion — (n,) vs (n,1) vs (1,n)

```python
a = np.array([1, 2, 3])   # shape (3,)  — 1D array
b = a.reshape(3, 1)        # shape (3, 1) — column vector
c = a.reshape(1, 3)        # shape (1, 3) — row vector

# These look similar but behave differently:
print(a.T)      # [1 2 3]  — transpose of 1D does nothing!
print(b.T)      # [[1 2 3]]  — shape (1, 3) — works correctly
```

**Rule:** Always know whether your arrays are 1D `(n,)` or 2D `(n,1)`. Use `.reshape(-1,1)` proactively.

### 17.2 Modifying Views Accidentally

```python
X = np.random.randn(100, 5)
X_train = X[:80]        # VIEW!
X_train[:] = 0          # This zeros out rows 0-79 of X!

# Safe version:
X_train = X[:80].copy()
X_train[:] = 0          # X is unchanged
```

### 17.3 Using Python's `and`/`or` Instead of `&`/`|`

```python
a = np.array([1, 2, 3, 4, 5])

# WRONG:
# a[a > 1 and a < 4]     # ValueError!
# a[a > 1 or a < 4]      # ValueError!

# RIGHT:
a[(a > 1) & (a < 4)]   # [2 3]
a[(a < 2) | (a > 4)]   # [1 5]
```

### 17.4 Axis=0 vs Axis=1 Confusion

```python
data = np.array([[1, 2, 3],
                 [4, 5, 6]])

# axis=0: collapses ROWS -> result has shape (3,) — one per column
data.sum(axis=0)   # [5, 7, 9]   <- column sums

# axis=1: collapses COLUMNS -> result has shape (2,) — one per row
data.sum(axis=1)   # [6, 15]     <- row sums

# Memory trick: "axis=0 means you're summing down the columns"
```

### 17.5 np.arange with Float Steps

```python
# Floating-point precision issue:
a = np.arange(0, 1, 0.1)
print(len(a))    # might be 10 or 11 depending on floating point

# Safe alternative:
a = np.linspace(0, 0.9, 10)   # always exactly 10 elements
```

---

## 18. Advanced NumPy Concepts to Know About

### 18.1 Linear Algebra Functions

```python
A = np.array([[3, 1], [1, 2]])

# Determinant
np.linalg.det(A)     # 5.0

# Inverse
np.linalg.inv(A)     # [[0.4, -0.2], [-0.2, 0.6]]

# Eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)

# Singular Value Decomposition (used in PCA, recommendation systems)
U, S, Vt = np.linalg.svd(A)

# Solve linear system Ax = b
b = np.array([5, 4])
x = np.linalg.solve(A, b)   # faster and more stable than inv(A) @ b

# Norm
np.linalg.norm(A)            # Frobenius norm
np.linalg.norm(A, axis=1)    # L2 norm of each row
```

### 18.2 Stacking and Splitting

```python
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])

# Stack vertically (stack rows)
np.vstack([a, b])    # shape (4, 2)
np.concatenate([a, b], axis=0)   # same

# Stack horizontally (stack columns)
np.hstack([a, b])    # shape (2, 4)
np.concatenate([a, b], axis=1)   # same

# Stack along new axis
np.stack([a, b], axis=0)    # shape (2, 2, 2) — adds new first dim

# Split
chunks = np.split(np.arange(12), 3)    # split into 3 equal arrays
```

### 18.3 np.einsum — Einstein Summation

```python
# Powerful notation for complex tensor operations
A = np.random.randn(3, 4)
B = np.random.randn(4, 5)

# Matrix multiplication:
C = np.einsum('ij,jk->ik', A, B)   # same as A @ B

# Batch matrix multiply:
A = np.random.randn(10, 3, 4)
B = np.random.randn(10, 4, 5)
C = np.einsum('bij,bjk->bik', A, B)   # shape (10, 3, 5)
```

`einsum` is used extensively in attention mechanisms in Transformers.

---

## 19. Summary and Connections

### What You've Learned

| Concept | Key Takeaway |
|---|---|
| ndarray | Contiguous typed memory — the foundation of all ML data structures |
| shape | The most important attribute — check it constantly when debugging |
| dtype | `float32` for GPU; `float64` for CPU; `uint8` for images; `bool` for masks |
| Vectorized ops | One NumPy call > one million Python iterations |
| Broadcasting | Implicit stretching of smaller arrays — eliminates explicit loops |
| Boolean indexing | `a[a > 0]` — the NumPy way to filter data |
| reshape | Change dimensions without copying data — essential for model input preparation |
| Axis operations | `axis=0` = across rows; `axis=1` = across columns |
| Views vs copies | Slices are views — use `.copy()` when you need independence |

### How Day 3 Connects to Future Days

```
Day 3: NumPy Basics
    |
    +-- Day 4: Pandas (built on NumPy arrays)
    +-- Day 5: Matplotlib (NumPy arrays -> plots)
    +-- Day 7-10: Probability/Statistics (NumPy operations)
    +-- Week 3: Machine Learning
    |       |
    |       +-- Linear Regression: X @ theta — matrix multiply
    |       +-- Normalization: (X - mean) / std — broadcasting
    |       +-- Gradient Descent: weights -= lr * gradients — vectorized update
    |
    +-- Week 5: Neural Networks
    |       |
    |       +-- Forward pass: W @ x + b — vectorized
    |       +-- Backpropagation: chain rule = matrix multiplies
    |       +-- Mini-batches: 4D arrays (batch, height, width, channels)
    |
    +-- Week 7: Deep Learning Frameworks
            |
            +-- PyTorch tensors = NumPy arrays on GPU
            +-- TensorFlow: similar ndarray model
```

### The Core Mental Model

Think of NumPy arrays as:

> **A typed, fixed-size, multidimensional spreadsheet stored in contiguous memory, on which you can apply mathematical operations to every cell simultaneously using optimized C and Fortran code.**

Every time you're about to write a Python loop over an array, ask yourself:
1. Can I express this as a vectorized operation? (`+`, `*`, `np.sqrt`, etc.)
2. Can I use boolean indexing instead of an `if` inside a loop?
3. Can I use `axis=` to aggregate without a loop?
4. Can I use broadcasting instead of tiling/repeating an array?

If yes to any of these, don't write the loop.

### Relationship to Java

| NumPy operation | Java equivalent |
|---|---|
| `np.array([1,2,3])` | `new int[]{1, 2, 3}` |
| `a.size` or `len(a)` | `arr.length` |
| `np.zeros_like(arr)` | `Arrays.fill(arr, 0)` (in-place) |
| `np.copy(arr)` | `Arrays.copyOf(arr, arr.length)` |
| `arr.sum()` | `Arrays.stream(arr).sum()` |
| `arr1 @ arr2` | manual nested loops (no equivalent!) |
| `arr[0, 1]` | `arr[0][1]` |
| `np.arange(n)` | `IntStream.range(0, n).toArray()` |

The most important difference: in Java, there's no equivalent of broadcasting. You'd need to write explicit nested loops. NumPy's broadcasting alone justifies learning it — it eliminates entire categories of boilerplate code.

---

## Self-Test Questions

Before proceeding to Day 4, answer these without looking at notes:

1. You have `X = np.zeros((1000, 20))`. What is `X.shape`? `X.ndim`? `X.size`?
2. How do you select all rows where column 3 is greater than 0.5?
3. What does `X.mean(axis=0)` return? What shape does it have?
4. You have a 1D array `a = np.arange(100)`. How do you reshape it to a 10x10 matrix?
5. Why does `np.array([1.0, 2, 3]).dtype` return `float64` and not `int64`?
6. What's the difference between `a[1:4]` (slice) and `a[[1,2,3]]` (fancy index) in terms of views vs copies?
7. How would you normalize all columns of a matrix to [0, 1] range?

---

*Close GitHub issue #14 when all tasks are done.*

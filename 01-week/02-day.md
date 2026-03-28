# Day 2 — Python Data Structures

**Phase:** Foundations | **Week:** 1 | **GitHub Issue:** #13

---

## 🎯 Objective
Master Python's core data structures — list, dict, tuple, set — at the level needed for AI/ML work. You will slice arrays, manipulate dicts, and write the kind of data-wrangling code that appears in every ML pipeline.

---

## 🧠 Concept Explanation

### Java → Python Cheat Sheet

| Java | Python | Key difference |
|---|---|---|
| `ArrayList<T>` | `list` | Dynamic, ordered, mutable |
| `HashMap<K,V>` | `dict` | Key-value, ordered (3.7+) |
| `HashSet<T>` | `set` | Unordered, no duplicates |
| `int[]` (fixed) | `tuple` | Immutable sequence |
| `LinkedList` | `collections.deque` | Fast append/pop both ends |

In AI/ML you will use `list` and `dict` constantly. `tuple` shows up as matrix shapes (e.g. `(28, 28, 1)`). `set` is great for deduplication.

---

### 1. Lists — Your Workhorse

```python
scores = [92, 87, 78, 95, 60]

# Indexing — like Java arrays
scores[0]    # 92  (first)
scores[-1]   # 60  (last — Python superpower)
scores[-2]   # 95  (second from last)

# Slicing — [start:stop:step]  (stop is exclusive, like Java substring)
scores[1:4]  # [87, 78, 95]   (index 1,2,3)
scores[:3]   # [92, 87, 78]   (first 3)
scores[2:]   # [78, 95, 60]   (from index 2 to end)
scores[::2]  # [92, 78, 60]   (every other element)
scores[::-1] # [60, 95, 78, 87, 92]  (reversed!)
```

**Key list methods for ML:**
```python
data = [3, 1, 4, 1, 5, 9, 2, 6]

len(data)           # 8   — size
data.append(7)      # add to end
data.extend([8, 9]) # merge another list
data.sort()         # in-place sort
sorted(data)        # returns new sorted list (non-destructive)
data.count(1)       # 2   — how many times 1 appears
data.index(5)       # position of first 5
sum(data)           # sum of all elements
min(data), max(data)
```

---

### 2. Dictionaries — Structured Data

In ML, a single training sample is almost always a dict:
```python
sample = {
    "age": 32,
    "salary": 75000,
    "department": "engineering",
    "churn": False   # label
}

# Access
sample["age"]           # 32
sample.get("age", 0)    # safe access with default (no KeyError)

# Add / update
sample["experience"] = 5
sample["salary"] = 80000

# Iterate — very common in ML pipelines
for key, value in sample.items():
    print(f"{key}: {value}")

# Keys and values as lists
list(sample.keys())     # ['age', 'salary', ...]
list(sample.values())   # [32, 75000, ...]

# Dict comprehension — like Java Collectors.toMap()
squared = {x: x**2 for x in range(1, 6)}
# {1:1, 2:4, 3:9, 4:16, 5:25}
```

---

### 3. Tuples — Immutable Records

```python
# Common in ML as shapes, coordinates, return values
image_shape = (224, 224, 3)   # height, width, channels
point = (3.5, 7.2)

# Unpack — clean way to handle multi-return
height, width, channels = image_shape
print(height)    # 224

# Tuples as dict keys (lists can't be dict keys — they're mutable)
grid = {(0, 0): "start", (3, 4): "end"}
```

---

### 4. Sets — Deduplication & Membership

```python
labels_seen = {"cat", "dog", "cat", "bird", "dog"}
print(labels_seen)  # {'cat', 'dog', 'bird'} — duplicates gone

# Fast membership check (O(1) like HashSet — unlike list O(n))
"cat" in labels_seen   # True

# Set operations — useful for comparing feature sets
train_labels = {"cat", "dog", "bird"}
test_labels  = {"cat", "fish", "bird"}

train_labels & test_labels   # intersection: {'cat', 'bird'}
train_labels | test_labels   # union:        {'cat', 'dog', 'bird', 'fish'}
train_labels - test_labels   # difference:   {'dog'}  (in train, not test)
```

---

### 5. Nested Structures — Real ML Data Shape

ML datasets are almost always lists of dicts:

```python
dataset = [
    {"age": 25, "salary": 50000, "churn": False},
    {"age": 45, "salary": 90000, "churn": True},
    {"age": 33, "salary": 67000, "churn": False},
]

# Get all ages
ages = [row["age"] for row in dataset]          # [25, 45, 33]

# Filter churned customers
churned = [row for row in dataset if row["churn"]]

# Average salary
avg_salary = sum(row["salary"] for row in dataset) / len(dataset)
```

This pattern — **list of dicts → list comprehension → aggregate** — is the foundation of data pipelines.

---

## 🔑 Key Terms

| Term | Meaning |
|---|---|
| **Mutable** | Can be changed after creation (list, dict, set) |
| **Immutable** | Cannot be changed (tuple, str, int) |
| **Slicing** | Extracting a sub-sequence with `[start:stop:step]` |
| **List comprehension** | One-line loop that builds a list `[expr for x in iter if cond]` |
| **Dict comprehension** | Same idea for dicts `{k: v for ...}` |
| **Unpacking** | Assigning multiple variables from a sequence in one line |

---

## 💻 Code Exercise

Create `day02.ipynb` and solve each task:

```python
# Task 1 — Slicing
temps = [22, 19, 25, 30, 28, 15, 17, 23, 29, 31]

# a) Last 3 temperatures
# b) Temperatures from index 3 to 7
# c) Every other temperature
# d) Reversed list (without modifying original)
```

```python
# Task 2 — Dict operations
student = {
    "name": "Hari",
    "scores": [85, 90, 78, 92, 88],
    "passed": True
}

# a) Print the average score (use sum() and len())
# b) Add a key "grade" = "A" if average >= 85, else "B"
# c) Print all keys and values using .items()
```

```python
# Task 3 — List of dicts (real ML pattern)
employees = [
    {"name": "Alice", "dept": "engineering", "salary": 95000},
    {"name": "Bob",   "dept": "marketing",   "salary": 60000},
    {"name": "Carol", "dept": "engineering", "salary": 105000},
    {"name": "Dave",  "dept": "marketing",   "salary": 72000},
]

# a) Get a list of all names
# b) Get only engineering employees
# c) Get the highest salary
# d) Get average salary per department as a dict:
#    {"engineering": 100000.0, "marketing": 66000.0}
```

```python
# Task 4 — Set operations
model_a_predictions = {"cat", "dog", "fish", "bird", "horse"}
model_b_predictions = {"cat", "lion", "fish", "tiger", "bird"}

# a) Labels both models agree on
# b) Labels only model_a predicted
# c) All unique labels across both models
# d) How many labels are unique to each model combined?
```

---

## 🔥 Mini Challenge

```python
# Given a list of ML training logs, extract insights:
logs = [
    {"epoch": 1, "loss": 0.95, "accuracy": 0.61},
    {"epoch": 2, "loss": 0.82, "accuracy": 0.70},
    {"epoch": 3, "loss": 0.74, "accuracy": 0.75},
    {"epoch": 4, "loss": 0.65, "accuracy": 0.81},
    {"epoch": 5, "loss": 0.58, "accuracy": 0.85},
]

# 1. Find the epoch with the highest accuracy
# 2. Get all epochs where loss < 0.75
# 3. Build a dict: {epoch_number: accuracy} for all epochs
# 4. What is the accuracy improvement from epoch 1 to epoch 5?
```

---

## ❓ Interview Questions

1. **What is the time complexity of `x in my_list` vs `x in my_set`? Why does this matter in ML?**
2. **Why can't a list be used as a dictionary key, but a tuple can?**
3. **You have a list of 10,000 customer records (list of dicts). How would you filter only those where `status == "active"` in one line?**
4. **What does `scores[-1]` return? What about `scores[-3:]`?**
5. **In a ML pipeline, you have feature names as a list and feature values as another list of the same length. How do you combine them into a dict in one line?** *(Hint: `zip`)*

---

## 📝 Summary

- `list` — ordered, mutable, indexed. Your go-to for sequences and datasets.
- `dict` — key-value store. Every ML sample/record is a dict.
- `tuple` — immutable. Used for shapes, coordinates, multi-return values.
- `set` — unordered, no duplicates. O(1) lookup, great for label sets.
- **List of dicts + list comprehension** = the backbone of data processing in Python.
- Negative indexing (`-1`, `-2`) and slicing (`[1:4]`, `[::-1]`) are Python superpowers you'll use daily.

---

## ✅ Done Checklist
- [ ] All 4 code exercise tasks completed in `day02.ipynb`
- [ ] Mini challenge solved
- [ ] Can explain mutability difference between list and tuple
- [ ] Close GitHub issue #13 when done

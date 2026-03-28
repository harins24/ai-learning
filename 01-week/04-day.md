# Day 4 — Pandas Basics

**Phase:** Foundations | **Week:** 1 | **GitHub Issue:** #15

---

## 🎯 Objective
Use Pandas to load, explore, and manipulate tabular data — the format most real-world ML datasets come in. Learn the DataFrame API that you will use in every ML project.

---

## 🧠 Concept Explanation

### What is Pandas?
Pandas gives you a `DataFrame` — a 2D labeled table with columns of potentially different types.

**Backend analogy:**
- A `DataFrame` is like a database `ResultSet` with column names, but you can mutate it, filter it, group it, and join it all in memory.
- A `Series` (single column) is like a `Map<Index, Value>` with vectorized operations.

```
DataFrame:
   name   age  salary  churn
0  Alice   32   95000  False
1  Bob     45   60000   True
2  Carol   28   72000  False

Each column = a Series
Each row    = a record
```

---

### Loading Data

```python
import pandas as pd

# From CSV (most common)
df = pd.read_csv("customers.csv")

# From dict (useful for building test data)
df = pd.DataFrame({
    "name":   ["Alice", "Bob", "Carol"],
    "age":    [32, 45, 28],
    "salary": [95000, 60000, 72000],
    "churn":  [False, True, False]
})

# From list of dicts (your Day 2 pattern → DataFrame)
records = [
    {"name": "Alice", "age": 32, "salary": 95000},
    {"name": "Bob",   "age": 45, "salary": 60000},
]
df = pd.DataFrame(records)
```

---

### Exploring a DataFrame — Always Do This First

```python
df.shape           # (rows, cols)
df.dtypes          # column types
df.head(5)         # first 5 rows
df.tail(5)         # last 5 rows
df.info()          # dtypes + null counts — your first data quality check
df.describe()      # count, mean, std, min, quartiles, max for numeric cols
df.columns         # column names
df.isnull().sum()  # null count per column
```

In a real ML project, `df.info()` and `df.describe()` are the first two things you run after loading data.

---

### Selecting Data

```python
# Single column → Series
df["age"]
df.age            # same thing

# Multiple columns → DataFrame
df[["name", "salary"]]

# Rows by position (like array index)
df.iloc[0]        # first row
df.iloc[0:3]      # rows 0,1,2
df.iloc[1, 2]     # row 1, col 2

# Rows by label/condition
df.loc[0]                          # row with index label 0
df.loc[df["age"] > 30]             # filter — like SQL WHERE
df.loc[df["churn"] == True, "salary"]  # churned customers' salaries
```

---

### Adding & Modifying Columns

```python
# New column
df["salary_k"] = df["salary"] / 1000          # salary in thousands
df["senior"]   = df["age"] > 40               # boolean flag

# Apply a function to a column
df["name_upper"] = df["name"].str.upper()
df["age_group"]  = df["age"].apply(lambda x: "young" if x < 35 else "senior")
```

---

### Filtering & Sorting

```python
# Filter
young = df[df["age"] < 35]
high_earners = df[df["salary"] > 70000]
churned_seniors = df[(df["churn"] == True) & (df["age"] > 40)]  # AND
risk = df[(df["salary"] < 65000) | (df["churn"] == True)]       # OR

# Sort
df.sort_values("salary", ascending=False)          # highest salary first
df.sort_values(["churn", "salary"], ascending=[True, False])
```

---

### Grouping & Aggregation

```python
# Average salary by churn status
df.groupby("churn")["salary"].mean()

# Multiple aggregations
df.groupby("churn").agg({
    "salary": ["mean", "min", "max"],
    "age":    "mean"
})

# Value counts — how many of each category
df["churn"].value_counts()
```

**Backend analogy:** `groupby().agg()` is exactly `GROUP BY ... HAVING` in SQL.

---

### Handling Missing Values (preview of Day 5)

```python
df.isnull().sum()          # count nulls per column
df.dropna()                # drop rows with any null
df.fillna(0)               # fill nulls with 0
df["age"].fillna(df["age"].mean())  # fill with column mean
```

---

## 🔑 Key Terms

| Term | Meaning |
|---|---|
| **DataFrame** | 2D labeled table — the core Pandas object |
| **Series** | 1D labeled array — a single column of a DataFrame |
| **Index** | Row labels (default: 0, 1, 2, ...) |
| **iloc** | Integer-location based indexing (by position) |
| **loc** | Label-based indexing (by index label or boolean mask) |
| **groupby** | Split-apply-combine — like SQL GROUP BY |
| **NaN** | Not a Number — Pandas' representation of missing values |

---

## 💻 Code Exercise

Download this dataset or create it inline. Create `day04.ipynb`:

```python
import pandas as pd
import numpy as np

# Dataset: employee records
np.random.seed(42)
n = 50
df = pd.DataFrame({
    "employee_id": range(1, n+1),
    "department":  np.random.choice(["engineering", "sales", "marketing", "hr"], n),
    "age":         np.random.randint(22, 60, n),
    "salary":      np.random.randint(40000, 120000, n),
    "years_exp":   np.random.randint(0, 20, n),
    "left":        np.random.choice([True, False], n, p=[0.3, 0.7])
})
# Introduce some nulls
df.loc[df.sample(5).index, "salary"] = np.nan

# Task 1 — Explore
# a) Print shape, dtypes, and first 5 rows
# b) How many nulls in each column?
# c) Summary stats for numeric columns

# Task 2 — Select & Filter
# a) Get all engineering employees
# b) Get employees with salary > 80000 who haven't left
# c) Get name and salary of the top 5 highest earners

# Task 3 — New columns
# a) Add "salary_band": "low" < 60000, "mid" 60000-90000, "high" > 90000
#    Hint: use pd.cut() or apply()
# b) Add "experience_level": "junior" < 3 years, "mid" 3-8, "senior" > 8

# Task 4 — Groupby
# a) Average salary per department
# b) Attrition rate (% who left) per department
# c) Count of employees per department who are "senior" experience level
```

---

## 🔥 Mini Challenge

```python
# Use the same df from above

# 1. Find the department with the highest average salary
# 2. Find employees who are outliers: salary > mean + 2*std
# 3. For each department, what % of employees have left?
#    Return as a dict: {"engineering": 0.28, "sales": 0.35, ...}
# 4. Create a pivot: rows = department, cols = experience_level,
#    values = average salary
#    Hint: df.pivot_table(...)
```

---

## ❓ Interview Questions

1. **What is the difference between `iloc` and `loc`?**
2. **You have a DataFrame with 100K rows. How do you quickly check data quality before building an ML model?**
3. **How does `groupby().agg()` in Pandas compare to SQL `GROUP BY`?**
4. **What is the difference between `df.dropna()` and `df.fillna(df.mean())`? When would you choose each?**
5. **You have a column "age" with some null values. You fill them with the mean. What is the problem with this approach for a classification model?**

---

## 📝 Summary

- `DataFrame` = in-memory table with column names and row index
- Always run `df.info()` and `df.describe()` after loading data
- `loc` filters by label/condition, `iloc` by position
- `groupby().agg()` = SQL GROUP BY — the most powerful analysis tool
- Adding derived columns (`df["new"] = ...`) is how you do feature engineering
- Missing value handling (`dropna`, `fillna`) is mandatory before ML training

---

## ✅ Done Checklist
- [ ] All 4 tasks completed in `day04.ipynb`
- [ ] Mini challenge: pivot table created
- [ ] Can explain difference between `loc` and `iloc` without looking it up
- [ ] Close GitHub issue #15 when done

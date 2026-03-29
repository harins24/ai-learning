# Day 5 — Data Cleaning Practice

**Phase:** Foundations | **Week:** 1 | **GitHub Issue:** #16

---

## 🎯 Objective
Learn to detect and fix dirty data — missing values, duplicates, wrong types, outliers, and inconsistent categories. Real-world ML models fail not because of bad algorithms but because of bad data.

---

## 🧠 Concept Explanation

### Why Data Cleaning?

> "Data scientists spend 80% of their time cleaning data and 20% complaining about it."

Raw data from production systems is almost always dirty:
- Database migrations leaving `NULL` values
- Frontend forms accepting "N/A", "none", "", "-" as empty
- Type mismatches from CSV exports (age stored as "32.0" string)
- Duplicate rows from JOIN bugs or double inserts
- Outliers from data entry errors (age = 999)

**Backend analogy:** Data cleaning = input validation at the data layer. Just like you validate request DTOs in Spring Boot before processing, you validate DataFrames before training.

---

### 1. Finding & Handling Missing Values

```python
import pandas as pd
import numpy as np

df.isnull().sum()              # count nulls per column
df.isnull().mean() * 100       # null percentage per column

# Strategies:
# 1. Drop rows (when few rows affected)
df.dropna(subset=["salary"])   # drop rows where salary is null

# 2. Drop columns (when > 50-70% missing — column is useless)
df.drop(columns=["remarks"])

# 3. Fill with constant
df["salary"].fillna(0)

# 4. Fill with mean/median (numeric)
df["salary"].fillna(df["salary"].mean(), inplace=True)
df["age"].fillna(df["age"].median(), inplace=True)  # median is robust to outliers

# 5. Fill with mode (categorical)
df["department"].fillna(df["department"].mode()[0], inplace=True)

# 6. Forward fill (for time-series data)
df["temperature"].fillna(method="ffill")
```

**When to use median vs mean?** If a column has outliers (e.g., salary with a few billionaires), use median — it's not affected by extremes.

---

### 2. Removing Duplicates

```python
df.duplicated().sum()              # count duplicate rows
df[df.duplicated()]                # show duplicate rows
df.drop_duplicates(inplace=True)   # remove all duplicates

# Duplicates on specific columns only
df.drop_duplicates(subset=["email"])          # keep first occurrence
df.drop_duplicates(subset=["email"], keep="last")
```

---

### 3. Fixing Data Types

```python
df.dtypes   # check types

# String → numeric
df["age"] = pd.to_numeric(df["age"], errors="coerce")
# errors="coerce" turns unconvertible values (like "N/A") into NaN

# String → datetime
df["signup_date"] = pd.to_datetime(df["signup_date"])
df["year"] = df["signup_date"].dt.year
df["month"] = df["signup_date"].dt.month

# Numeric → category (saves memory, faster groupby)
df["department"] = df["department"].astype("category")

# Boolean stored as 0/1
df["active"] = df["active"].astype(bool)
```

---

### 4. Standardizing Categorical Values

Inconsistent categories break groupby and one-hot encoding:

```python
df["status"].value_counts()
# Active      45
# active      12    ← same thing, different case
# ACTIVE       3    ← same
# Inactive    20

# Fix: lowercase + strip whitespace
df["status"] = df["status"].str.lower().str.strip()

# Fix typos / map values
df["status"] = df["status"].replace({
    "actve":    "active",
    "inactve":  "inactive",
    "yes":      "active",
    "no":       "inactive"
})
```

---

### 5. Detecting & Handling Outliers

```python
# Method 1 — IQR (Interquartile Range) — robust, recommended
Q1 = df["salary"].quantile(0.25)
Q3 = df["salary"].quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

outliers = df[(df["salary"] < lower) | (df["salary"] > upper)]
df_clean = df[(df["salary"] >= lower) & (df["salary"] <= upper)]

# Method 2 — Z-score (assumes normal distribution)
from scipy import stats
z_scores = np.abs(stats.zscore(df["salary"].dropna()))
df_clean = df[z_scores < 3]   # keep rows within 3 std devs

# Method 3 — Cap instead of remove (winsorization)
df["salary"] = df["salary"].clip(lower=lower, upper=upper)
```

---

### 6. Feature Consistency Checks

```python
# Logical validation
invalid_age = df[df["age"] < 0]
impossible  = df[df["experience"] > df["age"]]    # can't work more years than you've lived

# Range checks
df.loc[df["age"] < 18, "age"] = np.nan            # nullify impossible values
df.loc[df["salary"] < 0, "salary"] = 0
```

---

### Cleaning Pipeline Pattern

```python
def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()                                      # never mutate original

    # 1. Drop useless columns
    df.drop(columns=["id", "remarks"], errors="ignore", inplace=True)

    # 2. Remove duplicates
    df.drop_duplicates(inplace=True)

    # 3. Fix types
    df["age"]    = pd.to_numeric(df["age"], errors="coerce")
    df["salary"] = pd.to_numeric(df["salary"], errors="coerce")

    # 4. Standardize categories
    df["dept"] = df["dept"].str.lower().str.strip()

    # 5. Fill missing values
    df["age"].fillna(df["age"].median(), inplace=True)
    df["salary"].fillna(df["salary"].mean(), inplace=True)
    df["dept"].fillna("unknown", inplace=True)

    # 6. Cap outliers
    for col in ["age", "salary"]:
        Q1, Q3 = df[col].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        df[col] = df[col].clip(Q1 - 1.5*IQR, Q3 + 1.5*IQR)

    return df
```

**Backend analogy:** This is a validation + transformation filter chain — exactly like a Spring `OncePerRequestFilter` pipeline.

---

## 🔑 Key Terms

| Term | Meaning |
|---|---|
| **NaN** | Missing value marker in Pandas/NumPy |
| **Imputation** | Replacing missing values with a computed value (mean, median, mode) |
| **Outlier** | A data point far from the rest — may be an error or a genuine extreme |
| **IQR** | Interquartile Range — Q3 minus Q1; measures spread of middle 50% |
| **Winsorization** | Capping outliers at a threshold instead of removing them |
| **One-hot encoding** | Converting categorical column to binary columns (Day 25 topic) |
| **Data leakage** | Using test data info during training — cleaning must be fitted on train only |

---

## 💻 Code Exercise

Create `day05.ipynb`:

```python
import pandas as pd
import numpy as np

# Dirty dataset
data = {
    "name":       ["Alice", "Bob", "Carol", "Dave", "Alice", "Eve", "Frank"],
    "age":        [28, "45", None, -5, 28, 33, 999],
    "salary":     [75000, 60000, None, 52000, 75000, "N/A", 85000],
    "department": ["Engineering", "SALES", "engineering", "Sales", "Engineering", "hr", None],
    "email":      ["a@x.com", "b@x.com", "c@x.com", "d@x.com", "a@x.com", "e@x.com", "f@x.com"],
    "experience": [3, 20, 5, 2, 3, 8, 50]
}
df = pd.DataFrame(data)

# Task 1 — Audit the data
# a) Print dtypes and null counts
# b) Identify duplicate rows
# c) Find impossible values (age < 0, age > 100, experience > age)

# Task 2 — Fix types
# a) Convert "age" and "salary" to numeric (coerce errors)
# b) Standardize "department" to lowercase + strip

# Task 3 — Handle missing & bad values
# a) Nullify impossible ages (< 0 or > 100) and extreme experience (> 40)
# b) Fill missing salary with median
# c) Fill missing department with "unknown"
# d) Drop duplicate rows (keep first)

# Task 4 — Outlier detection
# a) Use IQR method to find salary outliers
# b) Cap salaries within IQR bounds (winsorize)
# c) Print before/after salary distribution (min, max, mean)

# Task 5 — Write the clean_dataframe() function
# Wrap all your steps into a reusable function
# Apply it to df and confirm the output is clean
```

---

## 🔥 Mini Challenge

```python
# Download the Titanic dataset (or use this snippet to create a similar one)
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
titanic = pd.read_csv(url)

# 1. Audit: null counts and % for every column
# 2. Drop column "Cabin" (>75% missing — useless)
# 3. Fill "Age" with median grouped by "Pclass" (better than global median)
#    Hint: df.groupby("Pclass")["Age"].transform("median")
# 4. Fill "Embarked" with mode
# 5. Drop any remaining rows with nulls
# 6. How many rows and columns remain after cleaning?
```

---

## ❓ Interview Questions

1. **Your model has 85% accuracy but performs poorly in production. Data cleaning was done on the full dataset. What went wrong?** *(data leakage)*
2. **When would you use `median` instead of `mean` to fill missing values?**
3. **You have a categorical column "city" with 200 unique values. 30% of rows are missing. What are your options?**
4. **What is the IQR method for outlier detection? Walk me through it.**
5. **You have 1 million rows and 50 features. `df.isnull().sum()` shows feature_X has 60% missing values. What do you do?**

---

## 📝 Summary

- Always audit first: `df.info()`, `df.describe()`, `df.isnull().sum()`, `df.duplicated().sum()`
- Missing values: drop (when rare), impute mean/median (numeric), mode (categorical)
- Use `pd.to_numeric(errors="coerce")` to safely convert strings with garbage values
- Standardize categoricals: `.str.lower().str.strip()` before groupby or encoding
- IQR outlier detection is robust and the industry standard
- Wrap cleaning in a function — ML pipelines must be reproducible
- Critical rule: fit imputation stats (mean, median) on training data only — apply to test

---

## ✅ Done Checklist
- [ ] All 5 tasks completed in `day05.ipynb`
- [ ] `clean_dataframe()` function written and tested
- [ ] Titanic mini challenge completed
- [ ] Can explain why you fit imputation on train set only
- [ ] Close GitHub issue #16 when done

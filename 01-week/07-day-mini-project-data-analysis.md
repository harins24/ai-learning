# Day 7 — Mini Project: Data Analysis

**Phase:** Foundations | **Week:** 1 | **GitHub Issue:** #18

---

## 🎯 Objective
Apply everything from Days 1–6 in one end-to-end data analysis project. Load a real dataset, clean it, explore it with Pandas, perform matrix operations with NumPy, and extract meaningful insights. Commit the notebook to your GitHub repo.

---

## 🧠 Project Brief

You will analyse the **Titanic survival dataset** — a classic ML dataset that contains passenger information from the 1912 Titanic disaster. Your job today is **not** to build a model — it is to deeply understand the data through exploratory data analysis (EDA).

> EDA is what separates engineers who deploy working models from those who deploy confidently wrong ones. Every ML project starts here.

**What you will produce:**
- A cleaned DataFrame ready for ML
- 8 analytical insights with supporting evidence
- A reusable `clean_titanic()` function
- All work in `day07.ipynb` committed to `harins24/ai-learning`

---

## 📂 Dataset

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)
df.head()
```

**Columns:**

| Column | Description |
|---|---|
| `Survived` | 0 = died, 1 = survived (our target label) |
| `Pclass` | Ticket class: 1 = 1st, 2 = 2nd, 3 = 3rd |
| `Name` | Passenger name |
| `Sex` | male / female |
| `Age` | Age in years |
| `SibSp` | # siblings/spouses aboard |
| `Parch` | # parents/children aboard |
| `Ticket` | Ticket number |
| `Fare` | Passenger fare |
| `Cabin` | Cabin number |
| `Embarked` | Port: C=Cherbourg, Q=Queenstown, S=Southampton |

---

## 🔨 Step-by-Step Guide

### Step 1 — Audit the Raw Data

```python
print("Shape:", df.shape)
print("\nDtypes:\n", df.dtypes)
print("\nNull counts:\n", df.isnull().sum())
print("\nNull %:\n", (df.isnull().mean() * 100).round(2))
print("\nDescribe:\n", df.describe())
print("\nSurvived value counts:\n", df["Survived"].value_counts())
```

Record your findings:
- How many rows and columns?
- Which columns have missing values? What %?
- What is the survival rate (% survived)?

---

### Step 2 — Clean the Data

```python
def clean_titanic(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1. Drop irrelevant columns
    df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"], inplace=True)

    # 2. Fill missing Age with median by Pclass (smarter than global median)
    df["Age"] = df.groupby("Pclass")["Age"].transform(
        lambda x: x.fillna(x.median())
    )

    # 3. Fill missing Embarked with mode
    df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)

    # 4. Drop any remaining nulls
    df.dropna(inplace=True)

    # 5. Encode Sex as binary (male=0, female=1)
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

    # 6. Encode Embarked as integers
    df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})

    return df

df_clean = clean_titanic(df)
print("Clean shape:", df_clean.shape)
print(df_clean.head())
```

---

### Step 3 — Exploratory Analysis (8 Insights)

Answer each question with code + a one-line conclusion:

```python
# Insight 1: What was the overall survival rate?
survival_rate = df["Survived"].mean() * 100
print(f"Overall survival rate: {survival_rate:.1f}%")
```

```python
# Insight 2: Did gender affect survival? (women & children first?)
gender_survival = df.groupby("Sex")["Survived"].mean() * 100
print(gender_survival)
# Conclusion: ___
```

```python
# Insight 3: Did ticket class affect survival?
class_survival = df.groupby("Pclass")["Survived"].mean() * 100
print(class_survival)
# Conclusion: ___
```

```python
# Insight 4: Age distribution of survivors vs non-survivors
survivors     = df[df["Survived"] == 1]["Age"].dropna()
non_survivors = df[df["Survived"] == 0]["Age"].dropna()

print(f"Survivors    — mean age: {survivors.mean():.1f}, median: {survivors.median():.1f}")
print(f"Non-survivors— mean age: {non_survivors.mean():.1f}, median: {non_survivors.median():.1f}")
# Conclusion: ___
```

```python
# Insight 5: Did fare correlate with survival?
fare_survival = df.groupby("Survived")["Fare"].mean()
print(fare_survival)
# Conclusion: ___
```

```python
# Insight 6: Family size effect (create FamilySize = SibSp + Parch + 1)
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
family_survival = df.groupby("FamilySize")["Survived"].mean() * 100
print(family_survival.sort_index())
# Conclusion: solo travellers vs families — who survived more?
```

```python
# Insight 7: Port of embarkation survival rates
port_survival = df.groupby("Embarked")["Survived"].mean() * 100
print(port_survival)
# Conclusion: ___
```

```python
# Insight 8: Pclass + Gender combined — most/least likely to survive
combo = df.groupby(["Pclass", "Sex"])["Survived"].mean() * 100
print(combo.unstack())
# Conclusion: which group had the highest/lowest survival rate?
```

---

### Step 4 — NumPy Analysis

```python
# Use the cleaned DataFrame for matrix operations
X = df_clean[["Pclass", "Age", "Fare", "SibSp", "Parch"]].values  # NumPy array

# a) Shape of feature matrix
print("Feature matrix shape:", X.shape)

# b) Mean and std of each feature (column-wise)
print("Feature means:", X.mean(axis=0).round(2))
print("Feature stds: ", X.std(axis=0).round(2))

# c) Normalize X (standardize each feature to mean=0, std=1)
X_norm = (X - X.mean(axis=0)) / X.std(axis=0)
print("Normalized means:", X_norm.mean(axis=0).round(4))  # should be ~0
print("Normalized stds: ", X_norm.std(axis=0).round(4))   # should be ~1

# d) Correlation matrix using NumPy
corr_matrix = np.corrcoef(X_norm.T)   # shape (5, 5)
print("Correlation matrix shape:", corr_matrix.shape)
print(np.round(corr_matrix, 2))
```

---

### Step 5 — Visualise (Optional but Recommended)

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Survival by sex
df.groupby("Sex")["Survived"].mean().plot(kind="bar", ax=axes[0], title="Survival by Sex")

# Survival by class
df.groupby("Pclass")["Survived"].mean().plot(kind="bar", ax=axes[1], title="Survival by Class")

# Age distribution
axes[2].hist(survivors, alpha=0.6, label="Survived", bins=20)
axes[2].hist(non_survivors, alpha=0.6, label="Died", bins=20)
axes[2].set_title("Age Distribution")
axes[2].legend()

plt.tight_layout()
plt.show()
```

---

### Step 6 — Summary Table

At the end of your notebook, write a markdown cell with your findings:

```markdown
## Key Findings

| Insight | Finding |
|---------|---------|
| Overall survival | X% survived |
| Gender | Women survived at X%, men at Y% |
| Class | 1st class: X%, 2nd: Y%, 3rd: Z% |
| Age | Survivors were slightly younger/older |
| Fare | Higher fare → higher survival |
| Family size | Solo travellers fared worse/better |
| Port | C had highest survival rate |
| Class + Gender | 1st class female: X% vs 3rd class male: Y% |
```

---

## ❓ Week 1 Review Questions

1. **You load a CSV and find `df["age"].dtype` is `object` instead of `int64`. What likely happened and how do you fix it?**
2. **What does `df.groupby("Pclass")["Age"].transform("median")` do differently from `df.groupby("Pclass")["Age"].mean()`?**
3. **You have a matrix of shape `(891, 5)`. You do `X.mean(axis=0)`. What shape is the result and what does each value represent?**
4. **What is the shape rule for matrix multiplication? Write one ML example.**
5. **You normalized your data using `mean` and `std` computed from the full dataset including test data. What is the problem?**

---

## 📝 Week 1 Summary

You now have the foundation tools of every ML engineer:

| Day | Skill | Where you'll use it |
|---|---|---|
| 1 | Python setup + syntax | Every day |
| 2 | Data structures | Data pipelines, configs |
| 3 | NumPy arrays | Feature matrices, math ops |
| 4 | Pandas DataFrames | Loading and exploring data |
| 5 | Data cleaning | Every real project |
| 6 | Linear algebra | Understanding every ML algorithm |
| 7 | EDA project | Starting every ML project |

**Coming in Week 2:** Probability, statistics, and the mathematical intuition behind ML.

---

## ✅ Done Checklist
- [ ] `day07.ipynb` created with all 6 steps complete
- [ ] 8 insights answered with conclusions
- [ ] `clean_titanic()` function works correctly
- [ ] NumPy analysis section complete
- [ ] Notebook committed to `harins24/ai-learning` repo
- [ ] Close GitHub issue #18 when done
- [ ] Close parent GitHub issue #1 (Week 1 complete!)

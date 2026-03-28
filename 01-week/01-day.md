# Day 1 — Setup Python + Jupyter

**Phase:** Foundations | **Week:** 1 | **GitHub Issue:** #12

---

## 🎯 Objective
Set up a working Python + Jupyter environment so every future lesson has a consistent, reproducible workspace. You will also write your first Python data snippet to verify everything works.

---

## 🧠 Concept Explanation

### Why Python for AI?
As a Java developer you are used to compiled, statically-typed, verbose code. Python trades that for:

| Java mindset | Python equivalent |
|---|---|
| `int[] arr = new int[]{1,2,3};` | `arr = [1, 2, 3]` |
| Compile → Run | Just run |
| Maven / Gradle ecosystem | pip / conda ecosystem |
| JVM | CPython interpreter |

The entire AI/ML ecosystem (NumPy, Pandas, PyTorch, scikit-learn, Hugging Face) is Python-first. Java has some ML libraries, but 99% of research and tooling happens in Python.

### What is Jupyter?
Think of Jupyter Notebook as **Postman for data science**:
- Instead of testing an API endpoint, you test a piece of code and see the result inline.
- Each **cell** is like a mini-function — run it independently, inspect the output, tweak, re-run.
- The notebook keeps your code, output, charts, and notes in one `.ipynb` file.

This is perfect for learning because you can experiment without restarting from scratch every time.

### Anatomy of a Jupyter Notebook
```
[ Cell 1 ] → Code cell  → runs Python, shows output below
[ Cell 2 ] → Markdown   → documentation / notes
[ Cell 3 ] → Code cell  → can use variables from Cell 1
```
Unlike a Java class where everything compiles together, cells share a single **kernel** (a running Python process). Order of execution matters.

---

## 🔑 Key Terms

| Term | Meaning |
|---|---|
| **Python** | Interpreted, dynamically-typed language. The lingua franca of AI. |
| **pip** | Python's package manager (like Maven for Python). |
| **conda** | Alternative package + environment manager by Anaconda. |
| **Virtual environment** | Isolated Python installation per project (like separate JVMs per app). |
| **Jupyter Notebook** | Browser-based interactive coding environment (`.ipynb` files). |
| **JupyterLab** | Next-gen Jupyter UI — tabs, file browser, terminal all in one. |
| **Kernel** | The running Python process behind a notebook. |
| **Cell** | A single executable unit in a notebook (code or markdown). |

---

## ⚙️ Setup Steps

### Option A — Anaconda (Recommended for beginners)
```bash
# 1. Download from https://www.anaconda.com/download
#    Includes Python + Jupyter + 250 data science packages out of the box

# 2. Verify installation
python --version       # should be 3.10+
jupyter --version

# 3. Launch JupyterLab
jupyter lab
```

### Option B — pip + venv (Lightweight)
```bash
# 1. Install Python from https://python.org (3.10+ recommended)

# 2. Create a virtual environment (like a project-scoped JVM)
python -m venv ai-env

# 3. Activate it
# Windows:
ai-env\Scripts\activate
# Mac/Linux:
source ai-env/bin/activate

# 4. Install Jupyter + essentials
pip install jupyterlab numpy pandas matplotlib

# 5. Launch
jupyter lab
```

### Option C — VS Code (If you prefer IDE)
- Install the **Jupyter** extension in VS Code
- Open any `.ipynb` file and run cells inline
- Same experience, no browser needed

---

## 💻 Code Exercise

Create a new notebook called `day01.ipynb` and run each cell:

```python
# Cell 1 — Verify Python version
import sys
print(f"Python version: {sys.version}")
```

```python
# Cell 2 — Your first data structures
# As a Java dev, think: ArrayList, HashMap, Set, Tuple(record)
my_list  = [10, 20, 30, 40, 50]        # like ArrayList<Integer>
my_dict  = {"name": "Hari", "days": 75} # like HashMap<String, Object>
my_tuple = (1, 2, 3)                    # like an immutable record
my_set   = {1, 2, 2, 3}                # like HashSet — no duplicates

print("List :", my_list)
print("Dict :", my_dict)
print("Tuple:", my_tuple)
print("Set  :", my_set)   # prints {1, 2, 3} — duplicate 2 removed
```

```python
# Cell 3 — List comprehension (Streams in one line)
# Java: list.stream().map(x -> x * 2).collect(Collectors.toList())
squares = [x**2 for x in range(1, 6)]
evens   = [x for x in range(1, 11) if x % 2 == 0]

print("Squares:", squares)  # [1, 4, 9, 16, 25]
print("Evens  :", evens)    # [2, 4, 6, 8, 10]
```

```python
# Cell 4 — Simple function
def greet_student(name: str, day: int) -> str:
    return f"Welcome {name}! Starting Day {day} of your AI journey."

print(greet_student("Hari", 1))
```

**Expected output:**
```
Python version: 3.x.x ...
List : [10, 20, 30, 40, 50]
Dict : {'name': 'Hari', 'days': 75}
Tuple: (1, 2, 3)
Set  : {1, 2, 3}
Squares: [1, 4, 9, 16, 25]
Evens  : [2, 4, 6, 8, 10]
Welcome Hari! Starting Day 1 of your AI journey.
```

---

## 🔥 Mini Challenge

Without running it first, predict the output of this code. Then run it and check:

```python
data = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3]

unique_sorted = sorted(set(data))
doubled = [x * 2 for x in unique_sorted if x > 3]

print(unique_sorted)
print(doubled)
```

**Questions to answer:**
1. What does `set(data)` do to the duplicates?
2. Why does `sorted()` return a list even though `set` has no order?
3. What values pass the `if x > 3` filter?

---

## ❓ Interview Questions

1. **What is a virtual environment in Python and why would you use one?** *(Hint: think about dependency conflicts across projects)*
2. **What is the difference between a list and a tuple in Python?**
3. **What does a Jupyter kernel represent? What happens when you restart it?**
4. **In Python, what does `[x**2 for x in range(5)]` produce?**
5. **You have a Java background — what is the Python equivalent of `HashMap<String, Integer>`?**

---

## 📝 Summary

- Python is the standard language for AI/ML — concise, interpreted, massive ecosystem
- Jupyter lets you run code interactively in cells — ideal for experimentation
- Virtual environments isolate dependencies per project (like separate classpaths)
- Core Python types: `list`, `dict`, `tuple`, `set` — you already know their Java equivalents
- List comprehensions replace verbose `for` loops and Java Streams

---

## ✅ Done Checklist
- [ ] Python 3.10+ installed and verified
- [ ] JupyterLab / VS Code Jupyter running
- [ ] `day01.ipynb` created and all 4 cells execute without error
- [ ] Mini challenge completed and explained
- [ ] Close GitHub issue #12 when done

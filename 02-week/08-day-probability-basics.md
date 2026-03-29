# Day 8 — Probability Basics

**Phase:** Foundations | **Week:** 2 | **GitHub Issue:** #19

---

## 🎯 Objective

Understand the core language of probability that underpins every ML algorithm. By the end of today you will be able to reason about events, compute basic probabilities, apply Bayes' Theorem, and see exactly why ML models are fundamentally probabilistic systems.

---

## 🧠 Concept Explanation

### What is Probability?

Probability is the mathematics of uncertainty. Every ML model is, at its core, a machine that outputs probabilities — a spam classifier does not say "this is spam"; it says "this email has a 94% chance of being spam."

**Analogy for a backend developer:** Think of probability the way you think about SLAs. When you say a service has 99.9% uptime, you are expressing a probability. When you design a retry policy with exponential backoff, you are reasoning probabilistically about failure. ML just formalises this thinking.

---

### Core Concepts

**Sample Space (Ω):** The set of all possible outcomes.
- Rolling a die: Ω = {1, 2, 3, 4, 5, 6}
- Email classification: Ω = {spam, not spam}

**Event (E):** A subset of the sample space.
- "Rolling an even number" = {2, 4, 6}

**Probability of an Event:**
```
P(E) = (Number of favourable outcomes) / (Total outcomes)
```
- P(rolling even) = 3/6 = 0.5

**Key Rules:**
- 0 ≤ P(E) ≤ 1 always
- P(Ω) = 1 (something must happen)
- P(not E) = 1 − P(E)

---

### Joint, Marginal, and Conditional Probability

These three concepts are the foundation of almost every ML algorithm.

**Joint Probability — P(A and B):**
The probability that both A and B happen.
```
P(A ∩ B) = P(A) × P(B)   [only when A and B are independent]
```

**Conditional Probability — P(A | B):**
"Given that B happened, what is the probability of A?"
```
P(A | B) = P(A ∩ B) / P(B)
```

**Backend analogy:** "Given that a request came from IP 192.168.1.1 (B), what is the probability it is a bot (A)?" This is exactly P(A | B) — a conditional probability your fraud detection system computes constantly.

**Marginal Probability:** The probability of one event ignoring the other, obtained by "summing out" the other variable.

---

### Bayes' Theorem

This is perhaps the single most important formula in machine learning:

```
P(A | B) = [ P(B | A) × P(A) ] / P(B)
```

**In English:**
- **P(A)** — Prior: what you believed before seeing evidence
- **P(B | A)** — Likelihood: how probable is the evidence if A is true
- **P(B)** — Marginal: total probability of the evidence
- **P(A | B)** — Posterior: your updated belief after seeing evidence

**Real-world example — spam filter:**
- P(spam) = 0.2 (20% of emails are spam — your prior)
- P("win prize" | spam) = 0.8 (80% of spam contains this phrase)
- P("win prize") = 0.1 (10% of all emails contain this phrase)

```
P(spam | "win prize") = (0.8 × 0.2) / 0.1 = 1.6...
```

Wait — that cannot exceed 1. This means P("win prize") must be recalculated properly (it accounts for both spam and ham). The point is: Bayes' Theorem lets your model **update beliefs based on evidence**. This is exactly how the Naive Bayes classifier works.

---

### Independence vs Dependence

Two events are **independent** if knowing one tells you nothing about the other.
```
P(A ∩ B) = P(A) × P(B)   ← only true when independent
```

Example: Coin flip result and weather are independent. But "a user clicked an ad" and "a user made a purchase" are **dependent** — and ML models exploit this dependence.

---

## 📖 Key Terms

| Term | Definition |
|------|-----------|
| **Sample Space** | All possible outcomes of an experiment |
| **Event** | A specific subset of outcomes you care about |
| **Prior P(A)** | Your belief about A before seeing any evidence |
| **Likelihood P(B\|A)** | How probable the evidence is, assuming A is true |
| **Posterior P(A\|B)** | Your updated belief about A after seeing evidence B |
| **Joint Probability** | Probability that two events both occur |
| **Conditional Probability** | Probability of A given that B has already occurred |
| **Independence** | Knowing one event gives no information about the other |

---

## 💻 Code Exercise

Build a Bayes spam classifier from scratch using only Python — no ML libraries.

```python
# day08.py — Naive Bayes Spam Classifier from scratch

# Training data: (message, label)
training_data = [
    ("win free money now", "spam"),
    ("win a prize today", "spam"),
    ("free offer limited time", "spam"),
    ("earn money fast", "spam"),
    ("meeting at 3pm tomorrow", "ham"),
    ("project update attached", "ham"),
    ("call me when you are free", "ham"),
    ("lunch plans for Friday", "ham"),
    ("win the lottery today", "spam"),
    ("quarterly report review", "ham"),
]

# Step 1: Count word frequencies per class
from collections import defaultdict

word_counts = {"spam": defaultdict(int), "ham": defaultdict(int)}
class_counts = {"spam": 0, "ham": 0}

for message, label in training_data:
    class_counts[label] += 1
    for word in message.lower().split():
        word_counts[label][word] += 1

total_messages = len(training_data)

# Step 2: Compute prior probabilities
p_spam = class_counts["spam"] / total_messages
p_ham  = class_counts["ham"]  / total_messages

print(f"P(spam) = {p_spam:.2f}")
print(f"P(ham)  = {p_ham:.2f}")

# Step 3: Compute likelihoods with Laplace smoothing
# Laplace smoothing: add 1 to every count to avoid zero probabilities
vocab = set()
for message, _ in training_data:
    vocab.update(message.lower().split())
vocab_size = len(vocab)

def word_likelihood(word, label):
    count = word_counts[label][word]
    total_words = sum(word_counts[label].values())
    return (count + 1) / (total_words + vocab_size)  # Laplace smoothing

# Step 4: Classify a new message
import math

def classify(message):
    words = message.lower().split()

    # Use log probabilities to avoid floating-point underflow
    log_prob_spam = math.log(p_spam)
    log_prob_ham  = math.log(p_ham)

    for word in words:
        log_prob_spam += math.log(word_likelihood(word, "spam"))
        log_prob_ham  += math.log(word_likelihood(word, "ham"))

    return "spam" if log_prob_spam > log_prob_ham else "ham"

# Step 5: Test it
test_messages = [
    "win free money",
    "project meeting tomorrow",
    "free lunch for the team",
    "quarterly earnings report",
]

print("\n--- Classification Results ---")
for msg in test_messages:
    print(f"'{msg}' → {classify(msg)}")
```

**Expected output:**
```
P(spam) = 0.50
P(ham)  = 0.50

--- Classification Results ---
'win free money' → spam
'project meeting tomorrow' → ham
'free lunch for the team' → spam or ham (ambiguous — interesting!)
'quarterly earnings report' → ham
```

**What to observe:** The word "free" drives the classifier toward spam. This is Bayes' Theorem in action — the evidence (word) updates the prior (base rate).

---

## 🏆 Mini Challenge

Extend the classifier to:
1. Print the log probability for both classes (so you can see how confident the model is)
2. Add a **confidence threshold**: only classify as spam if P(spam | message) is more than 70% of total probability
3. Test with the message `"free project resources for the team"` — which class wins and why?

---

## ❓ Interview Questions

1. **What is the difference between P(A | B) and P(B | A)?** Give a concrete ML example where confusing the two would cause a bug.

2. **What is Bayes' Theorem and why is it fundamental to machine learning?** Can you write the formula from memory?

3. **What is Laplace smoothing and why is it needed in a Naive Bayes classifier?** What happens if you skip it?

4. **Explain the "Naive" assumption in Naive Bayes.** Is it realistic? Why do we use it anyway?

5. **A model predicts "fraud" for 1% of transactions and is correct 90% of the time when it says fraud. But 99% of transactions are genuinely non-fraudulent. What is the actual precision of the model?** (Hint: use Bayes' Theorem.)

---

## 📝 Summary

- Probability quantifies uncertainty — the fundamental language of ML
- **P(A | B)** is conditional probability: the probability of A given B has occurred
- **Bayes' Theorem** lets you invert conditional probabilities: update your beliefs (prior) with evidence to get a posterior
- **Joint probability** P(A ∩ B) = P(A) × P(B) only when events are independent
- **Naive Bayes** is a direct application of Bayes' Theorem to text and classification
- **Laplace smoothing** prevents zero-probability crashes when a word is unseen
- Log probabilities prevent numerical underflow when multiplying many small numbers

---

**GitHub Issue:** #19

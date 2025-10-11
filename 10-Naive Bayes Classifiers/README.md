# Naive Bayes Classifiers — Full Python Implementation

This repository contains a complete and from-scratch implementation of **Naive Bayes Classifiers** in Python, covering:

- Gaussian Naive Bayes (for continuous data)
- Multinomial Naive Bayes (for count/frequency data)
- Bernoulli Naive Bayes (for binary features)

It also includes **comparisons with scikit-learn**, **evaluation metrics**, and **decision boundary visualizations**.

---

## 🧠 Features

- Mathematical foundations based on Bayes’ theorem
- Conditional independence assumption
- Implementations from scratch using only NumPy
- Comparison with `scikit-learn` versions
- Evaluation on Iris and Digits datasets
- Decision boundary visualization for 2D synthetic data
- Structured `main()` function for easy execution

---

## 📦 Requirements

Make sure you have Python ≥ 3.8 installed.  
Then install the dependencies:

```bash
pip install numpy matplotlib scikit-learn
```

---

## 🚀 Run the Program

To execute all Naive Bayes models and see results + visualizations:

```bash
python main.py
```

This will:
- Train Gaussian, Multinomial, and Bernoulli NB (from scratch + sklearn)
- Display accuracy and confusion matrices
- Plot decision boundaries for synthetic datasets

---

## 📊 Example Output

```
=== Gaussian Naive Bayes (Iris Dataset) ===
Scratch Accuracy: 0.9777
Sklearn Accuracy: 0.9777

=== Multinomial Naive Bayes (Digits Dataset) ===
Scratch Accuracy: 0.8351
Sklearn Accuracy: 0.8342
```

Decision boundaries will also open in separate matplotlib windows.

---

## 🧩 File Structure

```
main.py            # Main executable with all implementations and visualizations
README.md          # This file
```

---

## 🧠 Theory Recap

Naive Bayes is derived from Bayes’ theorem:

\[
P(y \mid \mathbf{x}) = \frac{P(\mathbf{x} \mid y) P(y)}{P(\mathbf{x})}
\]

The **Naive** assumption states that features are conditionally independent given the class:

\[
P(\mathbf{x} \mid y) = \prod_j P(x_j \mid y)
\]

This simplification makes Naive Bayes extremely efficient and robust for high-dimensional data.

---

## 🧰 Author

Developed by **Reza Mousavi (MECHATEK)**  
For educational and research purposes in machine learning and pattern recognition.


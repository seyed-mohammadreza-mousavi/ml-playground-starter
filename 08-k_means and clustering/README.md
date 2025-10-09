# 🧠 NumPy K-Means Implementation

A **from-scratch implementation** of the K-Means clustering algorithm using **NumPy**, including a **numerically stable K-Means++ initializer**, and comparison with **scikit-learn's KMeans**.

---

## 📘 Overview

This project demonstrates how K-Means works internally — without relying on deep learning or ML libraries — focusing on **algorithmic understanding** and **numerical stability**.

### ✨ Features

- K-Means clustering from scratch using **NumPy**
- Stable **K-Means++ initialization**
- Comparison with **scikit-learn**
- Synthetic dataset visualization (using `make_blobs`)
- Automatic detection of convergence via inertia (WCSS)

---

## 🧩 File Structure

```
.
├── kmeans_numpy_main.py   # main Python script with implementation and demo
├── README.md              # this file
```

---

## ⚙️ Installation

### Requirements
```bash
pip install numpy matplotlib scikit-learn
```

---

## 🚀 Usage

Simply run the main file:

```bash
python kmeans_numpy_main.py
```

This will:
1. Generate synthetic data (`make_blobs`)
2. Run K-Means (NumPy version)
3. Compare results with scikit-learn’s KMeans
4. Plot both cluster assignments and centroids

---

## 🧮 Algorithm Summary

### Objective Function
The K-Means objective minimizes the within-cluster sum of squares (WCSS):

$$
J(\{\mu_j\}, \{c_i\}) = \sum_{i=1}^n \lVert x_i - \mu_{c_i} \rVert_2^2
$$

### Update Steps
- **Assignment step (E-step):**

$$
c_i \leftarrow \arg\min_{j} \, \lVert x_i - \mu_j \rVert_2^2
$$

- **Update step (M-step):**
  $$
  \mu_j \leftarrow \frac{1}{|C_j|} \sum_{i:c_i=j} x_i
  $$

### K-Means++ Initialization
Improves centroid initialization using weighted sampling based on squared distances.

---

## 📊 Example Output

When you run the script, you will see two figures:

1. **NumPy K-Means (from scratch)** — cluster assignments and centroids  
2. **scikit-learn KMeans** — for comparison

And console output similar to:
```
NumPy K-Means -> inertia: 2465.32, iterations: 11
sklearn KMeans -> inertia: 2465.32, iterations: 11
```

---

## 🧠 Notes on Stability

- Small floating-point rounding errors in squared distances can lead to negative values during K-Means++ initialization.  
  This implementation fixes that using:
  ```python
  d2 = np.maximum(d2, 0.0)
  ```

- Empty clusters are reinitialized randomly for robustness.

---

## 🧑‍💻 Author

Created by **ChatGPT (GPT‑5)** — for educational and research purposes.  
You are free to modify, extend, and use it for learning.

---

## 📜 License

MIT License — free to use and modify.

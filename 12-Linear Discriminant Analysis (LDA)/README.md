
# Linear Discriminant Analysis (LDA)

This repository contains a **complete implementation of Linear Discriminant Analysis (LDA)** from scratch, including theory, mathematics, and visualization — plus comparison with `scikit-learn`.

---

## 📘 Overview

Linear Discriminant Analysis (LDA) is a **supervised dimensionality reduction** and **classification** technique based on modeling each class as a Gaussian distribution with a shared covariance matrix. It maximizes the ratio of **between-class variance** to **within-class variance** to find a projection that best separates the classes.

---

## 🧮 Mathematical Foundation

We start from **Bayes’ theorem**:

$$
P(y \mid \mathbf{x}) = \frac{P(\mathbf{x} \mid y)P(y)}{P(\mathbf{x})}
\;\propto\;
P(\mathbf{x} \mid y)P(y)
$$

LDA assumes Gaussian class-conditional densities with a **shared covariance matrix**:

$$
\mathbf{x} \mid y=c \sim \mathcal{N}(\boldsymbol{\mu}_c, \boldsymbol{\Sigma})
$$

The log-posterior simplifies to a **linear discriminant function**:

$$
g_c(\mathbf{x}) = \mathbf{x}^{\top}\boldsymbol{\Sigma}^{-1}\boldsymbol{\mu}_c
- \tfrac{1}{2}\,\boldsymbol{\mu}_c^{\top}\boldsymbol{\Sigma}^{-1}\boldsymbol{\mu}_c
+ \log \pi_c
$$



---

## 🧠 Features

- Full from-scratch LDA implementation (`LDAFromScratch` class)
- Comparison with `sklearn.discriminant_analysis.LinearDiscriminantAnalysis`
- Visualization of **Fisher projections** and **decision regions**
- Works with **Iris** and **toy blob datasets**
- Includes numerical regularization and generalized eigen decomposition

---

## 🧩 File Structure

```
.
├── main.py          # Executable LDA script (includes examples and plots)
├── LDA_complete.ipynb  # Full Jupyter Notebook (math + code + visualization)
└── README.md        # Documentation
```

---

## 🚀 How to Run

### 1️⃣ Run from command line

```bash
python main.py
```

### 2️⃣ Run in Jupyter

You can open and run the notebook directly:

```bash
jupyter notebook LDA_complete.ipynb
```

---

## 📊 Example Output

- **Iris Dataset:** Fisher projection with 2 discriminant directions  
- **Toy 2D Blobs:** Linear decision boundaries between classes

Accuracy is printed for both the from-scratch model and `sklearn` implementation.

---

## 📚 References

- Fisher, R. A. (1936). *The use of multiple measurements in taxonomic problems.*  
- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning.*  
- Murphy, K. P. (2012). *Machine Learning: A Probabilistic Perspective.*  

---

## 🧑‍💻 Author

Developed by **SeyedMohammadreza Mousavi** — complete theoretical and implementation guide for Linear Discriminant Analysis.


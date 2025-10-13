
# 🧠 Support Vector Regression (SVR) — From Theory to Practice

This repository provides a **complete implementation and visualization** of **Support Vector Regression (SVR)** using Python and `scikit-learn`.  
It includes the **math behind SVR**, full **code for training, tuning, and visualizing**, and several experiments for **kernel and hyperparameter analysis**.

---

## 📘 Overview

Support Vector Regression (SVR) is the regression counterpart of Support Vector Machines (SVM).  
It finds a function $f(\mathbf{x})$ that deviates from the actual targets $y_i$ by at most $\epsilon$ for each training point, and at the same time is as flat as possible.

Flatness means minimizing the norm of the weight vector $\mathbf{w}$, leading to the following **optimization problem**:

$$ \min_{w, b, \xi, \xi^*} \tfrac{1}{2}\|w\|^2 + C \sum_{i=1}^n (\xi_i + \xi_i^*) $$


subject to the constraints:

$$ y_{i} - \mathbf{w}^\top \phi(\mathbf{x}_{i}) - b \le \epsilon + \xi_{i} \\ \mathbf{w}^\top \phi(\mathbf{x}_{i}) + b - y_{i} \le \epsilon + \xi_{i}^* \\ \xi_{i}, \xi_{i}^* \ge 0, \quad i=1, \dots, n $$


- $C$ — regularization parameter controlling the trade-off between flatness and tolerance to deviations  
- $\epsilon$ — defines the width of the **$\epsilon$-insensitive tube**  
- $\xi_i$, $\xi_i^*$ — slack variables for points outside the tube  

---

## ⚙️ Dual Formulation

Using Lagrange multipliers $(\alpha_i, \alpha_i^*)$, we obtain the **dual optimization problem**:

$$
\max_{\{\alpha_{i}\},\, \{\alpha_{i}^{*}\}}\;
-\tfrac{1}{2}\sum_{i=1}^{n}\sum_{j=1}^{n}(\alpha_{i}-\alpha_{i}^{*})(\alpha_{j}-\alpha_{j}^{*})\,K(\mathbf{x}_{i},\mathbf{x}_{j})
-\epsilon\sum_{i=1}^{n}(\alpha_{i}+\alpha_{i}^{*})
+\sum_{i=1}^{n}y_{i}(\alpha_{i}-\alpha_{i}^{*})\\
\text{s.t.}\;
\sum_{i=1}^{n}(\alpha_{i}-\alpha_{i}^{*})=0,\;
0\le\alpha_{i},\,\alpha_{i}^{*}\le C
$$




The regression function becomes:

$$ f(\mathbf{x}) = \sum_{i=1}^{n} (\alpha_i - \alpha_i^*) K(\mathbf{x}_i, \mathbf{x}) + b $$

---

## 📊 Implementation Summary

### 1. Data Generation
A noisy, nonlinear dataset is generated from a **sinc** function for regression experiments.

### 2. Model Training
- **RBF SVR** trained via `GridSearchCV` for hyperparameters $(C, \epsilon, \gamma)$.
- Best model selected by lowest MSE across 5-fold cross-validation.

### 3. Evaluation Metrics
For both train and test sets:
- **MAE** (Mean Absolute Error)  
- **RMSE** (Root Mean Squared Error)  
- **R² Score** (Coefficient of Determination)

### 4. Visualization
- Regression fit with **ε-tube** and **support vectors**
- Residual plots
- Comparison of **Linear**, **Polynomial**, and **RBF** kernels
- RMSE heatmap for $(C, \epsilon)$
- Support vectors vs. ε analysis

---

## 🧩 Key Equations

### ε-Insensitive Loss

$$ L_\epsilon(y, f(\mathbf{x})) = \max\{0, |y - f(\mathbf{x})| - \epsilon\} $$

### Decision Function

$$ f(\mathbf{x}) = \sum_{i=1}^{n} (\alpha_i - \alpha_i^*) K(\mathbf{x}_i, \mathbf{x}) + b $$

### Common Kernels

$$
\begin{aligned}
K_{\text{linear}}(\mathbf{x}, \mathbf{z}) &= \mathbf{x}^\top \mathbf{z} \\
K_{\text{poly}}(\mathbf{x}, \mathbf{z}) &= (\gamma \mathbf{x}^\top \mathbf{z} + r)^d \\
K_{\text{rbf}}(\mathbf{x}, \mathbf{z}) &= \exp(-\gamma \|\mathbf{x} - \mathbf{z}\|^2)
\end{aligned}
$$

---

## 🧮 Run Locally

### 1️⃣ Clone the repository
```bash
git clone https://github.com/yourusername/SVR-Tutorial.git
cd SVR-Tutorial
```

### 2️⃣ Install dependencies
```bash
pip install numpy pandas matplotlib scikit-learn
```

### 3️⃣ Run the script
```bash
python main.py
```

---

## 📈 Example Output

- Tuned **RBF-SVR** performance metrics
- Visualization of fitted regression with ε-tube
- Comparison across kernel types
- RMSE heatmap (C vs ε)
- Plot of **support vectors** vs **ε**

---

## 📚 References

- Vapnik, V. *The Nature of Statistical Learning Theory*. Springer, 1995.  
- Smola, A. & Schölkopf, B. *A Tutorial on Support Vector Regression*, Statistics and Computing (2004).  
- [Scikit-learn Documentation — SVR](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html)

---

## 🧠 Author

**SeyedMohammadreza Mousavi (Reza)**  

---

**License:** MIT  
© 2025 — All rights reserved.

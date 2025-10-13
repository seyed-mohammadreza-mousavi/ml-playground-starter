# Polynomial Regression — From Theory to Practice

This repository provides a **complete implementation and explanation of Polynomial Regression**, including both theoretical foundations and practical code (from scratch and using scikit-learn).

## 📘 Overview

Polynomial Regression models a **nonlinear relationship** between input $x$ and target $y$ by using polynomial basis functions.  
Although the resulting curve is nonlinear in $x$, it is **linear in the parameters** $\mathbf{w}$.

$$ \hat{y}(x) = \sum_{j=0}^{M} w_j x^{j} = \mathbf{w}^\top \boldsymbol{\phi}(x) $$

where the **polynomial feature vector** is:

$$ \boldsymbol{\phi}(x) = [1,\;x,\;x^2,\;\ldots,\;x^M]^\top $$

---

## ⚙️ Mathematical Foundation

Given data points $\{(x_i, y_i)\}_{i=1}^n$, we can define the **design matrix**:

$$ \mathbf{X} = 
\begin{bmatrix}
1 & x_1 & x_1^2 & \dots & x_1^M \\
1 & x_2 & x_2^2 & \dots & x_2^M \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
1 & x_n & x_n^2 & \dots & x_n^M
\end{bmatrix}
$$

The **mean squared error** (MSE) loss is:

$$ \mathcal{L}(\mathbf{w}) = \frac{1}{n} \| \mathbf{y} - \mathbf{X}\mathbf{w} \|_2^2 $$

Minimizing this loss yields the **normal equation**:

$$ \mathbf{w}^* = (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{y} $$

### Regularized (Ridge) Version

To reduce overfitting, we can add an $\ell_2$ regularization term:

$$ \mathcal{L}_\lambda(\mathbf{w}) = \frac{1}{n} \| \mathbf{y} - \mathbf{X}\mathbf{w} \|_2^2 + \lambda \|\mathbf{w}\|_2^2 $$

which leads to the **ridge regression solution**:

$$ \mathbf{w}_\lambda^* = (\mathbf{X}^\top \mathbf{X} + n\lambda \mathbf{I})^{-1} \mathbf{X}^\top \mathbf{y} $$

---

## 🧠 Bias–Variance Tradeoff

- Low-degree polynomial → **Underfitting** (high bias).  
- High-degree polynomial → **Overfitting** (high variance).  
Use **cross-validation** to find the optimal degree and regularization strength.

---

## 🧩 Implementation Structure

### Files

| File | Description |
|------|--------------|
| `Polynomial_Regression.ipynb` | Complete Jupyter Notebook: theory, math, implementation, and plots. |
| `main.py` | Standalone Python script that runs the full workflow. |
| `README.md` | This documentation file. |

### Requirements

Install dependencies using:

```bash
pip install numpy matplotlib scikit-learn
```

---

## 🚀 How to Run

Run the main script directly:

```bash
python main.py
```

This will:

1. Generate synthetic nonlinear data.  
2. Fit polynomial models with varying degrees.  
3. Plot training and validation curves.  
4. Demonstrate ridge regularization effects.  
5. Compare the custom and scikit-learn implementations.  

---

## 📈 Example Equations (GitHub-Compatible)

All math in this project uses **single-line LaTeX blocks** for display equations.  
For instance, Bayes’ theorem is formatted like this:

$$ P(y \mid \mathbf{x}) = \frac{P(\mathbf{x} \mid y) P(y)}{P(\mathbf{x})} \propto P(\mathbf{x} \mid y) P(y) $$

Inline math such as $x \in \mathbb{R}$ or $\mathbf{w}^\top \mathbf{x}$ is also supported.

---

## 🧮 Cross-Validation Example

We use $K$-fold CV to find the optimal degree:

$$ \text{MSE}(M) = \frac{1}{K} \sum_{k=1}^{K} \text{MSE}_k(M) $$

The best degree minimizes validation MSE.

---

## 📊 Visualization Samples

- **Scatter plots** of raw data and true function.  
- **Fitted curves** for multiple polynomial degrees.  
- **Residual plots** for diagnostic checking.  
- **Validation curves** showing overfitting trends.  

---

## 📚 References

- Bishop, *Pattern Recognition and Machine Learning*, 2006  
- Hastie, Tibshirani, and Friedman, *Elements of Statistical Learning*  
- scikit-learn documentation: [https://scikit-learn.org/](https://scikit-learn.org/)

---

## 🧾 License

This project is released under the MIT License.

---
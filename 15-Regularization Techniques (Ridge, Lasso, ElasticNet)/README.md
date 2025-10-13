
# Regularization Techniques: Ridge, Lasso, and ElasticNet

This repository demonstrates **Regularization Techniques** in linear models using Python and scikit-learn.

It includes:
- Ridge Regression (L2)
- Lasso Regression (L1)
- ElasticNet (L1 + L2)
- Visualization of coefficient shrinkage, residuals, and cross-validation errors

---

## 📘 Overview

Given a dataset $\{(\mathbf{x}_i, y_i)\}_{i=1}^n$ where $\mathbf{x}_i \in \mathbb{R}^d$ and $y_i \in \mathbb{R}$, we fit a linear model

$$ y \approx X \boldsymbol{\beta} + \epsilon $$

The **Ordinary Least Squares (OLS)** objective is:

$$ \min_{\boldsymbol{\beta}} \frac{1}{2n} \| \mathbf{y} - X\boldsymbol{\beta} \|_2^2 $$

Regularization adds a penalty term $\Omega(\boldsymbol{\beta})$ to control model complexity and prevent overfitting.

---

## 🧮 Regularization Methods

### Ridge Regression (L2)

$$ \min_{\boldsymbol{\beta}} \frac{1}{2n}\| \mathbf{y} - X\boldsymbol{\beta} \|_2^2 + \alpha \|\boldsymbol{\beta}\|_2^2 $$

- Shrinks coefficients smoothly toward zero  
- Reduces variance but not to exact zeros  
- Useful for correlated features  

**Bayesian interpretation:** Gaussian prior on $\boldsymbol{\beta}$

$$ \boldsymbol{\beta} \sim \mathcal{N}(0, \tau^2 I), \quad \alpha \propto \frac{1}{\tau^2} $$

---

### Lasso Regression (L1)

$$ \min_{\boldsymbol{\beta}} \frac{1}{2n}\| \mathbf{y} - X\boldsymbol{\beta} \|_2^2 + \alpha \|\boldsymbol{\beta}\|_1 $$

- Performs feature selection (sparse $\boldsymbol{\beta}$)  
- Some coefficients become exactly zero  

**Bayesian interpretation:** Laplace prior

$$ p(\beta_j) \propto \exp\!\left(-\frac{|\beta_j|}{b}\right), \quad \alpha \propto \frac{1}{b} $$

**Soft-thresholding update:**

$$ \beta_j \leftarrow \frac{1}{\|X_{\cdot j}\|_2^2/n} \, \mathrm{sign}(z_j) \max(|z_j| - \alpha, 0) $$

---

### ElasticNet (L1 + L2)

$$ \min_{\boldsymbol{\beta}} \frac{1}{2n}\| \mathbf{y} - X\boldsymbol{\beta} \|_2^2 + \alpha \left( \rho \|\boldsymbol{\beta}\|_1 + \tfrac{1}{2}(1 - \rho)\|\boldsymbol{\beta}\|_2^2 \right) $$

where $\rho \in [0,1]$ controls the mix between L1 and L2 penalties.

- $\rho = 1$ → Lasso  
- $\rho = 0$ → Ridge  
- Combines sparsity (L1) with stability (L2)

---

## ⚙️ Implementation

### 1️⃣ Generate synthetic data

```python
from sklearn.datasets import make_regression
X, y = make_regression(n_samples=600, n_features=8, noise=25.0, random_state=42)
```

### 2️⃣ Add polynomial features

```python
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)
```

### 3️⃣ Train/test split and scaling

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.25, random_state=42)
scaler = StandardScaler()
```

### 4️⃣ Model pipeline examples

**Ridge example:**
```python
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

ridge = Pipeline([("scaler", scaler), ("reg", Ridge(alpha=1.0))])
ridge.fit(X_train, y_train)
```

**Lasso example:**
```python
from sklearn.linear_model import Lasso

lasso = Pipeline([("scaler", scaler), ("reg", Lasso(alpha=0.1, max_iter=10000))])
lasso.fit(X_train, y_train)
```

**ElasticNet example:**
```python
from sklearn.linear_model import ElasticNet

enet = Pipeline([("scaler", scaler), ("reg", ElasticNet(alpha=0.5, l1_ratio=0.7, max_iter=10000))])
enet.fit(X_train, y_train)
```

---

## 📊 Evaluation

**RMSE** and **R²** metrics are computed as:

$$ \text{RMSE} = \sqrt{\frac{1}{n} \sum_i (y_i - \hat{y}_i)^2} $$

$$ R^2 = 1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - \bar{y})^2} $$

Cross-validation RMSE is obtained using:

```python
from sklearn.model_selection import cross_val_score

def cv_rmse(model, X, y, cv=5):
    scores = cross_val_score(model, X, y, cv=cv, scoring="neg_mean_squared_error")
    return np.sqrt(-scores).mean()
```

---

## 📈 Visualizations

- **CV RMSE vs α** — to find optimal regularization strength  
- **Coefficient paths** — to visualize shrinkage  
- **Residual plots** — to check model fit  

Example Ridge coefficient path:

$$ \text{Plot: } \alpha \text{ (log scale)} \rightarrow \beta_j $$

---

## 🧠 Bias–Variance Tradeoff

As $\alpha$ increases:

- Variance ↓  
- Bias ↑  

Select $\alpha$ by cross-validation for best generalization.

---

## 🧩 Bayesian MAP Interpretation

From Bayes' theorem:

$$ P(\boldsymbol{\beta} \mid \mathbf{y}, X) \propto P(\mathbf{y} \mid X, \boldsymbol{\beta}) P(\boldsymbol{\beta}) $$

- Gaussian prior → Ridge  
- Laplace prior → Lasso  

---

## 🚀 Running the Script

Run all experiments and plots from the terminal:

```bash
python main.py
```

This executes:
- OLS baseline  
- Ridge, Lasso, and ElasticNet  
- CV performance  
- Coefficient and residual visualizations

---

## 📦 Dependencies

- Python ≥ 3.8  
- scikit-learn ≥ 0.21  
- numpy, pandas, matplotlib  

Install with:

```bash
pip install numpy pandas matplotlib scikit-learn
```

---

## 🧾 License

MIT License © 2025 Seyed Mohammadreza Mousavi

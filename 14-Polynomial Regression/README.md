# Polynomial Regression — Theory, Implementation & Visualization

This project demonstrates **Polynomial Regression** from scratch using NumPy and Matplotlib, including:

- Mathematical derivation of the model  
- Normal Equation and Gradient Descent solvers  
- Ridge (L2) regularization  
- K-Fold Cross Validation for model selection  
- Visualization of underfitting vs overfitting behavior  

---

## 🧮 Theoretical Background

Polynomial regression models a non-linear relationship by expanding the input variable into **polynomial features**.

Given training data $\{(x_i, y_i)\}_{i=1}^n$, we define a feature mapping:

$$ \phi(x) = [\,1,\, x,\, x^2,\, \ldots,\, x^d\,]^T $$

Stacking all samples, we get the **design matrix** $X \in \mathbb{R}^{n \times (d+1)}$ and the target vector $y \in \mathbb{R}^n$.  
The model is:

$$ \hat{y} = X\theta $$

where $\theta \in \mathbb{R}^{d+1}$ are the model coefficients.

---

### 🎯 Objective Function

We minimize the **Mean Squared Error (MSE)**:

$$ \mathcal{L}(\theta) = \frac{1}{n} \| X\theta - y \|_2^2 $$

The analytical (Normal Equation) solution is:

$$ \theta^\* = (X^T X)^{-1} X^T y $$

---

### 🛡️ Ridge Regularization

To prevent overfitting, we add an $L_2$ penalty term:

$$ \mathcal{L}_\lambda(\theta) = \frac{1}{n} \| X\theta - y \|_2^2 + \lambda \| D\theta \|_2^2 $$

where $D = \text{diag}(0, 1, 1, \ldots, 1)$ ensures the bias term is not penalized.  
The regularized solution becomes:

$$ \theta_\lambda^\* = (X^T X + \lambda D)^{-1} X^T y $$

---

### 🔁 Gradient Descent

We can also find $\theta$ iteratively via Gradient Descent:

$$ \nabla_\theta \mathcal{L}_\lambda(\theta) = \frac{2}{n} X^T(X\theta - y) + 2\lambda D\theta $$

and update:

$$ \theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}_\lambda(\theta) $$

where $\eta$ is the learning rate.

---

## ⚙️ Implementation Details

The script provides two classes:

- `PolyRegNormalEq`: Closed-form solution using the normal equation  
- `PolyRegGD`: Iterative optimization using gradient descent  

Also included:

- `make_poly_features`: Builds the polynomial design matrix  
- `kfold_cv_poly`: Performs simple K-Fold cross-validation  
- `train_val_split`: Splits data into train and validation sets  

---

## 📊 Visualization

The script visualizes:

- Fits of different polynomial degrees (e.g., 1, 3, 5, 9, 15)
- Cross-validation MSE vs. model degree  
- Ridge-regularized and Gradient Descent fits  

Example plot:  
True function $y = \sin(2\pi x) + 0.3x^3 + \epsilon$  
Demonstrates bias–variance tradeoff.

---

## 🧠 Bias–Variance Insight

- **Low degree ($d \approx 1$)** → underfitting (high bias)  
- **High degree ($d \gg 10$)** → overfitting (high variance)  
- **Moderate degree + regularization** → good generalization

---

## 🧰 How to Run

1. Clone or download the repo.
2. Install dependencies:
   ```bash
   pip install numpy matplotlib
   ```
3. Run:
   ```bash
   python main.py
   ```

---

## 📘 File Overview

| File | Description |
|------|--------------|
| `main.py` | Full Python implementation with visualization |
| `Polynomial_Regression.ipynb` | Annotated notebook with math, code, and plots |
| `README.md` | This documentation file |

---

## 🧩 Example Output

- Polynomial fits for multiple degrees  
- Cross-validation curve for degree selection  
- Ridge-regularized and GD comparison  

---

## 🧾 License

This project is provided for educational and research purposes.  
Feel free to reuse and adapt with proper attribution.

---

**Author:** Reza / MECHATEK  
**Date:** 2025-10-12

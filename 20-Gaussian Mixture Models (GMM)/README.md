
# Gaussian Mixture Models (GMM) — From Scratch

A complete implementation of **Gaussian Mixture Models (GMM)** using the **Expectation-Maximization (EM)** algorithm in Python (NumPy), including derivations, visualizations, and comparison with `scikit-learn`.

---

## 📘 Overview

This repository contains:

- A Jupyter Notebook (`GMM_Gaussian_Mixture_Models.ipynb`) with detailed math explanations.
- A standalone Python script (`main.py`) implementing GMM from scratch.
- Visualization utilities for 1D and 2D data.
- Comparison with `sklearn.mixture.GaussianMixture`.
- Model selection via **AIC** and **BIC**.

---

## 🧠 Mathematical Background

### 1. Mixture Model Definition

A Gaussian Mixture Model assumes that each data point $\\mathbf{x}_n \\in \\mathbb{R}^D$ is generated from one of $K$ Gaussian components:

$$
p(\\mathbf{x}_n) = \\sum_{k=1}^{K} \\pi_k \\; \\mathcal{N}(\\mathbf{x}_n \\mid \\boldsymbol{\\mu}_k, \\boldsymbol{\\Sigma}_k)
$$

where $\\pi_k$ are the mixing coefficients satisfying $\\sum_k \\pi_k = 1$ and $\\pi_k \\ge 0$.  
Each Gaussian is defined as:

$$
\\mathcal{N}(\\mathbf{x} \\mid \\boldsymbol{\\mu}, \\boldsymbol{\\Sigma}) = \\frac{1}{(2\\pi)^{D/2} \\; |\\boldsymbol{\\Sigma}|^{1/2}} \\exp\\Big(-\\tfrac{1}{2}(\\mathbf{x}-\\boldsymbol{\\mu})^{\\top}\\boldsymbol{\\Sigma}^{-1}(\\mathbf{x}-\\boldsymbol{\\mu})\\Big)
$$

---

### 2. Responsibilities (E-Step)

The **responsibility** $\\gamma_{nk}$ represents the posterior probability that component $k$ generated point $n$:

$$
\\gamma_{nk} = p(z_n=k \\mid \\mathbf{x}_n) = \\frac{\\pi_k\\,\\mathcal{N}(\\mathbf{x}_n \\mid \\boldsymbol{\\mu}_k,\\boldsymbol{\\Sigma}_k)}{\\sum_{j=1}^{K} \\pi_j\\,\\mathcal{N}(\\mathbf{x}_n \\mid \\boldsymbol{\\mu}_j,\\boldsymbol{\\Sigma}_j)}
$$

---

### 3. EM Algorithm

We maximize the **log-likelihood**:

$$
\\mathcal{L}(\\Theta) = \\sum_{n=1}^{N} \\log \\Big( \\sum_{k=1}^{K} \\pi_k\\,\\mathcal{N}(\\mathbf{x}_n \\mid \\boldsymbol{\\mu}_k, \\boldsymbol{\\Sigma}_k) \\Big)
$$

The EM steps alternate as follows:

$$
\\begin{aligned}
\\text{E-step:} \\quad & \\gamma_{nk} \\leftarrow \\frac{\\pi_k\\,\\mathcal{N}(\\mathbf{x}_n \\mid \\boldsymbol{\\mu}_k,\\boldsymbol{\\Sigma}_k)}{\\sum_{j=1}^{K} \\pi_j\\,\\mathcal{N}(\\mathbf{x}_n \\mid \\boldsymbol{\\mu}_j,\\boldsymbol{\\Sigma}_j)} \\\\
N_k &= \\sum_{n=1}^{N} \\gamma_{nk} \\\\
\\text{M-step:} \\quad & \\pi_k \\leftarrow \\frac{N_k}{N}, \\qquad \\boldsymbol{\\mu}_k \\leftarrow \\frac{1}{N_k}\\sum_{n=1}^{N} \\gamma_{nk}\\mathbf{x}_n, \\\\
& \\boldsymbol{\\Sigma}_k \\leftarrow \\frac{1}{N_k}\\sum_{n=1}^{N} \\gamma_{nk}(\\mathbf{x}_n - \\boldsymbol{\\mu}_k)(\\mathbf{x}_n - \\boldsymbol{\\mu}_k)^{\\top}
\\end{aligned}
$$

The algorithm iterates until the change in log-likelihood falls below a small threshold.

---

## ⚙️ Implementation Highlights

- **Language:** Python 3  
- **Libraries:** NumPy, Matplotlib, scikit-learn  
- **Initialization:** KMeans or random  
- **Covariance Type:** Full (positive-definite matrices)  
- **Regularization:** Adds small diagonal term $\\epsilon I$ for stability

---

## 📊 Visualizations

### 2D Example
- EM fitting on synthetic blobs
- Mixture-density contour visualization
- Log-likelihood progression plot

### 1D Example
- Fit a mixture of two Gaussians
- Histogram + fitted PDF curve

---

## 📈 Model Selection

The **AIC** and **BIC** criteria are computed as:

$$
\\text{AIC} = 2p - 2\\mathcal{L}_{\\max}, \\qquad \\text{BIC} = p\\,\\log(N) - 2\\mathcal{L}_{\\max}
$$

where $p$ is the number of model parameters, and $\\mathcal{L}_{\\max}$ is the log-likelihood at convergence.

---

## 🚀 Usage

```bash
# Clone repository
git clone https://github.com/yourusername/gmm-from-scratch.git
cd gmm-from-scratch

# Run the script
python main.py
```

The script will:
1. Fit and visualize a 2D GMM.
2. Compare with scikit-learn’s `GaussianMixture`.
3. Fit a 1D mixture and plot the estimated density.

---

## 🧩 File Structure

```
GMM_Gaussian_Mixture_Models.ipynb   # Full tutorial + math
main.py                             # Executable Python script
README.md                           # Documentation
```

---

## 📚 References

- Bishop, *Pattern Recognition and Machine Learning* (2006), Ch. 9  
- Murphy, *Machine Learning: A Probabilistic Perspective* (2012), Ch. 11  
- scikit-learn Documentation: [GaussianMixture](https://scikit-learn.org/stable/modules/mixture.html)

---

**Author:** Reza Mousavi  
**License:** MIT

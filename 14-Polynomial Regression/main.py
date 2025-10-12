"""
Polynomial Regression - Full Implementation (main.py)
Created on 2025-10-12

Includes:
- Mathematical explanation comments
- Polynomial feature expansion
- Normal Equation & Gradient Descent
- Ridge regularization
- K-Fold CV
- Visualization (Matplotlib)
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, Optional, List


def train_val_split(X: np.ndarray, y: np.ndarray, val_ratio: float = 0.2, shuffle: bool = True):
    n = X.shape[0]
    idx = np.arange(n)
    if shuffle:
        np.random.shuffle(idx)
    n_val = int(np.floor(val_ratio * n))
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]
    return X[tr_idx], X[val_idx], y[tr_idx], y[val_idx]


def make_poly_features(x: np.ndarray, degree: int) -> np.ndarray:
    x = np.asarray(x).reshape(-1, 1)
    powers = [np.ones_like(x)]
    for d in range(1, degree + 1):
        powers.append(powers[-1] * x)
    return np.hstack(powers)


def mse(y_true, y_pred):
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    return float(np.mean((y_true - y_pred) ** 2))


@dataclass
class PolyRegNormalEq:
    degree: int
    lam: float = 0.0
    penalize_bias: bool = False
    coef_: Optional[np.ndarray] = None

    def fit(self, x, y):
        X = make_poly_features(x, self.degree)
        y = y.reshape(-1, 1)
        D = np.eye(X.shape[1])
        if not self.penalize_bias:
            D[0, 0] = 0.0
        A = X.T @ X + self.lam * D
        b = X.T @ y
        self.coef_ = np.linalg.solve(A, b).reshape(-1)
        return self

    def predict(self, x):
        if self.coef_ is None:
            raise RuntimeError("Model not fitted yet.")
        X = make_poly_features(x, self.degree)
        return X @ self.coef_


@dataclass
class PolyRegGD:
    degree: int
    lam: float = 0.0
    penalize_bias: bool = False
    lr: float = 1e-2
    epochs: int = 5000
    coef_: Optional[np.ndarray] = None

    def fit(self, x, y):
        X = make_poly_features(x, self.degree)
        y = y.reshape(-1, 1)
        n, p = X.shape
        w = np.zeros((p, 1))
        D = np.eye(p)
        if not self.penalize_bias:
            D[0, 0] = 0.0
        for _ in range(self.epochs):
            yhat = X @ w
            grad = (2.0 / n) * (X.T @ (yhat - y)) + 2.0 * self.lam * (D @ w)
            w -= self.lr * grad
        self.coef_ = w.reshape(-1)
        return self

    def predict(self, x):
        if self.coef_ is None:
            raise RuntimeError("Model not fitted yet.")
        X = make_poly_features(x, self.degree)
        return X @ self.coef_


def kfold_indices(n: int, k: int, shuffle: bool = True):
    idx = np.arange(n)
    if shuffle:
        np.random.shuffle(idx)
    return np.array_split(idx, k)


def kfold_cv_poly(x, y, degrees, lambdas, k=5):
    x = x.reshape(-1)
    y = y.reshape(-1)
    folds = kfold_indices(len(x), k, shuffle=True)
    results = {}
    best = (None, None, float('inf'))
    for d in degrees:
        for lam in lambdas:
            val_losses = []
            for i in range(k):
                val_idx = folds[i]
                tr_idx = np.hstack([folds[j] for j in range(k) if j != i])
                x_tr, x_val = x[tr_idx], x[val_idx]
                y_tr, y_val = y[tr_idx], y[val_idx]
                model = PolyRegNormalEq(degree=d, lam=lam).fit(x_tr, y_tr)
                pred_val = model.predict(x_val)
                val_losses.append(mse(y_val, pred_val))
            mean_val = float(np.mean(val_losses))
            results[(d, lam)] = mean_val
            if mean_val < best[2]:
                best = (d, lam, mean_val)
    return best[0], best[1], results


def main():
    np.random.seed(42)
    n = 250
    x = np.random.uniform(-1.0, 1.0, size=n)
    def f_true(x): return np.sin(2 * np.pi * x) + 0.3 * (x ** 3)
    noise = 0.15 * np.random.randn(n)
    y = f_true(x) + noise

    X_tr, X_val, y_tr, y_val = train_val_split(x.reshape(-1,1), y, val_ratio=0.25)
    print(f"Train size: {len(X_tr)}, Val size: {len(X_val)}")

    degrees = [1, 3, 5, 9, 15]
    lam = 0.0
    grid = np.linspace(-1.2, 1.2, 400)

    plt.figure()
    plt.scatter(x, y, s=12, alpha=0.6, label="Data")
    plt.plot(grid, f_true(grid), linewidth=2, label="True function")
    for d in degrees:
        model = PolyRegNormalEq(degree=d, lam=lam).fit(X_tr.reshape(-1), y_tr)
        plt.plot(grid, model.predict(grid), label=f"deg={d}")
    plt.legend()
    plt.title("Polynomial Regression fits (Normal Equation)")
    plt.show()

    degrees_grid = list(range(0, 16))
    lambdas_grid = [0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    best_d, best_lam, cv_results = kfold_cv_poly(x, y, degrees_grid, lambdas_grid, k=5)
    print(f"Best degree: {best_d}, Best lambda: {best_lam}, CV-MSE: {cv_results[(best_d, best_lam)]:.5f}")

    mses_vs_degree = [cv_results[(d, best_lam)] for d in degrees_grid]
    plt.figure()
    plt.plot(degrees_grid, mses_vs_degree, marker='o')
    plt.xlabel("Degree d")
    plt.ylabel("CV MSE")
    plt.title(f"Validation MSE vs Degree (lambda={best_lam})")
    plt.show()

    final_model = PolyRegNormalEq(degree=best_d, lam=best_lam).fit(X_tr.reshape(-1), y_tr)
    plt.figure()
    plt.scatter(X_tr.reshape(-1), y_tr, s=12, alpha=0.6, label="Train")
    plt.scatter(X_val.reshape(-1), y_val, s=12, alpha=0.6, label="Val")
    plt.plot(grid, f_true(grid), linewidth=2, label="True function")
    plt.plot(grid, final_model.predict(grid), linewidth=2, label=f"Final (d={best_d}, λ={best_lam})")
    plt.legend()
    plt.title("Final Polynomial Regression Fit")
    plt.show()

    print("Train MSE:", mse(y_tr, final_model.predict(X_tr.reshape(-1))))
    print("Val   MSE:", mse(y_val, final_model.predict(X_val.reshape(-1))))

    gd_model = PolyRegGD(degree=min(12, best_d), lam=best_lam, lr=5e-3, epochs=4000)
    gd_model.fit(X_tr.reshape(-1), y_tr)
    plt.figure()
    plt.scatter(X_tr.reshape(-1), y_tr, s=12, alpha=0.6, label="Train")
    plt.plot(grid, f_true(grid), linewidth=2, label="True function")
    plt.plot(grid, gd_model.predict(grid), linewidth=2, label=f"GD fit (deg={gd_model.degree})")
    plt.legend()
    plt.title("Gradient Descent Fit")
    plt.show()


if __name__ == "__main__":
    main()

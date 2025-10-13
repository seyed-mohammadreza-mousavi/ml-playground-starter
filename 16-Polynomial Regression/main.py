"""
Polynomial Regression — main.py
---------------------------------
Implements polynomial regression end-to-end:
  * Generate synthetic nonlinear data
  * Build polynomial design matrix
  * Solve least-squares and ridge regression in closed form
  * Evaluate degree vs MSE (validation curve)
  * Visualize fits and residuals
  * Compare with scikit-learn implementation
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline


# ===============================================================
# Helper functions
# ===============================================================
def make_synthetic_data(n=100, noise_std=0.2):
    """Create 1D data from a smooth nonlinear function with Gaussian noise."""
    X = np.sort(np.random.uniform(-3, 3, size=n))
    y_true = np.sin(X) + 0.5 * np.cos(2*X)
    y = y_true + np.random.normal(scale=noise_std, size=n)
    return X.reshape(-1, 1), y, y_true


def polynomial_design_matrix(x, degree):
    """Generate polynomial design matrix [1, x, x^2, ..., x^degree]."""
    X = np.hstack([x**j for j in range(degree + 1)])
    return X


def fit_closed_form(X, y, lam=0.0):
    """Closed-form (ridge) regression solution."""
    n, d = X.shape
    A = X.T @ X + n * lam * np.eye(d)
    b = X.T @ y
    w = np.linalg.solve(A, b)
    return w


def predict_with_weights(X, w):
    return X @ w


def cv_score_for_degree(degree, X, y, K=5):
    """Compute K-fold CV MSE for given polynomial degree."""
    kf = KFold(n_splits=K, shuffle=True, random_state=0)
    scores = []
    for tr_idx, va_idx in kf.split(X):
        Xtr, Xva = X[tr_idx], X[va_idx]
        ytr, yva = y[tr_idx], y[va_idx]
        model = make_pipeline(PolynomialFeatures(degree=degree, include_bias=True),
                              LinearRegression())
        model.fit(Xtr, ytr)
        pred = model.predict(Xva)
        scores.append(mean_squared_error(yva, pred))
    return np.mean(scores)


# ===============================================================
# Main workflow
# ===============================================================
def main():
    np.random.seed(42)

    # 1. Generate data
    X, y, y_true = make_synthetic_data(n=120, noise_std=0.25)

    plt.figure()
    plt.scatter(X, y, s=18, alpha=0.7, label="data")
    plt.plot(X, y_true, linewidth=2, label="ground truth")
    plt.title("Synthetic Data")
    plt.legend()
    plt.show()

    # 2. Fit polynomial regression of various degrees
    degrees = [1, 3, 5, 9]
    x_plot = np.linspace(X.min(), X.max(), 400).reshape(-1, 1)

    for deg in degrees:
        Phi = polynomial_design_matrix(X, deg)
        Phi_plot = polynomial_design_matrix(x_plot, deg)
        w = fit_closed_form(Phi, y, lam=0.0)
        y_hat_plot = predict_with_weights(Phi_plot, w)

        plt.figure()
        plt.scatter(X, y, s=18, alpha=0.6, label="data")
        plt.plot(x_plot, y_hat_plot, linewidth=2, label=f"degree={deg}")
        plt.title(f"Polynomial Regression (degree={deg})")
        plt.xlabel("x"); plt.ylabel("y"); plt.legend()
        plt.show()

    # 3. Validation curve (train/val split)
    Xtr, Xval, ytr, yval = train_test_split(X, y, test_size=0.3, random_state=0)
    degrees_range = list(range(0, 16))
    mse_tr, mse_val = [], []

    for deg in degrees_range:
        Phi_tr = polynomial_design_matrix(Xtr, deg)
        Phi_val = polynomial_design_matrix(Xval, deg)
        w = fit_closed_form(Phi_tr, ytr, lam=0.0)
        mse_tr.append(mean_squared_error(ytr, predict_with_weights(Phi_tr, w)))
        mse_val.append(mean_squared_error(yval, predict_with_weights(Phi_val, w)))

    plt.figure()
    plt.plot(degrees_range, mse_tr, marker="o", label="train MSE")
    plt.plot(degrees_range, mse_val, marker="o", label="val MSE")
    plt.title("Validation Curve: Degree vs MSE")
    plt.xlabel("Degree"); plt.ylabel("MSE"); plt.legend()
    plt.show()

    best_deg = degrees_range[int(np.argmin(mse_val))]
    print(f"Best degree (validation): {best_deg}")

    # 4. Ridge regularization comparison
    deg = max(12, int(best_deg) + 3)
    Phi = polynomial_design_matrix(X, deg)
    x_plot = np.linspace(X.min(), X.max(), 400).reshape(-1, 1)
    Phi_plot = polynomial_design_matrix(x_plot, deg)

    lambdas = [0.0, 1e-4, 1e-2, 1e-1, 1.0]
    for lam in lambdas:
        w_ridge = fit_closed_form(Phi, y, lam=lam)
        y_plot = predict_with_weights(Phi_plot, w_ridge)

        plt.figure()
        plt.scatter(X, y, s=18, alpha=0.6, label="data")
        plt.plot(x_plot, y_plot, linewidth=2, label=f"lambda={lam}")
        plt.title(f"Ridge Regularization (deg={deg})")
        plt.xlabel("x"); plt.ylabel("y"); plt.legend()
        plt.show()

    # 5. Residual analysis
    deg = int(best_deg)
    Phi = polynomial_design_matrix(X, deg)
    w = fit_closed_form(Phi, y, lam=0.0)
    y_pred = predict_with_weights(Phi, w)
    residuals = y - y_pred

    plt.figure()
    plt.scatter(y_pred, residuals, s=18, alpha=0.7)
    plt.axhline(0.0, color='gray')
    plt.title(f"Residuals vs Predictions (degree={deg})")
    plt.xlabel("Predicted y"); plt.ylabel("Residual")
    plt.show()

    # 6. scikit-learn comparison
    lin_model = make_pipeline(PolynomialFeatures(degree=deg, include_bias=True),
                              LinearRegression())
    lin_model.fit(X, y)
    y_lin = lin_model.predict(x_plot)

    plt.figure()
    plt.scatter(X, y, s=18, alpha=0.6, label="data")
    plt.plot(x_plot, y_lin, linewidth=2, label=f"sklearn degree={deg}")
    plt.title("scikit-learn Polynomial Regression")
    plt.legend(); plt.show()

    # 7. Cross-validation degree selection
    cv_mse = [cv_score_for_degree(d, X, y, K=5) for d in degrees_range]
    plt.figure()
    plt.plot(degrees_range, cv_mse, marker="o")
    plt.title("K-Fold CV: Degree vs MSE")
    plt.xlabel("Degree"); plt.ylabel("CV MSE")
    plt.show()

    cv_best_deg = degrees_range[int(np.argmin(cv_mse))]
    print(f"Best degree (CV): {cv_best_deg}")


if __name__ == "__main__":
    main()

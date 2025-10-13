"""
main.py — Regularization Techniques (Ridge, Lasso, ElasticNet)
===============================================================

This script demonstrates regularization in linear models, including:
- Ridge Regression (L2)
- Lasso Regression (L1)
- ElasticNet (L1 + L2)

It covers:
1. Mathematical background
2. Implementation using scikit-learn
3. Cross-validation and model comparison
4. Visualization of coefficient paths and residuals
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import warnings
from sklearn.exceptions import ConvergenceWarning


# ------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------

def cv_rmse(model, X, y, cv=5):
    """Compute cross-validated RMSE."""
    scores = cross_val_score(model, X, y, cv=cv, scoring="neg_mean_squared_error")
    return np.sqrt(-scores).mean()


def plot_residuals(y_true, y_pred, title):
    """Simple residual plot."""
    res = y_true - y_pred
    plt.figure()
    plt.scatter(y_pred, res, alpha=0.6)
    plt.axhline(0.0, color='r', linestyle='--')
    plt.xlabel("Predictions")
    plt.ylabel("Residuals")
    plt.title(title)
    plt.show()


# ------------------------------------------------------------
# Main function
# ------------------------------------------------------------

def main():
    # Ignore harmless convergence warnings for near-Ridge ElasticNet
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    # --------------------------------------------------------
    # 1. Dataset
    # --------------------------------------------------------
    np.random.seed(42)
    X, y = make_regression(
        n_samples=600, n_features=8, n_informative=5, noise=25.0, random_state=42
    )

    # Polynomial features → correlation and complexity
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_poly, y, test_size=0.25, random_state=42
    )

    # --------------------------------------------------------
    # 2. OLS baseline
    # --------------------------------------------------------
    print("\n=== Ordinary Least Squares (OLS) ===")
    ols = Pipeline([
        ("scaler", StandardScaler()),
        ("reg", LinearRegression())
    ])
    ols.fit(X_train, y_train)
    y_pred_ols = ols.predict(X_test)

    rmse_ols = np.sqrt(mean_squared_error(y_test, y_pred_ols))
    r2_ols = r2_score(y_test, y_pred_ols)
    cv_ols = cv_rmse(ols, X_train, y_train)

    print(f"OLS Test RMSE: {rmse_ols:.4f}")
    print(f"OLS Test R^2 : {r2_ols:.4f}")
    print(f"OLS CV RMSE  : {cv_ols:.4f}")

    # --------------------------------------------------------
    # 3. Ridge Regression
    # --------------------------------------------------------
    print("\n=== Ridge Regression ===")
    alphas = np.logspace(-4, 2, 30)
    ridge_cv_rmse = []

    for a in alphas:
        ridge = Pipeline([
            ("scaler", StandardScaler()),
            ("reg", Ridge(alpha=a, random_state=42))
        ])
        ridge_cv_rmse.append(cv_rmse(ridge, X_train, y_train))

    best_alpha_ridge = alphas[int(np.argmin(ridge_cv_rmse))]

    plt.figure()
    plt.semilogx(alphas, ridge_cv_rmse, marker="o")
    plt.xlabel("alpha")
    plt.ylabel("CV RMSE")
    plt.title("Ridge: CV RMSE vs alpha")
    plt.show()

    ridge_best = Pipeline([
        ("scaler", StandardScaler()),
        ("reg", Ridge(alpha=best_alpha_ridge, random_state=42))
    ])
    ridge_best.fit(X_train, y_train)
    y_pred_ridge = ridge_best.predict(X_test)

    rmse_ridge = np.sqrt(mean_squared_error(y_test, y_pred_ridge))
    r2_ridge = r2_score(y_test, y_pred_ridge)

    print(f"Best alpha (Ridge): {best_alpha_ridge:.6f}")
    print(f"Ridge Test RMSE   : {rmse_ridge:.4f}")
    print(f"Ridge Test R^2    : {r2_ridge:.4f}")

    # --------------------------------------------------------
    # 4. Lasso Regression
    # --------------------------------------------------------
    print("\n=== Lasso Regression ===")
    lasso_cv_rmse = []
    for a in alphas:
        lasso = Pipeline([
            ("scaler", StandardScaler()),
            ("reg", Lasso(alpha=a, max_iter=20000, random_state=42))
        ])
        lasso_cv_rmse.append(cv_rmse(lasso, X_train, y_train))

    best_alpha_lasso = alphas[int(np.argmin(lasso_cv_rmse))]

    plt.figure()
    plt.semilogx(alphas, lasso_cv_rmse, marker="o")
    plt.xlabel("alpha")
    plt.ylabel("CV RMSE")
    plt.title("Lasso: CV RMSE vs alpha")
    plt.show()

    lasso_best = Pipeline([
        ("scaler", StandardScaler()),
        ("reg", Lasso(alpha=best_alpha_lasso, max_iter=20000, random_state=42))
    ])
    lasso_best.fit(X_train, y_train)
    y_pred_lasso = lasso_best.predict(X_test)

    rmse_lasso = np.sqrt(mean_squared_error(y_test, y_pred_lasso))
    r2_lasso = r2_score(y_test, y_pred_lasso)

    print(f"Best alpha (Lasso): {best_alpha_lasso:.6f}")
    print(f"Lasso Test RMSE   : {rmse_lasso:.4f}")
    print(f"Lasso Test R^2    : {r2_lasso:.4f}")

    # --------------------------------------------------------
    # 5. ElasticNet
    # --------------------------------------------------------
    print("\n=== ElasticNet ===")
    rhos = np.linspace(0.1, 1.0, 10)  # skip 0.0 to avoid warnings
    en_results = []

    for a in alphas:
        for r in rhos:
            en = Pipeline([
                ("scaler", StandardScaler()),
                ("reg", ElasticNet(alpha=a, l1_ratio=r, max_iter=30000, random_state=42))
            ])
            en_results.append((a, r, cv_rmse(en, X_train, y_train)))

    en_df = pd.DataFrame(en_results, columns=["alpha", "rho", "cv_rmse"])
    idx_best = en_df["cv_rmse"].idxmin()
    best_alpha_en = en_df.loc[idx_best, "alpha"]
    best_rho_en = en_df.loc[idx_best, "rho"]

    en_best = Pipeline([
        ("scaler", StandardScaler()),
        ("reg", ElasticNet(alpha=best_alpha_en, l1_ratio=best_rho_en, max_iter=30000, random_state=42))
    ])
    en_best.fit(X_train, y_train)
    y_pred_en = en_best.predict(X_test)

    rmse_en = np.sqrt(mean_squared_error(y_test, y_pred_en))
    r2_en = r2_score(y_test, y_pred_en)

    print(f"Best (alpha, rho): ({best_alpha_en:.6f}, {best_rho_en:.2f})")
    print(f"ElasticNet Test RMSE: {rmse_en:.4f}")
    print(f"ElasticNet Test R^2 : {r2_en:.4f}")

    # --------------------------------------------------------
    # 6. Coefficient Paths
    # --------------------------------------------------------
    print("\nPlotting coefficient paths...")

    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    coef_ridge, coef_lasso = [], []

    for a in alphas:
        rr = Ridge(alpha=a, random_state=42).fit(X_train_std, y_train)
        coef_ridge.append(rr.coef_)
        ll = Lasso(alpha=a, max_iter=20000, random_state=42).fit(X_train_std, y_train)
        coef_lasso.append(ll.coef_)

    coef_ridge = np.array(coef_ridge)
    coef_lasso = np.array(coef_lasso)

    plt.figure()
    for j in range(coef_ridge.shape[1]):
        plt.semilogx(alphas, coef_ridge[:, j])
    plt.title("Ridge Coefficient Paths")
    plt.xlabel("alpha")
    plt.ylabel("Coefficient")
    plt.show()

    plt.figure()
    for j in range(coef_lasso.shape[1]):
        plt.semilogx(alphas, coef_lasso[:, j])
    plt.title("Lasso Coefficient Paths")
    plt.xlabel("alpha")
    plt.ylabel("Coefficient")
    plt.show()

    # --------------------------------------------------------
    # 7. Model Comparison Summary
    # --------------------------------------------------------
    summary = pd.DataFrame([
        ("OLS", rmse_ols, r2_ols, cv_ols),
        ("Ridge", rmse_ridge, r2_ridge, min(ridge_cv_rmse)),
        ("Lasso", rmse_lasso, r2_lasso, min(lasso_cv_rmse)),
        ("ElasticNet", rmse_en, r2_en, en_df["cv_rmse"].min())
    ], columns=["Model", "Test RMSE", "Test R^2", "CV RMSE"])

    print("\n=== Model Comparison ===")
    print(summary.to_string(index=False))

    # --------------------------------------------------------
    # 8. Residual Diagnostics
    # --------------------------------------------------------
    print("\nPlotting residuals...")
    plot_residuals(y_test, y_pred_ols, "Residuals: OLS")
    plot_residuals(y_test, y_pred_ridge, "Residuals: Ridge")
    plot_residuals(y_test, y_pred_lasso, "Residuals: Lasso")
    plot_residuals(y_test, y_pred_en, "Residuals: ElasticNet")

    print("\n✅ All done! Regularization experiment complete.")


# ------------------------------------------------------------
# Script entry point
# ------------------------------------------------------------
if __name__ == "__main__":
    main()
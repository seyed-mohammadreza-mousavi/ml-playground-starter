import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, r2_score


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def fit_pcr(X_train_raw, y_train, n_components):
    """
    Fit a PCR model:
      1. Standardize X
      2. Perform PCA (n_components)
      3. Regress y on principal components

    Returns: (scaler, pca, linear_model)
    """
    scaler = StandardScaler(with_mean=True, with_std=True)
    pca = PCA(n_components=n_components, svd_solver='full', random_state=42)
    lr = LinearRegression()

    Xs = scaler.fit_transform(X_train_raw)
    Z = pca.fit_transform(Xs)
    lr.fit(Z, y_train)
    return scaler, pca, lr


def pcr_predict(scaler, pca, lr, X_raw):
    """Predict using the trained PCR model."""
    Xs = scaler.transform(X_raw)
    Z = pca.transform(Xs)
    return lr.predict(Z)


def recover_original_coefficients(scaler, pca, lr):
    """
    Recover coefficients in the original feature space.

    For standardized data Xs = (X_raw - mu)/sigma:
      Z = Xs V  (where sklearn PCA.components_ = V^T)
      y = Z * gamma + b

    So in raw space:
      beta_raw = (1/sigma) * V_k * gamma
      intercept = b - mu^T * beta_raw
    """
    Vt = pca.components_
    V = Vt.T
    gamma = lr.coef_
    sigma = scaler.scale_
    mu = scaler.mean_

    beta_pcr = V @ gamma
    beta_raw = beta_pcr / sigma
    beta0 = lr.intercept_ - mu @ beta_raw
    return beta0, beta_raw


# ---------------------------------------------------------------------------
# Main Function
# ---------------------------------------------------------------------------

def main():
    # -----------------------------------------------------------------------
    # Step 1: Generate synthetic data with multicollinearity
    # -----------------------------------------------------------------------
    print("Generating synthetic data...")
    n_samples = 600
    n_features = 20
    effective_rank = 5

    X_raw, y, coef_true = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=8,
        noise=12.0,
        coef=True,
        effective_rank=effective_rank,
        random_state=42
    )

    rng = np.random.default_rng(42)
    X_raw = np.hstack([
        X_raw,
        X_raw[:, [0]] + 0.01 * rng.normal(size=(n_samples, 1)),
        X_raw[:, [1]] + 0.01 * rng.normal(size=(n_samples, 1))
    ])

    feature_names = [f"x{j+1}" for j in range(X_raw.shape[1])]

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.25, random_state=42
    )

    # -----------------------------------------------------------------------
    # Step 2: Cross-validation to choose number of components
    # -----------------------------------------------------------------------
    print("Performing cross-validation to select number of components...")
    p = X_train_raw.shape[1]
    ks = np.arange(1, min(p, 40) + 1)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    cv_mse = []
    for k in ks:
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=k, svd_solver='full', random_state=42)),
            ("lr", LinearRegression())
        ])
        scores = cross_val_score(pipe, X_train_raw, y_train, scoring="neg_mean_squared_error", cv=cv)
        cv_mse.append(-scores.mean())

    best_k = ks[int(np.argmin(cv_mse))]
    print(f"Best number of components: k = {best_k}")

    # -----------------------------------------------------------------------
    # Step 3: Fit final PCR and compare with OLS/Ridge
    # -----------------------------------------------------------------------
    print("Fitting final PCR and comparing with OLS and Ridge...")
    scaler, pca, lr = fit_pcr(X_train_raw, y_train, n_components=int(best_k))

    ols = Pipeline([("scaler", StandardScaler()), ("lr", LinearRegression())])
    ridge = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=5.0))])

    ols.fit(X_train_raw, y_train)
    ridge.fit(X_train_raw, y_train)

    y_pred_pcr = pcr_predict(scaler, pca, lr, X_test_raw)
    y_pred_ols = ols.predict(X_test_raw)
    y_pred_ridge = ridge.predict(X_test_raw)

    mse_pcr = mean_squared_error(y_test, y_pred_pcr)
    mse_ols = mean_squared_error(y_test, y_pred_ols)
    mse_ridge = mean_squared_error(y_test, y_pred_ridge)

    r2_pcr = r2_score(y_test, y_pred_pcr)
    r2_ols = r2_score(y_test, y_pred_ols)
    r2_ridge = r2_score(y_test, y_pred_ridge)

    results = pd.DataFrame({
        "Model": ["PCR", "OLS (std)", "Ridge (alpha=5)"],
        "Test MSE": [mse_pcr, mse_ols, mse_ridge],
        "Test R^2": [r2_pcr, r2_ols, r2_ridge]
    })
    print(results, "\\n")

    # -----------------------------------------------------------------------
    # Step 4: Visualization
    # -----------------------------------------------------------------------
    print("Generating plots...")

    # Explained variance ratio
    pca_full = PCA(n_components=min(X_train_raw.shape), svd_solver='full', random_state=42)
    Xs_full = StandardScaler().fit_transform(X_train_raw)
    pca_full.fit(Xs_full)
    evr = pca_full.explained_variance_ratio_

    plt.figure()
    plt.plot(np.arange(1, len(evr)+1), evr, marker="o")
    plt.xlabel("Component index")
    plt.ylabel("Explained variance ratio")
    plt.title("Explained Variance Ratio per Component")
    plt.tight_layout()
    plt.show()

    # CV MSE curve
    plt.figure()
    plt.plot(ks, cv_mse, marker="o")
    plt.axvline(best_k, linestyle="--", color="r")
    plt.xlabel("Number of components k")
    plt.ylabel("CV MSE")
    plt.title("PCR Model Selection via Cross-Validation")
    plt.tight_layout()
    plt.show()

    # Predicted vs True
    plt.figure()
    plt.scatter(y_test, y_pred_pcr, s=15, alpha=0.7)
    plt.xlabel("True y")
    plt.ylabel("Predicted y (PCR)")
    plt.title("PCR: Predicted vs True (Test Set)")
    plt.tight_layout()
    plt.show()

    # Residuals
    residuals = y_test - y_pred_pcr
    plt.figure()
    plt.scatter(y_pred_pcr, residuals, s=15, alpha=0.7)
    plt.axhline(0.0, linestyle="--", color="r")
    plt.xlabel("Predicted y (PCR)")
    plt.ylabel("Residuals")
    plt.title("PCR: Residuals vs Predicted")
    plt.tight_layout()
    plt.show()

    # -----------------------------------------------------------------------
    # Step 5: Recover coefficients in raw space
    # -----------------------------------------------------------------------
    beta0_raw, beta_raw = recover_original_coefficients(scaler, pca, lr)
    coef_df = pd.DataFrame({
        "feature": feature_names,
        "beta_raw": beta_raw
    }).sort_values("beta_raw", key=lambda s: np.abs(s), ascending=False).reset_index(drop=True)
    print("Top 10 coefficients (raw feature space):")
    print(coef_df.head(10))

    # Sanity check: predictions via recovered coefficients
    y_pred_check = beta0_raw + X_test_raw @ beta_raw
    diff = np.max(np.abs(y_pred_check - y_pred_pcr))
    print(f"Sanity check (max diff between methods): {diff:.6f}")


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()

"""
Support Vector Regression (SVR) — From Theory to Practice
---------------------------------------------------------

This script demonstrates Support Vector Regression (SVR) end-to-end:
1. Mathematical foundation (briefly summarized)
2. Data generation and preprocessing
3. Model training (RBF kernel SVR)
4. Evaluation metrics (MAE, RMSE, R²)
5. Visualization of fit, epsilon-tube, support vectors
6. Kernel comparison
7. Hyperparameter heatmap (C vs epsilon)
8. Support vectors vs epsilon

Mathematical Background
-----------------------
SVR minimizes:
    (1/2)||w||² + C * Σ(ξ_i + ξ_i*)
subject to:
    y_i - wᵀφ(x_i) - b ≤ ε + ξ_i
    wᵀφ(x_i) + b - y_i ≤ ε + ξ_i*
    ξ_i, ξ_i* ≥ 0

Dual form (using kernels):
maximize over α_i, α_i*:
    -1/2 ΣΣ (α_i - α_i*)(α_j - α_j*)K(x_i, x_j)
    - ε Σ (α_i + α_i*) + Σ y_i(α_i - α_i*)
subject to:
    Σ (α_i - α_i*) = 0
    0 ≤ α_i, α_i* ≤ C

Prediction:
    f(x) = Σ (α_i - α_i*) K(x_i, x) + b
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import sklearn
import sys


# ----------------------------
# Helper: Backward-compatible RMSE
# ----------------------------
def safe_rmse(y_true, y_pred):
    try:
        return mean_squared_error(y_true, y_pred, squared=False)
    except TypeError:
        return np.sqrt(mean_squared_error(y_true, y_pred))


# ----------------------------
# Evaluation Function
# ----------------------------
def evaluate(model, X_tr, y_tr, X_te, y_te, label="SVR"):
    y_tr = np.ravel(y_tr)
    y_te = np.ravel(y_te)
    yhat_tr = model.predict(X_tr)
    yhat_te = model.predict(X_te)
    metrics = {
        "Model": label,
        "MAE_train": mean_absolute_error(y_tr, yhat_tr),
        "RMSE_train": safe_rmse(y_tr, yhat_tr),
        "R2_train": r2_score(y_tr, yhat_tr),
        "MAE_test": mean_absolute_error(y_te, yhat_te),
        "RMSE_test": safe_rmse(y_te, yhat_te),
        "R2_test": r2_score(y_te, yhat_te),
    }
    return metrics, yhat_tr, yhat_te


# ----------------------------
# Main Execution
# ----------------------------
def main():
    print("\n🚀 Support Vector Regression (SVR) Tutorial — Running...\n")

    # 1️⃣ Generate synthetic nonlinear regression dataset
    np.random.seed(42)
    n = 250
    X = np.linspace(-3, 3, n).reshape(-1, 1)
    y_true = np.sinc(X).ravel()  # sin(pi x)/(pi x)
    noise = 0.1 * np.random.randn(n)
    y = y_true + noise

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    # 2️⃣ Hyperparameter tuning with RBF kernel
    param_grid = {
        "C": [0.5, 1.0, 10.0, 100.0],
        "epsilon": [0.01, 0.05, 0.1, 0.2],
        "gamma": ["scale", 0.1, 0.5, 1.0]
    }

    print("\n🔍 Running GridSearchCV for RBF SVR...")
    svr = SVR(kernel="rbf")
    gcv = GridSearchCV(svr, param_grid, cv=5, scoring="neg_mean_squared_error", n_jobs=-1)
    gcv.fit(X_train, y_train)
    best_svr = gcv.best_estimator_
    print("✅ Best Parameters:", gcv.best_params_)

    # 3️⃣ Evaluate tuned model
    metrics_rbf, yhat_tr, yhat_te = evaluate(best_svr, X_train, y_train, X_test, y_test, label="SVR-RBF (tuned)")
    print("\n📊 Performance:\n", pd.DataFrame([metrics_rbf]))

    # 4️⃣ Visualize predictions, epsilon-tube, support vectors
    Xd = np.linspace(X.min(), X.max(), 1000).reshape(-1, 1)
    yhat_d = best_svr.predict(Xd)
    eps = best_svr.epsilon

    plt.figure(figsize=(8, 5))
    plt.scatter(X_train, y_train, s=18, alpha=0.7, label="Train")
    plt.scatter(X_test, y_test, s=18, alpha=0.7, label="Test")
    plt.plot(Xd, yhat_d, linewidth=2, label="SVR fit")
    plt.plot(Xd, yhat_d + eps, "--", label="+epsilon")
    plt.plot(Xd, yhat_d - eps, "--", label="-epsilon")
    sv = best_svr.support_
    plt.scatter(X_train[sv], y_train[sv], s=60, facecolors="none", edgecolors="k", label="Support vectors")
    plt.title("SVR (RBF) fit with epsilon-tube and support vectors")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()

    # 5️⃣ Kernel Comparison
    print("\n🌀 Comparing Kernels...")
    kernels = [
        ("linear", SVR(kernel="linear", C=10.0, epsilon=0.05)),
        ("poly (deg=3)", SVR(kernel="poly", degree=3, C=10.0, epsilon=0.05, gamma="scale")),
        ("rbf", SVR(kernel="rbf", C=10.0, epsilon=0.05, gamma="scale")),
    ]

    rows = []
    plt.figure(figsize=(8, 5))
    plt.scatter(X_train, y_train, s=16, alpha=0.5, label="Train")
    plt.scatter(X_test, y_test, s=16, alpha=0.5, label="Test")

    Xd = np.linspace(X.min(), X.max(), 500).reshape(-1, 1)
    for name, mdl in kernels:
        mdl.fit(X_train, y_train)
        metrics, _, _ = evaluate(mdl, X_train, y_train, X_test, y_test, label=name)
        rows.append(metrics)
        yhat_d = mdl.predict(Xd)
        plt.plot(Xd, yhat_d, linewidth=2, label=name)

    plt.title("Kernel Comparison: Linear vs Poly vs RBF")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()
    print(pd.DataFrame(rows).sort_values("RMSE_test"))

    # 6️⃣ Hyperparameter grid (C, epsilon)
    print("\n📈 Building heatmap of RMSE across (C, epsilon)...")

    def safe_rmse_local(y_true, y_pred):
        try:
            return mean_squared_error(y_true, y_pred, squared=False)
        except TypeError:
            return np.sqrt(mean_squared_error(y_true, y_pred))

    Cs = np.logspace(-1, 2, 6)
    epsilons = np.array([0.01, 0.03, 0.05, 0.1, 0.2, 0.3])
    rmse = np.zeros((len(Cs), len(epsilons)))

    for i, C in enumerate(Cs):
        for j, eps in enumerate(epsilons):
            mdl = SVR(kernel="rbf", C=C, epsilon=eps, gamma="scale")
            mdl.fit(X_train, y_train)
            yhat = mdl.predict(X_test)
            rmse[i, j] = safe_rmse_local(y_test, yhat)

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(rmse, origin="lower", aspect="auto")
    ax.set_xticks(range(len(epsilons)))
    ax.set_xticklabels([f"{e:.2f}" for e in epsilons])
    ax.set_yticks(range(len(Cs)))
    ax.set_yticklabels([f"{c:.2f}" for c in Cs])
    ax.set_xlabel(r"$\epsilon$")
    ax.set_ylabel("C")
    ax.set_title("Test RMSE across (C, epsilon) for RBF SVR")
    plt.colorbar(im, ax=ax, label="RMSE")
    plt.show()

    # 7️⃣ Support Vectors vs epsilon
    print("\n🔢 Counting support vectors vs epsilon...")
    eps_grid = [0.01, 0.03, 0.05, 0.1, 0.2, 0.3]
    rows = []

    for eps in eps_grid:
        mdl = SVR(kernel="rbf", C=10.0, epsilon=eps, gamma="scale")
        mdl.fit(X_train, y_train)
        rows.append({"epsilon": eps, "n_support_vectors": mdl.support_.size})

    df_sv = pd.DataFrame(rows)
    print(df_sv)

    plt.figure(figsize=(7, 4))
    plt.plot(df_sv["epsilon"], df_sv["n_support_vectors"], marker="o")
    plt.title("Number of Support Vectors vs epsilon")
    plt.xlabel("epsilon")
    plt.ylabel("# Support Vectors")
    plt.show()

    # 8️⃣ Versions for reproducibility
    print("\n📦 Environment:")
    print("Python:", sys.version)
    print("NumPy:", np.__version__)
    print("pandas:", pd.__version__)
    print("scikit-learn:", sklearn.__version__)

    print("\n✅ SVR pipeline completed successfully.")


# ----------------------------
# Entry Point
# ----------------------------
if __name__ == "__main__":
    main()
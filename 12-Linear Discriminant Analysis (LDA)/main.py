"""
Linear Discriminant Analysis (LDA): Theory, Implementation, and Visualization
------------------------------------------------------------------------------

This script contains:
- Mathematical background (in comments)
- A from-scratch implementation of Linear Discriminant Analysis
- Visualization and comparison with sklearn’s LDA
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional
from sklearn.datasets import load_iris, make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as SklearnLDA


# -------------------------------------------------------------------
# Mathematical Background (Quick Summary)
# -------------------------------------------------------------------
# Bayes’ theorem:
#   P(y | x) ∝ P(x | y) * P(y)
#
# LDA assumes:
#   x | y=c ~ N(μ_c, Σ)
#   => log P(y=c | x) = xᵀΣ⁻¹μ_c - ½μ_cᵀΣ⁻¹μ_c + log π_c
#   -> Linear decision boundaries.
#
# Fisher criterion:
#   Maximize J(W) = |WᵀS_BW| / |WᵀS_WW|
#   where:
#       S_W = within-class scatter
#       S_B = between-class scatter
#   Solve S_B w = λ S_W w
# -------------------------------------------------------------------


@dataclass
class LDAFromScratch:
    n_components: Optional[int] = None
    reg: float = 1e-6  # regularization term

    classes_: np.ndarray = None
    priors_: np.ndarray = None
    means_: np.ndarray = None
    Sigma_: np.ndarray = None
    W_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        C = len(self.classes_)
        N, d = X.shape

        # Compute class means and priors
        self.means_ = np.vstack([X[y_idx == c].mean(axis=0) for c in range(C)])
        counts = np.bincount(y_idx)
        self.priors_ = counts / counts.sum()

        # Compute within-class (S_W) and between-class (S_B) scatter matrices
        mu = X.mean(axis=0)
        S_W = np.zeros((d, d))
        S_B = np.zeros((d, d))
        for c in range(C):
            Xc = X[y_idx == c]
            centered = Xc - self.means_[c]
            S_W += centered.T @ centered
            mean_diff = (self.means_[c] - mu).reshape(-1, 1)
            S_B += counts[c] * (mean_diff @ mean_diff.T)

        # Shared covariance
        self.Sigma_ = S_W / (N - C)
        self.Sigma_ += self.reg * np.eye(d)  # regularization

        # Fisher projection
        m = self.n_components if self.n_components is not None else min(d, C - 1)
        from numpy.linalg import solve, eig, qr

        A = solve(self.Sigma_, S_B)
        eigvals, eigvecs = eig(A)
        order = np.argsort(-eigvals.real)
        W = eigvecs[:, order[:m]].real
        Q, _ = qr(W)
        self.W_ = Q[:, :m]
        return self

    def transform(self, X: np.ndarray):
        if self.W_ is None:
            raise ValueError("Model not fitted.")
        return np.asarray(X, dtype=float) @ self.W_

    def decision_function(self, X: np.ndarray):
        X = np.asarray(X, dtype=float)
        Sigma_inv = np.linalg.inv(self.Sigma_)
        C = len(self.classes_)
        scores = np.zeros((X.shape[0], C))
        for c in range(C):
            mu_c = self.means_[c]
            term1 = X @ (Sigma_inv @ mu_c)
            term2 = -0.5 * (mu_c.T @ Sigma_inv @ mu_c)
            term3 = np.log(self.priors_[c] + 1e-12)
            scores[:, c] = term1 + term2 + term3
        return scores

    def predict(self, X: np.ndarray):
        scores = self.decision_function(X)
        idx = scores.argmax(axis=1)
        return self.classes_[idx]


def plot_decision_regions_2d(model, X, y, title="Decision Regions", h=0.02):
    """Visualize 2D decision boundaries."""
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid).reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.25)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=25, edgecolor='k')
    plt.title(title)
    plt.show()


def main():
    np.random.seed(42)
    # ------------------------------------------------------------
    # Example 1: Iris dataset
    # ------------------------------------------------------------
    print("\\n=== Iris Dataset ===")
    iris = load_iris()
    X = StandardScaler().fit_transform(iris.data)
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    lda_fs = LDAFromScratch(n_components=2, reg=1e-4).fit(X_train, y_train)
    y_pred_fs = lda_fs.predict(X_test)

    print("From-scratch LDA Accuracy:", accuracy_score(y_test, y_pred_fs))
    print(classification_report(y_test, y_pred_fs))
    print("Confusion Matrix:\\n", confusion_matrix(y_test, y_pred_fs))

    # Plot Fisher projection
    Z_train = lda_fs.transform(X_train)
    plt.scatter(Z_train[:, 0], Z_train[:, 1], c=y_train, s=25, edgecolor='k')
    plt.title("Iris — Fisher LDA Projection")
    plt.xlabel("LD1"); plt.ylabel("LD2")
    plt.show()

    # Compare with sklearn
    sk_lda = SklearnLDA(n_components=2)
    sk_lda.fit(X_train, y_train)
    y_pred_sk = sk_lda.predict(X_test)
    print("sklearn LDA Accuracy:", accuracy_score(y_test, y_pred_sk))

    # ------------------------------------------------------------
    # Example 2: Toy 2D Blobs
    # ------------------------------------------------------------
    print("\\n=== 2D Blobs Example ===")
    Xb, yb = make_blobs(n_samples=600, centers=3, cluster_std=2.0, random_state=7)
    Xb = StandardScaler().fit_transform(Xb)
    Xb_tr, Xb_te, yb_tr, yb_te = train_test_split(Xb, yb, test_size=0.3, stratify=yb, random_state=7)

    lda_blobs = LDAFromScratch(n_components=2, reg=1e-4).fit(Xb_tr, yb_tr)
    yp = lda_blobs.predict(Xb_te)
    print("From-scratch LDA (blobs) accuracy:", accuracy_score(yb_te, yp))
    plot_decision_regions_2d(lda_blobs, Xb_tr, yb_tr, title="LDA Decision Regions (Train)")

    print("\\nScript completed successfully.")


if __name__ == "__main__":
    main()

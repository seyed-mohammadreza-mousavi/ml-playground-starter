"""
Linear AdaBoost Classifier — Theory, Implementation, and Visualization
=====================================================================

This script implements a full AdaBoost pipeline (binary classification)
with Decision Stumps as weak learners.

Mathematical background
-----------------------
AdaBoost combines many weak learners h_t(x) into a strong classifier:

    F_T(x) = Σ_{t=1}^T α_t h_t(x)

Final prediction:

    ŷ(x) = sign(F_T(x))

Exponential loss minimized:

    L(F) = Σ_i exp(-y_i F(x_i))

At each round t:

1. Weighted error:
       ε_t = Σ_i D_t(i) [y_i ≠ h_t(x_i)]
2. Learner weight:
       α_t = ½ ln((1 - ε_t) / ε_t)
3. Weight update:
       D_{t+1}(i) ∝ D_t(i) exp(-α_t y_i h_t(x_i))

-----------------------------------------------------------------------
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from sklearn.datasets import make_classification, make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier


# --------------------------------------------------------------------
# Weak learner: Decision Stump
# --------------------------------------------------------------------
@dataclass
class DecisionStump:
    feature: int = 0
    threshold: float = 0.0
    polarity: int = 1  # +1 or -1

    def predict(self, X):
        feature_values = X[:, self.feature]
        preds = np.ones(X.shape[0], dtype=int)
        if self.polarity == 1:
            preds[feature_values < self.threshold] = -1
        else:
            preds[feature_values >= self.threshold] = -1
        return preds


# --------------------------------------------------------------------
# AdaBoost implementation from scratch
# --------------------------------------------------------------------
class AdaBoostScratch:
    def __init__(self, n_estimators=50, learning_rate=1.0):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.alphas = []
        self.stumps = []

    def _best_stump(self, X, y, w):
        n_samples, n_features = X.shape
        best_stump = DecisionStump()
        min_error = np.inf

        for f in range(n_features):
            values = np.unique(X[:, f])
            thresholds = (values[:-1] + values[1:]) / 2.0 if len(values) > 1 else values
            for polarity in (+1, -1):
                for thr in thresholds:
                    stump = DecisionStump(f, thr, polarity)
                    preds = stump.predict(X)
                    error = np.dot(w, (preds != y))
                    if error < min_error:
                        min_error = error
                        best_stump = stump
        return best_stump, min_error

    def fit(self, X, y):
        n_samples = X.shape[0]
        w = np.ones(n_samples) / n_samples
        self.alphas, self.stumps = [], []

        for t in range(self.n_estimators):
            stump, error = self._best_stump(X, y, w)
            error = np.clip(error, 1e-10, 0.4999999)
            alpha = 0.5 * np.log((1 - error) / error)
            alpha *= self.learning_rate
            preds = stump.predict(X)
            w *= np.exp(-alpha * y * preds)
            w /= np.sum(w)
            self.alphas.append(alpha)
            self.stumps.append(stump)
        return self

    def decision_function(self, X):
        F = np.zeros(X.shape[0])
        for alpha, stump in zip(self.alphas, self.stumps):
            F += alpha * stump.predict(X)
        return F

    def predict(self, X):
        return np.sign(self.decision_function(X)).astype(int)


# --------------------------------------------------------------------
# Helper utilities
# --------------------------------------------------------------------
def to_pm1(y01):
    """Convert {0,1} labels to {-1,+1}."""
    return np.where(y01 == 1, 1, -1).astype(int)


def prob_from_scores(scores):
    """Map signed scores to probabilities using sigmoid."""
    return 1 / (1 + np.exp(-scores))


def plot_decision_boundary(model, X, y, title, scaler=None, h=0.02):
    """Visualize 2D decision boundaries."""
    if scaler is not None:
        X_plot = scaler.transform(X)
    else:
        X_plot = X
    x_min, x_max = X_plot[:, 0].min() - 1, X_plot[:, 0].max() + 1
    y_min, y_max = X_plot[:, 1].min() - 1, X_plot[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid).reshape(xx.shape)

    plt.figure(figsize=(6, 5))
    plt.contourf(xx, yy, Z, alpha=0.3, levels=[-np.inf, 0, np.inf])
    plt.scatter(X_plot[y == 1, 0], X_plot[y == 1, 1], s=20, label="+1")
    plt.scatter(X_plot[y == -1, 0], X_plot[y == -1, 1], s=20, marker="x", label="-1")
    plt.title(title)
    plt.legend()
    plt.show()


# --------------------------------------------------------------------
# Main execution
# --------------------------------------------------------------------
def main():
    # --- Dataset A: linear-ish ---
    X1, y1 = make_classification(n_samples=600, n_features=2, n_redundant=0, n_informative=2,
                                 n_clusters_per_class=1, class_sep=1.2, flip_y=0.05, random_state=0)
    y1_pm = to_pm1(y1)
    X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1_pm, test_size=0.3, random_state=0)

    scaler1 = StandardScaler().fit(X1_train)
    X1_train_s, X1_test_s = scaler1.transform(X1_train), scaler1.transform(X1_test)

    # --- Dataset B: moons ---
    X2, y2 = make_moons(n_samples=600, noise=0.25, random_state=0)
    y2_pm = to_pm1(y2)
    X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2_pm, test_size=0.3, random_state=0)

    scaler2 = StandardScaler().fit(X2_train)
    X2_train_s, X2_test_s = scaler2.transform(X2_train), scaler2.transform(X2_test)

    # --- Train scratch AdaBoost ---
    model1 = AdaBoostScratch(n_estimators=50, learning_rate=1.0).fit(X1_train_s, y1_train)
    model2 = AdaBoostScratch(n_estimators=150, learning_rate=0.8).fit(X2_train_s, y2_train)

    pred1 = model1.predict(X1_test_s)
    pred2 = model2.predict(X2_test_s)

    print("=== From-scratch AdaBoost ===")
    print("Dataset A accuracy:", accuracy_score(y1_test, pred1))
    print("Dataset B accuracy:", accuracy_score(y2_test, pred2))

    auc1 = roc_auc_score((y1_test == 1).astype(int), prob_from_scores(model1.decision_function(X1_test_s)))
    auc2 = roc_auc_score((y2_test == 1).astype(int), prob_from_scores(model2.decision_function(X2_test_s)))
    print("AUC A:", auc1, " | AUC B:", auc2)

    plot_decision_boundary(model1, X1_train, y1_train, "Scratch AdaBoost (Dataset A)", scaler1)
    plot_decision_boundary(model2, X2_train, y2_train, "Scratch AdaBoost (Dataset B)", scaler2)

    # --- scikit-learn comparison (fixed for >=1.6) ---
    sk_model1 = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators=50,
        learning_rate=1.0,
        random_state=0
    ).fit(X1_train_s, (y1_train == 1).astype(int))

    sk_model2 = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators=150,
        learning_rate=0.8,
        random_state=0
    ).fit(X2_train_s, (y2_train == 1).astype(int))

    sk_pred1 = np.where(sk_model1.predict(X1_test_s) == 1, 1, -1)
    sk_pred2 = np.where(sk_model2.predict(X2_test_s) == 1, 1, -1)

    print("\n=== scikit-learn AdaBoost ===")
    print("Dataset A accuracy:", accuracy_score(y1_test, sk_pred1))
    print("Dataset B accuracy:", accuracy_score(y2_test, sk_pred2))
    print("Confusion A:\n", confusion_matrix(y1_test, sk_pred1))
    print("Confusion B:\n", confusion_matrix(y2_test, sk_pred2))

    plot_decision_boundary(model1, X1_train, y1_train, "scikit-learn AdaBoost (Dataset A)", scaler1)
    plot_decision_boundary(model2, X2_train, y2_train, "scikit-learn AdaBoost (Dataset B)", scaler2)

    print("\nReady: Linear AdaBoost Classifier example finished.")


# --------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------
if __name__ == "__main__":
    main()
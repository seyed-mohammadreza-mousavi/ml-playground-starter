"""
Naive Bayes Classifiers — Complete Implementation
-------------------------------------------------
Implements Gaussian, Multinomial, and Bernoulli Naive Bayes from scratch.
Includes dataset demos (Iris, Digits), evaluation, and decision-boundary visualization.

Run:
    python main.py
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_digits, make_blobs, make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler, Binarizer
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB


# ==========================================================
# 1. From-Scratch Implementations
# ==========================================================

class GaussianNB_Scratch:
    def __init__(self, var_smoothing=1e-9):
        self.var_smoothing = var_smoothing

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y)
        self.classes_ = np.unique(y)
        self.theta_ = []
        self.var_ = []
        self.class_prior_ = []
        for c in self.classes_:
            Xc = X[y == c]
            self.theta_.append(Xc.mean(axis=0))
            self.var_.append(Xc.var(axis=0) + self.var_smoothing)
            self.class_prior_.append(len(Xc) / len(X))
        self.theta_, self.var_, self.class_prior_ = map(np.array, (self.theta_, self.var_, self.class_prior_))
        return self

    def _jll(self, X):
        jll = []
        for i, c in enumerate(self.classes_):
            mean, var = self.theta_[i], self.var_[i]
            log_prob = -0.5 * np.sum(np.log(2 * np.pi * var) + (X - mean) ** 2 / var, axis=1)
            jll.append(np.log(self.class_prior_[i]) + log_prob)
        return np.array(jll).T

    def predict(self, X):
        jll = self._jll(np.asarray(X))
        return self.classes_[np.argmax(jll, axis=1)]


class MultinomialNB_Scratch:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y)
        self.classes_ = np.unique(y)
        n_classes, n_features = len(self.classes_), X.shape[1]
        self.feature_count_ = np.zeros((n_classes, n_features))
        self.class_count_ = np.zeros(n_classes)

        for i, c in enumerate(self.classes_):
            Xc = X[y == c]
            self.feature_count_[i] = Xc.sum(axis=0)
            self.class_count_[i] = Xc.shape[0]

        smoothed_fc = self.feature_count_ + self.alpha
        smoothed_cc = smoothed_fc.sum(axis=1, keepdims=True)
        self.feature_log_prob_ = np.log(smoothed_fc) - np.log(smoothed_cc)
        self.class_log_prior_ = np.log(self.class_count_) - np.log(self.class_count_.sum())
        return self

    def _jll(self, X):
        return self.class_log_prior_ + X @ self.feature_log_prob_.T

    def predict(self, X):
        jll = self._jll(np.asarray(X))
        return self.classes_[np.argmax(jll, axis=1)]


class BernoulliNB_Scratch:
    def __init__(self, alpha=1.0, binarize_threshold=0.0):
        self.alpha = alpha
        self.binarize_threshold = binarize_threshold

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y)
        Xb = (X > self.binarize_threshold).astype(float)
        self.classes_ = np.unique(y)
        n_classes, n_features = len(self.classes_), X.shape[1]
        self.feature_prob_ = np.zeros((n_classes, n_features))
        self.class_count_ = np.zeros(n_classes)

        for i, c in enumerate(self.classes_):
            Xc = Xb[y == c]
            self.feature_prob_[i] = (Xc.sum(axis=0) + self.alpha) / (len(Xc) + 2 * self.alpha)
            self.class_count_[i] = len(Xc)

        self.class_log_prior_ = np.log(self.class_count_) - np.log(self.class_count_.sum())
        return self

    def _jll(self, X):
        Xb = (np.asarray(X) > self.binarize_threshold).astype(float)
        jll = []
        for i, c in enumerate(self.classes_):
            log_theta = np.log(self.feature_prob_[i])
            log_1m = np.log(1 - self.feature_prob_[i])
            ll = (Xb * log_theta + (1 - Xb) * log_1m).sum(axis=1)
            jll.append(self.class_log_prior_[i] + ll)
        return np.array(jll).T

    def predict(self, X):
        jll = self._jll(X)
        return self.classes_[np.argmax(jll, axis=1)]


# ==========================================================
# 2. Visualization Helper
# ==========================================================

def plot_decision_boundary(model, X, y, title):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    plt.figure()
    plt.contourf(xx, yy, Z, alpha=0.25)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=15, edgecolor='k')
    plt.title(title)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()


# ==========================================================
# 3. Main Demonstration
# ==========================================================

def main():
    print("\n=== Gaussian Naive Bayes (Iris Dataset) ===")
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    gnb_s = GaussianNB_Scratch().fit(X_train, y_train)
    y_pred_s = gnb_s.predict(X_test)
    print("Scratch Accuracy:", accuracy_score(y_test, y_pred_s))

    gnb_lib = GaussianNB().fit(X_train, y_train)
    y_pred_lib = gnb_lib.predict(X_test)
    print("Sklearn Accuracy:", accuracy_score(y_test, y_pred_lib))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_lib))

    # ------------------------------------------------------
    print("\n=== Multinomial Naive Bayes (Digits Dataset) ===")
    digits = load_digits()
    Xd, yd = digits.data, digits.target
    Xd = MinMaxScaler().fit_transform(Xd) * 10
    Xtr, Xte, ytr, yte = train_test_split(Xd, yd, test_size=0.3, random_state=42, stratify=yd)

    mnb_s = MultinomialNB_Scratch(alpha=1.0).fit(Xtr, ytr)
    yp_s = mnb_s.predict(Xte)
    print("Scratch Accuracy:", accuracy_score(yte, yp_s))

    mnb_lib = MultinomialNB(alpha=1.0).fit(Xtr, ytr)
    yp_lib = mnb_lib.predict(Xte)
    print("Sklearn Accuracy:", accuracy_score(yte, yp_lib))
    print("Confusion Matrix:\n", confusion_matrix(yte, yp_lib))

    # ------------------------------------------------------
    print("\n=== Bernoulli Naive Bayes (Binarized Digits) ===")
    bin_thresh = 0.5
    Xb = Binarizer(threshold=bin_thresh).fit_transform(MinMaxScaler().fit_transform(digits.data))
    Xtr_b, Xte_b, ytr_b, yte_b = train_test_split(Xb, yd, test_size=0.3, random_state=42, stratify=yd)

    bnb_s = BernoulliNB_Scratch(alpha=1.0).fit(Xtr_b, ytr_b)
    ypb_s = bnb_s.predict(Xte_b)
    print("Scratch Accuracy:", accuracy_score(yte_b, ypb_s))

    bnb_lib = BernoulliNB(alpha=1.0).fit(Xtr_b, ytr_b)
    ypb_lib = bnb_lib.predict(Xte_b)
    print("Sklearn Accuracy:", accuracy_score(yte_b, ypb_lib))
    print("Confusion Matrix:\n", confusion_matrix(yte_b, ypb_lib))

    # ------------------------------------------------------
    print("\n=== Decision Boundary Visualization ===")
    Xb2, yb2 = make_blobs(n_samples=500, centers=3, cluster_std=1.3, random_state=42)
    Xtr_b2, Xte_b2, ytr_b2, yte_b2 = train_test_split(Xb2, yb2, test_size=0.3, random_state=42)
    gnb_vis = GaussianNB_Scratch().fit(Xtr_b2, ytr_b2)
    plot_decision_boundary(gnb_vis, Xte_b2, yte_b2, "Gaussian NB — Decision Boundary")

    # Nonnegative data for Multinomial NB visualization
    X_pos, y_pos = make_classification(n_samples=400, n_features=2, n_informative=2, n_redundant=0,
                                       n_clusters_per_class=1, random_state=7)
    X_pos = MinMaxScaler().fit_transform(X_pos) * 10.0
    Xtr_p, Xte_p, ytr_p, yte_p = train_test_split(X_pos, y_pos, test_size=0.3, random_state=42)
    mnb_vis = MultinomialNB_Scratch(alpha=1.0).fit(Xtr_p, ytr_p)
    plot_decision_boundary(mnb_vis, Xte_p, yte_p, "Multinomial NB — Decision Boundary")

    # Bernoulli NB visualization
    Xb3 = (X_pos > 5.0).astype(float)
    Xtr_b3, Xte_b3, ytr_b3, yte_b3 = train_test_split(Xb3, y_pos, test_size=0.3, random_state=42)
    bnb_vis = BernoulliNB_Scratch(alpha=1.0, binarize_threshold=0.5).fit(Xtr_b3, ytr_b3)
    plot_decision_boundary(bnb_vis, Xte_b3, yte_b3, "Bernoulli NB — Decision Boundary")


# ==========================================================
# 4. Entry Point
# ==========================================================

if __name__ == "__main__":
    main()
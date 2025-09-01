#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SVM Tutorial Runner (main.py)
- Reproduces a comprehensive SVM walkthrough as a single runnable script.
- Saves plots to ./svm_outputs and prints metrics to stdout.
- Uses only matplotlib for charts and ensures one chart per figure, with default colors.
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC, SVR
from sklearn.metrics import (
    accuracy_score, classification_report, ConfusionMatrixDisplay,
    RocCurveDisplay, PrecisionRecallDisplay, roc_auc_score,
    mean_squared_error, r2_score
)
from sklearn.datasets import make_moons, make_circles, make_classification, make_regression

import warnings
warnings.filterwarnings('ignore')
np.random.seed(42)


# ---------- Utilities ----------

def ensure_outdir(dirpath: Path):
    dirpath.mkdir(parents=True, exist_ok=True)


def save_or_show(fig_path: Path, show: bool):
    if fig_path is not None:
        plt.savefig(fig_path, bbox_inches='tight', dpi=150)
    if show:
        plt.show()
    else:
        plt.close()


def plot_decision_function_2d(model, X, y, title, outdir: Path, filename: str, show: bool):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 400),
                         np.linspace(y_min, y_max, 400))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    plt.figure()
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=10)
    plt.title(title)
    plt.xlabel('x1'); plt.ylabel('x2')
    save_or_show(outdir / filename, show)


def make_and_split(dataset='moons', noise=0.3, test_size=0.25):
    if dataset == 'moons':
        X, y = make_moons(n_samples=600, noise=noise, random_state=42)
    elif dataset == 'circles':
        X, y = make_circles(n_samples=600, noise=noise, factor=0.5, random_state=42)
    else:  # 'linear'
        X, y = make_classification(n_samples=800, n_features=2, n_redundant=0,
                                   n_informative=2, n_clusters_per_class=1,
                                   flip_y=0.05, class_sep=1.5, random_state=42)
    return train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)


# ---------- Sections ----------

def sec_linear_svm(outdir: Path, show: bool):
    print("\n[Section] Linear SVM (2D)")
    X_train, X_test, y_train, y_test = make_and_split('linear', noise=0.1)
    clf = Pipeline([('scaler', StandardScaler()), ('clf', LinearSVC(C=1.0))])
    clf.fit(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))
    print(f"Accuracy (LinearSVC): {acc:.4f}")
    X = np.vstack([X_train, X_test]); y = np.hstack([y_train, y_test])
    plot_decision_function_2d(clf, X, y, "Linear SVM decision boundary", outdir, "linear_svm.png", show)


def sec_rbf_svm(outdir: Path, show: bool):
    print("\n[Section] RBF SVM (moons)")
    X_train, X_test, y_train, y_test = make_and_split('moons', noise=0.25)
    clf = Pipeline([('scaler', StandardScaler()), ('clf', SVC(kernel='rbf', C=1.0, gamma='scale'))])
    clf.fit(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))
    print(f"Accuracy (RBF SVC): {acc:.4f}")
    X = np.vstack([X_train, X_test]); y = np.hstack([y_train, y_test])
    plot_decision_function_2d(clf, X, y, "RBF SVM on Two Moons", outdir, "rbf_svm.png", show)


def sec_poly_svm(outdir: Path, show: bool):
    print("\n[Section] Polynomial SVM (circles)")
    X_train, X_test, y_train, y_test = make_and_split('circles', noise=0.15)
    clf = Pipeline([('scaler', StandardScaler()), ('clf', SVC(kernel='poly', degree=3, coef0=1.0, C=1.0, gamma='scale'))])
    clf.fit(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))
    print(f"Accuracy (Poly SVC): {acc:.4f}")
    X = np.vstack([X_train, X_test]); y = np.hstack([y_train, y_test])
    plot_decision_function_2d(clf, X, y, "Polynomial SVM on Circles", outdir, "poly_svm.png", show)


def sec_gridsearch_breast_cancer(outdir: Path, show: bool):
    print("\n[Section] GridSearchCV (Breast Cancer)")
    data = datasets.load_breast_cancer()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
    pipe = Pipeline([('scaler', StandardScaler()), ('clf', SVC(kernel='rbf', probability=True))])
    param_grid = {'clf__C': [0.1, 1, 10], 'clf__gamma': ['scale', 0.01, 0.001]}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(pipe, param_grid, scoring='roc_auc', cv=cv, n_jobs=-1)
    grid.fit(X_train, y_train)
    print("Best params:", grid.best_params_)
    y_proba = grid.best_estimator_.predict_proba(X_test)[:, 1]
    print("ROC-AUC:", roc_auc_score(y_test, y_proba))
    ConfusionMatrixDisplay.from_predictions(y_test, grid.predict(X_test))
    plt.title("Confusion Matrix (RBF SVM)")
    save_or_show(outdir / "bc_confusion.png", show)


def sec_multiclass_iris(outdir: Path, show: bool):
    print("\n[Section] Multiclass (Iris): OvO vs OvR")
    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
    ovo = Pipeline([('scaler', StandardScaler()), ('clf', SVC(kernel='rbf'))])
    ovr = Pipeline([('scaler', StandardScaler()), ('clf', LinearSVC(C=1.0))])
    for name, clf in [("OvO", ovo), ("OvR", ovr)]:
        clf.fit(X_train, y_train)
        acc = accuracy_score(y_test, clf.predict(X_test))
        print(f"{name} accuracy: {acc:.4f}")


def sec_svr(outdir: Path, show: bool):
    print("\n[Section] SVR Regression")
    X, y = make_regression(n_samples=1000, n_features=8, noise=15.0, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    svr = Pipeline([('scaler', StandardScaler()), ('svr', SVR(kernel='rbf', C=10.0, epsilon=0.1))])
    svr.fit(X_train, y_train)
    y_pred = svr.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print("SVR RMSE:", rmse)
    print("SVR R2:", r2_score(y_test, y_pred))
    plt.scatter(y_test, y_pred, s=10)
    plt.xlabel("Actual"); plt.ylabel("Predicted")
    plt.title("SVR Predicted vs Actual")
    save_or_show(outdir / "svr.png", show)


# ---------- Main ----------

ALL_SECTIONS = {
    "linear": sec_linear_svm,
    "rbf": sec_rbf_svm,
    "poly": sec_poly_svm,
    "grid": sec_gridsearch_breast_cancer,
    "multi": sec_multiclass_iris,
    "svr": sec_svr,
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sections", nargs="*", default=list(ALL_SECTIONS.keys()))
    parser.add_argument("--outdir", type=str, default="svm_outputs")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    ensure_outdir(outdir)
    show = not args.no_show

    for sec in args.sections:
        if sec in ALL_SECTIONS:
            ALL_SECTIONS[sec](outdir, show)
        else:
            print("Unknown section:", sec)

    print("\nDone. Figures saved in", outdir.resolve())

if __name__ == "__main__":
    main()

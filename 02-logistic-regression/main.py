"""
main.py — Complete Logistic Regression Tutorial (CLI)

Usage (all steps):
    python main.py --all

Or run specific parts, e.g.:
    python main.py --from_scratch --sklearn --plots --tune --interpret --regpath --poly --calibrate --imbalance --multiclass --save

Plots and artifacts are saved to ./02-logistic-regression/outputs by default.
"""

import os
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification, make_moons, load_iris
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, roc_auc_score,
    average_precision_score, precision_recall_curve,
    classification_report
)
import joblib


# -------------------------
# Global config / utils
# -------------------------
RANDOM_STATE = 42
np.set_printoptions(suppress=True, precision=4)
pd.set_option("display.max_columns", 100)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)


def savefig(name: str):
    """Save current matplotlib figure to outputs/NAME.png"""
    path = OUT / f"{name}.png"
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    print(f"[saved] {path}")
    plt.close()


def print_header(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


# -------------------------
# From-scratch logistic regression (NumPy)
# -------------------------
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def log_loss_np(y_true, y_prob, l2, w):
    eps = 1e-15
    y_prob = np.clip(y_prob, eps, 1 - eps)
    ce = -(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob)).mean()
    reg = 0.5 * l2 * np.sum(w * w)
    return ce + reg


def predict_proba_np_scaled(Xs, w, b):
    return sigmoid(Xs @ w + b)


def fit_logreg_gd(
    X, y, lr=0.2, l2=0.01, n_iter=5000, tol=1e-6,
    early_stopping=True, patience=100, verbose=True
):
    n, d = X.shape
    # Standardize
    mu = X.mean(axis=0)
    sigma = X.std(axis=0) + 1e-12
    Xs = (X - mu) / sigma

    rng = np.random.default_rng(RANDOM_STATE)
    w = rng.normal(scale=0.01, size=d)
    b = 0.0

    best_loss = np.inf
    best_wb = (w.copy(), b)
    patience_left = patience
    history = []

    for t in range(1, n_iter + 1):
        p = predict_proba_np_scaled(Xs, w, b)
        loss = log_loss_np(y, p, l2, w)
        history.append(loss)

        grad_p = (p - y) / n
        grad_w = Xs.T @ grad_p + l2 * w
        grad_b = grad_p.sum()

        w -= lr * grad_w
        b -= lr * grad_b

        if early_stopping:
            if loss + tol < best_loss:
                best_loss = loss
                best_wb = (w.copy(), b)
                patience_left = patience
            else:
                patience_left -= 1
                if patience_left <= 0:
                    if verbose:
                        print(f"[from_scratch] early stop at iter={t}, best_loss={best_loss:.6f}")
                    break

    w, b = best_wb
    model = {"w": w, "b": b, "mu": mu, "sigma": sigma, "l2": l2, "history": np.array(history)}
    return model


def predict_proba_model(model, X):
    Xs = (X - model["mu"]) / model["sigma"]
    return predict_proba_np_scaled(Xs, model["w"], model["b"])


# -------------------------
# Plots
# -------------------------
def plot_loss(history):
    plt.figure()
    plt.plot(history)
    plt.title("From-scratch Logistic Regression — Training Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    savefig("loss_from_scratch")


def plot_roc_pr(y_true, y_prob, prefix=""):
    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC AUC={auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    savefig(f"roc_{prefix}" if prefix else "roc")

    # PR
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    plt.figure()
    plt.plot(rec, prec, label=f"AP={ap:.3f}")
    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    savefig(f"pr_{prefix}" if prefix else "pr")


def plot_decision_boundary_2d(clf, X, y, title="Decision Boundary", name="decision_boundary"):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300)
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    probs = clf.predict_proba(grid)[:, 1].reshape(xx.shape)
    plt.figure()
    plt.contourf(xx, yy, probs, levels=25, alpha=0.6)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=15, edgecolors="k", linewidths=0.2)
    plt.title(title)
    plt.xlabel("x0")
    plt.ylabel("x1")
    savefig(name)


def plot_regularization_path(Xtr, ytr, Cs, name="regularization_path"):
    coefs = []
    for C in Cs:
        lr = Pipeline([
            ("scaler", StandardScaler()),
            ("logreg", LogisticRegression(C=C, solver="lbfgs", max_iter=2000, random_state=RANDOM_STATE))
        ])
        lr.fit(Xtr, ytr)
        coefs.append(lr.named_steps["logreg"].coef_.ravel())
    coefs = np.array(coefs)

    plt.figure()
    for j in range(coefs.shape[1]):
        plt.plot(Cs, coefs[:, j], label=f"x{j}")
    plt.xscale("log")
    plt.xlabel("C (inverse regularization strength)")
    plt.ylabel("Coefficient value")
    plt.title("Regularization Path")
    plt.legend()
    savefig(name)


# -------------------------
# Workflows
# -------------------------
def make_datasets():
    # Linear-ish dataset
    X_lin, y_lin = make_classification(
        n_samples=1200, n_features=5, n_informative=3, n_redundant=0, n_repeated=0,
        n_clusters_per_class=2, class_sep=1.5, flip_y=0.03, random_state=RANDOM_STATE
    )
    # 2D moons for boundary
    X_2d, y_2d = make_moons(n_samples=800, noise=0.2, random_state=RANDOM_STATE)
    return (X_lin, y_lin), (X_2d, y_2d)


def run_from_scratch(Xtr, Xte, ytr, yte, do_plots=True):
    print_header("From-scratch Logistic Regression (NumPy)")
    model = fit_logreg_gd(Xtr, ytr, lr=0.2, l2=0.01, n_iter=5000, patience=100, verbose=True)

    proba_tr = predict_proba_model(model, Xtr)
    proba_te = predict_proba_model(model, Xte)

    for split, y_true, y_prob in [("Train", ytr, proba_tr), ("Test", yte, proba_te)]:
        yhat = (y_prob >= 0.5).astype(int)
        acc = accuracy_score(y_true, yhat)
        prec = precision_score(y_true, yhat)
        rec = recall_score(y_true, yhat)
        f1 = f1_score(y_true, yhat)
        auc = roc_auc_score(y_true, y_prob)
        print(f"{split}: acc={acc:.3f} prec={prec:.3f} rec={rec:.3f} f1={f1:.3f} roc_auc={auc:.3f}")

    print("\nConfusion matrix (Test):")
    print(confusion_matrix(yte, (proba_te >= 0.5).astype(int)))
    print("\nClassification report (Test):")
    print(classification_report(yte, (proba_te >= 0.5).astype(int)))

    if do_plots:
        plot_loss(model["history"])
        plot_roc_pr(yte, proba_te, prefix="from_scratch")


def run_sklearn(Xtr, Xte, ytr, yte, do_plots=True):
    print_header("scikit-learn Pipeline Baseline")
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(random_state=RANDOM_STATE, max_iter=2000))
    ])
    pipe.fit(Xtr, ytr)
    yprob_te = pipe.predict_proba(Xte)[:, 1]
    yhat_te = (yprob_te >= 0.5).astype(int)

    print(f"Test accuracy: {accuracy_score(yte, yhat_te):.3f}")
    print(f"ROC AUC: {roc_auc_score(yte, yprob_te):.3f}")

    if do_plots:
        plot_roc_pr(yte, yprob_te, prefix="sklearn")

    return pipe


def run_tuning(Xtr, Xte, ytr, yte):
    print_header("Hyperparameter Tuning (GridSearchCV)")
    param_grid = {
        "logreg__C": [0.01, 0.1, 1.0, 10.0],
        "logreg__penalty": ["l2"],
        "logreg__solver": ["lbfgs", "liblinear"]
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    grid = GridSearchCV(
        Pipeline([("scaler", StandardScaler()),
                  ("logreg", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE))]),
        param_grid=param_grid, cv=cv, scoring="roc_auc", n_jobs=None
    )
    grid.fit(Xtr, ytr)
    print("Best params:", grid.best_params_)
    print(f"Best CV ROC AUC: {grid.best_score_:.3f}")
    test_auc = roc_auc_score(yte, grid.best_estimator_.predict_proba(Xte)[:, 1])
    print(f"Test ROC AUC (best): {test_auc:.3f}")
    return grid.best_estimator_


def run_interpretation(pipe, Xtr, ytr):
    print_header("Model Interpretation — Coefficients & Odds Ratios")
    pipe_small = Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(C=1.0, penalty="l2", solver="lbfgs",
                                     max_iter=2000, random_state=RANDOM_STATE))
    ])
    pipe_small.fit(Xtr, ytr)
    coef = pipe_small.named_steps["logreg"].coef_.ravel()
    features = [f"x{i}" for i in range(Xtr.shape[1])]
    odds_ratios = np.exp(coef)
    interp_df = pd.DataFrame({"feature": features, "coef": coef, "odds_ratio": odds_ratios}).sort_values(
        "coef", ascending=False
    ).reset_index(drop=True)
    print(interp_df.to_string(index=False))


def run_reg_path(Xtr, ytr):
    print_header("Regularization Path (vary C)")
    Cs = np.logspace(-3, 2, 12)
    plot_regularization_path(Xtr, ytr, Cs, name="regularization_path")


def run_poly_boundary(X2d, y2d):
    print_header("Polynomial Features + Decision Boundary (2D Moons)")
    poly_pipe = Pipeline([
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("scaler", StandardScaler(with_mean=False)),
        ("logreg", LogisticRegression(max_iter=3000, random_state=RANDOM_STATE))
    ])
    poly_pipe.fit(X2d, y2d)
    plot_decision_boundary_2d(
        poly_pipe, X2d, y2d,
        title="Logistic Regression + Polynomial Features on Moons",
        name="decision_boundary_poly_moons"
    )


def run_calibration(pipe, Xte, yte):
    print_header("Calibration Curve")
    plt.figure()
    CalibrationDisplay.from_estimator(pipe, Xte, yte, n_bins=10)
    plt.title("Calibration Curve — Logistic Regression")
    savefig("calibration_curve")


def run_imbalance_demo():
    print_header("Class Imbalance Demo (AP)")
    Xi, yi = make_classification(
        n_samples=2000, n_features=6, weights=[0.9, 0.1],
        n_informative=4, class_sep=1.0, random_state=RANDOM_STATE
    )
    Xi_tr, Xi_te, yi_tr, yi_te = train_test_split(Xi, yi, test_size=0.25, stratify=yi, random_state=RANDOM_STATE)

    base = Pipeline([("scaler", StandardScaler()),
                     ("logreg", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE))])
    base.fit(Xi_tr, yi_tr)
    probi = base.predict_proba(Xi_te)[:, 1]
    print("No class weights — AP:", f"{average_precision_score(yi_te, probi):.3f}")

    bal = Pipeline([("scaler", StandardScaler()),
                    ("logreg", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE))])
    bal.fit(Xi_tr, yi_tr)
    probi_bal = bal.predict_proba(Xi_te)[:, 1]
    print("Balanced class weights — AP:", f"{average_precision_score(yi_te, probi_bal):.3f}")


def run_multiclass():
    print_header("Multiclass Logistic Regression (Iris)")
    iris = load_iris()
    Xi, yi = iris.data, iris.target
    Xi_tr, Xi_te, yi_tr, yi_te = train_test_split(Xi, yi, test_size=0.25, stratify=yi, random_state=RANDOM_STATE)

    multi = Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(multi_class="ovr", max_iter=2000, random_state=RANDOM_STATE))
    ])
    multi.fit(Xi_tr, yi_tr)
    pred = multi.predict(Xi_te)
    print("Multiclass accuracy (Iris):", f"{accuracy_score(yi_te, pred):.3f}")
    print("\nClassification report:\n", classification_report(yi_te, pred))


def run_save_load(model, Xte, yte, name="logreg_pipeline.joblib"):
    print_header("Save & Load Model")
    path = OUT / name
    joblib.dump(model, path)
    print(f"Saved to: {path}")
    loaded = joblib.load(path)
    auc = roc_auc_score(yte, loaded.predict_proba(Xte)[:, 1])
    print(f"Loaded model test ROC AUC: {auc:.3f}")


# -------------------------
# Main
# -------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Complete Logistic Regression Tutorial (CLI)")
    p.add_argument("--all", action="store_true", help="Run everything.")
    p.add_argument("--from_scratch", action="store_true", help="Run NumPy from-scratch training.")
    p.add_argument("--sklearn", action="store_true", help="Run sklearn pipeline baseline.")
    p.add_argument("--tune", action="store_true", help="Run GridSearchCV tuning.")
    p.add_argument("--interpret", action="store_true", help="Print coefficients & odds ratios.")
    p.add_argument("--regpath", action="store_true", help="Plot regularization path.")
    p.add_argument("--poly", action="store_true", help="Decision boundary with polynomial features on moons.")
    p.add_argument("--calibrate", action="store_true", help="Plot calibration curve (uses sklearn model).")
    p.add_argument("--imbalance", action="store_true", help="Run class imbalance AP demo.")
    p.add_argument("--multiclass", action="store_true", help="Run multiclass demo on Iris.")
    p.add_argument("--save", action="store_true", help="Save & reload best sklearn model.")
    return p.parse_args()


def main():
    args = parse_args()

    # Datasets
    (X_lin, y_lin), (X_2d, y_2d) = make_datasets()
    Xtr, Xte, ytr, yte = train_test_split(X_lin, y_lin, test_size=0.25, stratify=y_lin, random_state=RANDOM_STATE)

    # Defaults for chaining
    best_sklearn = None

    if args.all or args.from_scratch:
        run_from_scratch(Xtr, Xte, ytr, yte, do_plots=True)

    if args.all or args.sklearn:
        best_sklearn = run_sklearn(Xtr, Xte, ytr, yte, do_plots=True)

    if args.all or args.tune:
        best_sklearn = run_tuning(Xtr, Xte, ytr, yte)

    if args.all or args.interpret:
        run_interpretation(best_sklearn, Xtr, ytr) if best_sklearn else run_interpretation(None, Xtr, ytr)

    if args.all or args.regpath:
        run_reg_path(Xtr, ytr)

    if args.all or args.poly:
        run_poly_boundary(X_2d, y_2d)

    if args.all or args.calibrate:
        # Ensure we have a fitted sklearn model
        if best_sklearn is None:
            best_sklearn = run_sklearn(Xtr, Xte, ytr, yte, do_plots=False)
        run_calibration(best_sklearn, Xte, yte)

    if args.all or args.imbalance:
        run_imbalance_demo()

    if args.all or args.multiclass:
        run_multiclass()

    if args.all or args.save:
        if best_sklearn is None:
            best_sklearn = run_sklearn(Xtr, Xte, ytr, yte, do_plots=False)
        run_save_load(best_sklearn, Xte, yte)


if __name__ == "__main__":
    main()

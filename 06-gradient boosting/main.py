#!/usr/bin/env python3
"""
Gradient Boosting Models — XGBoost & LightGBM
Comprehensive demo with classification, regression, and early stopping
Author: Seyed Mohammadreza Mousavi
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    mean_squared_error,
    r2_score,
)
from sklearn.inspection import PartialDependenceDisplay

import xgboost as xgb
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import (
    LGBMClassifier,
    LGBMRegressor,
    early_stopping,
    log_evaluation,
)


# ---------------------------------------------------------------------
# Visualization helper
# ---------------------------------------------------------------------
def decision_boundary(model, X, y, title="Decision Boundary", proba=False):
    """Visualize 2D decision boundaries."""
    X = np.asarray(X)
    y = np.asarray(y)
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 400),
        np.linspace(y_min, y_max, 400),
    )

    if proba and hasattr(model, "predict_proba"):
        Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, levels=20, cmap="coolwarm", alpha=0.7)
    else:
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        cmap_light = ListedColormap([[0.85, 0.92, 1.0], [1.0, 0.88, 0.88]])
        plt.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.6)

    cmap_bold = ListedColormap([[0.2, 0.4, 0.8], [0.8, 0.2, 0.2]])
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor="k", s=25)
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------
# Classification demo
# ---------------------------------------------------------------------
def classification_demo():
    print("\n=== Classification Demo ===")
    X, y = make_classification(
        n_samples=1000,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        n_clusters_per_class=1,
        class_sep=1.8,
        random_state=42,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # XGBoost
    xgb_model = XGBClassifier(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=-1,
        random_state=42,
    )
    xgb_model.fit(X_train, y_train)
    yp_xgb = xgb_model.predict(X_test)
    print(f"XGBoost Accuracy: {accuracy_score(y_test, yp_xgb):.3f}")
    decision_boundary(xgb_model, X_test, y_test, "XGBoost Decision Boundary", proba=True)

    # LightGBM
    lgb_model = LGBMClassifier(
        n_estimators=400,
        max_depth=5,
        num_leaves=15,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=0.5,
        random_state=42,
        verbosity=-1,
    )
    lgb_model.fit(X_train, y_train)
    yp_lgb = lgb_model.predict(X_test)
    print(f"LightGBM Accuracy: {accuracy_score(y_test, yp_lgb):.3f}")
    decision_boundary(lgb_model, X_test, y_test, "LightGBM Decision Boundary", proba=True)

    # Feature importance + PDP
    X_train_df = pd.DataFrame(X_train, columns=["x1", "x2"])
    for model, name in zip([xgb_model, lgb_model], ["XGBoost", "LightGBM"]):
        model.fit(X_train_df, y_train)
        fi = model.feature_importances_
        plt.bar(["x1", "x2"], fi)
        plt.title(f"{name} Feature Importance")
        plt.show()

        PartialDependenceDisplay.from_estimator(model, X_train_df, features=["x1", "x2"])
        plt.suptitle(f"{name} Partial Dependence (x1, x2)")
        plt.tight_layout()
        plt.show()


# ---------------------------------------------------------------------
# Regression demo
# ---------------------------------------------------------------------
def regression_demo():
    print("\n=== Regression Demo ===")
    rng = np.random.RandomState(42)
    Xr = np.linspace(-3, 3, 600).reshape(-1, 1)
    yr_true = np.sinh(Xr).ravel() + 0.5 * np.cos(3 * Xr).ravel()
    yr = yr_true + rng.normal(0, 0.5, size=yr_true.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        Xr, yr, test_size=0.25, random_state=42
    )

    xgbr = XGBRegressor(
        n_estimators=800,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        n_jobs=-1,
        random_state=42,
    )
    xgbr.fit(X_train, y_train)
    y_pred_xgb = xgbr.predict(X_test)

    lgbr = LGBMRegressor(
        n_estimators=1200,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=-1,
    )
    lgbr.fit(X_train, y_train)
    y_pred_lgb = lgbr.predict(X_test)

    def metrics(y_true, y_pred):
        rmse = mean_squared_error(y_true, y_pred, squared=False)
        r2 = r2_score(y_true, y_pred)
        return rmse, r2

    rmse_x, r2_x = metrics(y_test, y_pred_xgb)
    rmse_l, r2_l = metrics(y_test, y_pred_lgb)
    print(f"XGBoost — RMSE: {rmse_x:.3f}, R²: {r2_x:.3f}")
    print(f"LightGBM — RMSE: {rmse_l:.3f}, R²: {r2_l:.3f}")

    # Plot
    sort_idx = np.argsort(X_test.ravel())
    plt.figure(figsize=(9, 5))
    plt.scatter(X_train, y_train, s=10, alpha=0.25, label="Train")
    plt.scatter(X_test, y_test, s=15, alpha=0.6, label="Test")
    plt.plot(X_test[sort_idx], y_pred_xgb[sort_idx], lw=2, label="XGBoost", color="#1f77b4")
    plt.plot(X_test[sort_idx], y_pred_lgb[sort_idx], lw=2, label="LightGBM", color="#2ca02c")
    plt.title("Regression: Predictions vs Noisy Truth")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------
# Early Stopping demo
# ---------------------------------------------------------------------
def early_stopping_demo():
    print("\n=== Early Stopping Demo ===")
    X, y = make_classification(
        n_samples=1200,
        n_features=20,
        n_informative=8,
        n_redundant=4,
        random_state=42,
    )
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # XGBoost (use low-level API for 2.x)
    xgb_es = XGBClassifier(
        n_estimators=2000,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        n_jobs=-1,
        random_state=42,
    )

    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dval = xgb.DMatrix(X_val, label=y_val)

    evals_result = {}
    xgb_model = xgb.train(
        params=xgb_es.get_xgb_params(),
        dtrain=dtrain,
        num_boost_round=2000,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=50,
        evals_result=evals_result,
        verbose_eval=False,
    )

    print(f"XGBoost best_iteration: {xgb_model.best_iteration}")

    # LightGBM
    lgb_es = LGBMClassifier(
        n_estimators=4000,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=-1,
    )

    lgb_es.fit(
        X_tr,
        y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[
            early_stopping(stopping_rounds=50),
            log_evaluation(0),
        ],
    )

    print(f"LightGBM best_iteration_: {lgb_es.best_iteration_}")


# ---------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------
def main():
    classification_demo()
    regression_demo()
    early_stopping_demo()


if __name__ == "__main__":
    main()
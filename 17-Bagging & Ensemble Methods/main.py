"""
Bagging & Ensemble Methods — main.py
------------------------------------
Full demonstration of:
  - BaggingClassifier, RandomForestClassifier (classification)
  - BaggingRegressor, RandomForestRegressor (regression)
with proper math background and visualizations.

Author: Your Name
"""

from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, load_breast_cancer, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, BaggingRegressor, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from sklearn.metrics import mean_squared_error, r2_score
import inspect


def plot_decision_boundary(clf, X, y, ax, title=''):
    """Utility to visualize 2D decision boundaries."""
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.25)
    ax.scatter(X[:, 0], X[:, 1], c=y, s=10, alpha=0.8)
    ax.set_title(title)


def classification_demo():
    """Run Bagging and Random Forest on synthetic and real datasets."""
    print("\n=== Classification: Moons Dataset ===")
    X, y = make_moons(n_samples=800, noise=0.3, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Compatibility for old/new sklearn
    bag_param = "estimator" if "estimator" in inspect.signature(BaggingClassifier).parameters else "base_estimator"

    models = {
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Bagging (Tree base)': BaggingClassifier(
            **{bag_param: DecisionTreeClassifier(random_state=42)},
            n_estimators=200, bootstrap=True, oob_score=True, random_state=42
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=200, oob_score=True, n_jobs=-1, random_state=42
        )
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (name, clf) in zip(axes, models.items()):
        clf.fit(X_train, y_train)
        plot_decision_boundary(clf, X_train, y_train, ax,
                               title=f"{name}\nTrain acc={clf.score(X_train, y_train):.3f}")
    plt.tight_layout()
    plt.show()

    print("\n=== Test Accuracy & OOB Scores ===")
    for name, clf in models.items():
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        oob = getattr(clf, "oob_score_", None)
        print(f"{name:18s} | Test acc: {acc:.3f} | OOB: {oob if oob is not None else '—'}")

    print("\n=== Real Dataset: Breast Cancer ===")
    data = load_breast_cancer()
    X_bc, y_bc = data.data, data.target
    Xtr, Xte, ytr, yte = train_test_split(X_bc, y_bc, test_size=0.25, random_state=42, stratify=y_bc)

    rf = RandomForestClassifier(n_estimators=400, oob_score=True, n_jobs=-1, random_state=42)
    rf.fit(Xtr, ytr)
    yp = rf.predict(Xte)
    print("Random Forest (Breast Cancer)")
    print("Test accuracy:", accuracy_score(yte, yp))
    print("OOB score:", rf.oob_score_)

    ConfusionMatrixDisplay.from_estimator(rf, Xte, yte)
    plt.title("Confusion Matrix — Random Forest (Breast Cancer)")
    plt.show()

    importances = rf.feature_importances_
    idx = np.argsort(importances)[-15:]
    plt.barh(range(len(idx)), importances[idx])
    plt.yticks(range(len(idx)), [data.feature_names[i] for i in idx])
    plt.xlabel("Mean decrease in impurity (MDI)")
    plt.title("Top Feature Importances — Random Forest")
    plt.tight_layout()
    plt.show()


def regression_demo():
    """Run Bagging and Random Forest for regression (Diabetes dataset)."""
    print("\n=== Regression: Diabetes Dataset ===")
    X, y = load_diabetes(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42)

    # Compatibility check
    bag_param = "estimator" if "estimator" in inspect.signature(BaggingRegressor).parameters else "base_estimator"

    tree = DecisionTreeRegressor(random_state=42)
    bag = BaggingRegressor(
        **{bag_param: DecisionTreeRegressor(random_state=42)},
        n_estimators=200, oob_score=True, bootstrap=True, random_state=42
    )
    rf = RandomForestRegressor(n_estimators=400, oob_score=True, n_jobs=-1, random_state=42)

    for name, model in [("Tree", tree), ("Bagging", bag), ("Random Forest", rf)]:
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        rmse = sqrt(mean_squared_error(yte, pred))  # version-safe RMSE
        r2 = r2_score(yte, pred)
        oob = getattr(model, "oob_score_", None)
        print(f"{name:14s} | RMSE={rmse:.3f} | R2={r2:.3f} | OOB={oob if oob is not None else '—'}")

    # OOB vs. number of trees (RF)
    print("\n=== OOB Score vs. n_estimators (Random Forest) ===")
    oobs, ns = [], list(range(10, 401, 40))
    for n in ns:
        rf_tmp = RandomForestRegressor(n_estimators=n, oob_score=True, n_jobs=-1, random_state=42)
        rf_tmp.fit(Xtr, ytr)
        oobs.append(rf_tmp.oob_score_)

    plt.plot(ns, oobs, marker="o")
    plt.xlabel("Number of trees (n_estimators)")
    plt.ylabel("OOB R²")
    plt.title("OOB Score vs. Number of Trees — Random Forest (Diabetes)")
    plt.show()


def main():
    """Main entry point."""
    print("==== Bagging & Ensemble Methods ====")
    classification_demo()
    regression_demo()
    print("\n✅ All experiments completed successfully.")


if __name__ == "__main__":
    main()
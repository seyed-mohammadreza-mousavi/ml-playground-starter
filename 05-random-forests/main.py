"""
Random Forest Regression Demo
Author: Seyed Mohammadreza Mousavi
Description:
    Demonstrates how to train, evaluate, and visualize a Random Forest Regressor
    with scikit-learn, including RMSE and R² metrics.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.inspection import PartialDependenceDisplay


def train_random_forest_regression(n_samples=1200, n_features=6, noise=15.0, random_state=42):
    """Generate synthetic regression data and train a Random Forest regressor."""
    # Generate data
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=4,
        noise=noise,
        random_state=random_state,
    )

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=random_state
    )

    # Initialize model
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        max_features="sqrt",
        n_jobs=-1,
        random_state=random_state,
    )

    # Train model
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    return model, X_test, y_test, y_pred


def evaluate_model(y_test, y_pred):
    """Compute RMSE and R²."""
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    return rmse, r2


def plot_results(y_test, y_pred):
    """Plot predicted vs. true values."""
    plt.figure(figsize=(6, 5))
    plt.scatter(y_test, y_pred, s=15, alpha=0.7, edgecolor='k')
    m = min(y_test.min(), y_pred.min())
    M = max(y_test.max(), y_pred.max())
    plt.plot([m, M], [m, M], 'r--', lw=2)
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.title("Random Forest Regression — Predicted vs True")
    plt.tight_layout()
    plt.show()


def plot_partial_dependence(model, X_train):
    """Plot partial dependence for the first two features."""
    fig = plt.figure(figsize=(10, 4))
    PartialDependenceDisplay.from_estimator(model, X_train, features=[0, 1])
    plt.suptitle("Partial Dependence (features 0 and 1)")
    plt.tight_layout()
    plt.show()


def main():
    print("Training Random Forest Regressor...")
    model, X_test, y_test, y_pred = train_random_forest_regression()

    print("Evaluating model...")
    rmse, r2 = evaluate_model(y_test, y_pred)
    print(f"RMSE: {rmse:.2f}")
    print(f"R²:   {r2:.3f}")

    print("Visualizing results...")
    plot_results(y_test, y_pred)
    plot_partial_dependence(model, X_test)


if __name__ == "__main__":
    main()

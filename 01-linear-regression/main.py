"""
main.py - Linear Regression Example
-----------------------------------
This script demonstrates Linear Regression:
1. From scratch using Gradient Descent
2. Using scikit-learn's LinearRegression

Author: Your Name
Repo: ml-playground-starter
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def generate_data(n_samples=100, seed=42):
    np.random.seed(seed)
    X = 2 * np.random.rand(n_samples, 1)
    y = 4 + 3 * X + np.random.randn(n_samples, 1)
    return X, y


def linear_regression_scratch(X, y, lr=0.1, epochs=1000):
    w, b = 0.0, 0.0
    n = len(X)

    for _ in range(epochs):
        y_pred = w * X + b
        dw = (-2 / n) * np.sum(X * (y - y_pred))
        db = (-2 / n) * np.sum(y - y_pred)
        w -= lr * dw
        b -= lr * db
    return w, b


def plot_results(X, y, w, b, sklearn_model=None):
    plt.scatter(X, y, color="blue", alpha=0.6, label="Data")

    # scratch fit
    plt.plot(X, w * X + b, color="red", linewidth=2, label="Scratch Fit")

    # sklearn fit
    if sklearn_model:
        plt.plot(X, sklearn_model.predict(X), color="green", linewidth=2, label="sklearn Fit")

    plt.xlabel("X")
    plt.ylabel("y")
    plt.legend()
    plt.title("Linear Regression Comparison")
    plt.show()


def main():
    # Generate synthetic data
    X, y = generate_data()

    # Train from scratch
    w, b = linear_regression_scratch(X, y)
    print(f"[Scratch] w = {w:.2f}, b = {b:.2f}")

    # Train with scikit-learn
    model = LinearRegression()
    model.fit(X, y)
    print(f"[sklearn] w = {model.coef_[0][0]:.2f}, b = {model.intercept_[0]:.2f}")

    # Plot results
    plot_results(X, y, w, b, sklearn_model=model)

    # Prediction example
    X_new = np.array([[0], [2]])
    y_pred = model.predict(X_new)
    print(f"Predictions for {X_new.flatten()}: {y_pred.flatten()}")

if __name__ == "__main__":
    main()


"""
K-Nearest Neighbors (KNN) from Scratch
--------------------------------------
This script implements KNN from scratch with visualization and comparison to scikit-learn.
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

class KNN:
    """A simple K-Nearest Neighbors classifier implemented from scratch."""

    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def _euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def predict(self, X):
        return np.array([self._predict(x) for x in X])

    def _predict(self, x):
        distances = [self._euclidean_distance(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_labels = [self.y_train[i] for i in k_indices]
        most_common = Counter(k_labels).most_common(1)
        return most_common[0][0]


def plot_decision_boundary(model, X, y, title="Decision Boundary (KNN)"):
    """Plot decision boundaries for a 2D dataset."""
    h = 0.1
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid)
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(7, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap='coolwarm')
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()


def main():
    """Main function to demonstrate KNN from scratch and with scikit-learn."""
    np.random.seed(42)

    # Generate synthetic 2D dataset
    X, y = make_classification(n_samples=200, n_features=2, n_informative=2,
                               n_redundant=0, n_clusters_per_class=1, random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Plot dataset
    plt.figure(figsize=(6, 5))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='k')
    plt.title("Synthetic 2D Dataset")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

    # Train and evaluate custom KNN
    knn = KNN(k=5)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    acc_scratch = accuracy_score(y_test, y_pred)
    print(f"Accuracy (Scratch KNN): {acc_scratch:.3f}")

    # Visualize decision boundary (scratch)
    plot_decision_boundary(knn, X, y, title="Decision Boundary - KNN (from Scratch)")

    # Compare with scikit-learn implementation
    sk_knn = KNeighborsClassifier(n_neighbors=5)
    sk_knn.fit(X_train, y_train)
    y_pred_lib = sk_knn.predict(X_test)
    acc_lib = accuracy_score(y_test, y_pred_lib)
    print(f"Accuracy (scikit-learn KNN): {acc_lib:.3f}")

    plot_decision_boundary(sk_knn, X, y, title="Decision Boundary - KNN (scikit-learn)")

    print("\nSummary:")
    print("- KNN is a non-parametric, instance-based algorithm.")
    print("- Works well for small, low-dimensional datasets.")
    print("- Computationally expensive for large data.\n")


if __name__ == "__main__":
    main()

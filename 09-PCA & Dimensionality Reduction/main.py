
"""
Principal Component Analysis (PCA) from Scratch
-----------------------------------------------
This script demonstrates PCA step-by-step — from mathematical intuition
to a working implementation with visualization and comparison to scikit-learn.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA as SKPCA

class PCA_Scratch:
    """Principal Component Analysis implemented from scratch using NumPy."""

    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, X):
        # Standardize (center) the data
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        # Compute covariance matrix
        cov = np.cov(X_centered, rowvar=False)

        # Eigen decomposition
        eig_vals, eig_vecs = np.linalg.eigh(cov)

        # Sort by descending eigenvalues
        sorted_idx = np.argsort(eig_vals)[::-1]
        eig_vals = eig_vals[sorted_idx]
        eig_vecs = eig_vecs[:, sorted_idx]

        # Store components and explained variance
        self.components = eig_vecs[:, :self.n_components]
        self.explained_variance = eig_vals[:self.n_components]
        self.explained_variance_ratio = eig_vals[:self.n_components] / np.sum(eig_vals)

    def transform(self, X):
        X_centered = X - self.mean
        return np.dot(X_centered, self.components)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


def plot_2d_projection(X_proj, y, title, labels):
    """Plot 2D projection of dataset after PCA."""
    plt.figure(figsize=(8,6))
    colors = ['red', 'green', 'blue']
    for label, color in zip(np.unique(y), colors):
        plt.scatter(X_proj[y == label, 0], X_proj[y == label, 1],
                    color=color, alpha=0.7, label=labels[label])
    plt.title(title)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()


def main():
    """Main function to demonstrate PCA from scratch and compare with sklearn."""
    np.random.seed(42)

    # Load dataset
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names

    # Standardize
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    # PCA from scratch
    pca = PCA_Scratch(n_components=2)
    X_pca = pca.fit_transform(X_std)

    print("Explained variance ratio (scratch):", pca.explained_variance_ratio)

    # Plot projection (scratch)
    plot_2d_projection(X_pca, y, "PCA Projection (from Scratch)", iris.target_names)

    # Compare with scikit-learn
    sk_pca = SKPCA(n_components=2)
    X_pca_sk = sk_pca.fit_transform(X_std)
    print("Explained variance ratio (scikit-learn):", sk_pca.explained_variance_ratio_)

    plot_2d_projection(X_pca_sk, y, "PCA Projection (scikit-learn)", iris.target_names)

    # Explained variance bar plot
    plt.figure(figsize=(7,5))
    plt.bar(range(1, len(pca.explained_variance_ratio)+1), pca.explained_variance_ratio*100,
            color='steelblue', alpha=0.7)
    plt.ylabel("Explained Variance (%)")
    plt.xlabel("Principal Components")
    plt.title("Explained Variance by Components (PCA from Scratch)")
    plt.grid(alpha=0.3)
    plt.show()

    print("\nSummary:")
    print("- PCA identifies directions of maximum variance.")
    print("- Eigenvectors define new axes (principal components).")
    print("- Works well for feature compression and visualization.")


if __name__ == "__main__":
    main()

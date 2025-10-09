#!/usr/bin/env python3
"""
K-Means Clustering (NumPy Implementation)
-----------------------------------------
This script implements a numerically stable version of K-Means and K-Means++ initialization from scratch.
It also compares results with scikit-learn's KMeans for validation.

Usage:
    python kmeans_numpy_main.py
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans


def pairwise_sq_dists(X, Y):
    """Compute squared Euclidean distances between all rows of X and Y."""
    return np.sum((X[:, None, :] - Y[None, :, :]) ** 2, axis=2)


def init_centroids_kmeans_plus_plus(X, k, rng):
    """Numerically stable K-Means++ initialization."""
    n = X.shape[0]
    centroids = [X[rng.integers(0, n)]]

    for _ in range(1, k):
        d2 = np.min(pairwise_sq_dists(X, np.array(centroids)), axis=1)
        d2 = np.maximum(d2, 0.0)
        total = np.sum(d2)

        if total == 0 or not np.isfinite(total):
            centroids.append(X[rng.integers(0, n)])
            continue

        probs = d2 / total
        probs = np.maximum(probs, 0)
        probs /= np.sum(probs)

        next_idx = rng.choice(n, p=probs)
        centroids.append(X[next_idx])

    return np.array(centroids)


def kmeans_numpy(X, k=3, init="k-means++", max_iter=300, tol=1e-4, rng=None):
    """K-Means clustering implemented from scratch with stable initialization."""
    rng = np.random.default_rng(rng)
    n, d = X.shape

    if init == "random":
        centroids = X[rng.choice(n, k, replace=False)]
    elif init == "k-means++":
        centroids = init_centroids_kmeans_plus_plus(X, k, rng)
    else:
        raise ValueError("init must be 'random' or 'k-means++'")

    prev_inertia = None
    for i in range(max_iter):
        dists = pairwise_sq_dists(X, centroids)
        labels = np.argmin(dists, axis=1)

        new_centroids = np.zeros_like(centroids)
        for j in range(k):
            cluster_points = X[labels == j]
            if len(cluster_points) > 0:
                new_centroids[j] = np.mean(cluster_points, axis=0)
            else:
                new_centroids[j] = X[rng.integers(0, n)]

        inertia = np.sum((X - centroids[labels]) ** 2)

        if prev_inertia is not None and abs(prev_inertia - inertia) < tol * prev_inertia:
            return new_centroids, labels, inertia, i + 1

        centroids = new_centroids
        prev_inertia = inertia

    return centroids, labels, inertia, max_iter


def main():
    print("\n=== NumPy K-Means Implementation ===\n")
    X, _ = make_blobs(n_samples=600, centers=4, cluster_std=1.10, random_state=42)

    C_np, labels_np, inertia_np, n_iter_np = kmeans_numpy(X, k=4, init="k-means++", max_iter=200, tol=1e-4, rng=42)
    print(f"NumPy K-Means -> inertia: {inertia_np:.2f}, iterations: {n_iter_np}")

    plt.figure(figsize=(6, 5))
    plt.scatter(X[:, 0], X[:, 1], s=10, c=labels_np, cmap="viridis", alpha=0.7)
    plt.scatter(C_np[:, 0], C_np[:, 1], marker="X", s=200, c="red")
    plt.title("NumPy K-Means (Stable K-Means++)")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.tight_layout()
    plt.show()

    print("\n=== scikit-learn Comparison ===\n")
    km = KMeans(n_clusters=4, n_init=10, random_state=42, init="k-means++")
    km.fit(X)
    print(f"sklearn KMeans -> inertia: {km.inertia_:.2f}, iterations: {km.n_iter_}")

    plt.figure(figsize=(6, 5))
    plt.scatter(X[:, 0], X[:, 1], s=10, c=km.labels_, cmap="viridis", alpha=0.7)
    plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], marker="X", s=200, c="red")
    plt.title("scikit-learn KMeans — Clusters and Centroids")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

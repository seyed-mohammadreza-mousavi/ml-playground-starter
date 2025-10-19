"""
Linear t-SNE & UMAP Visualization (with PCA baseline)
=====================================================

This script visualizes high-dimensional data using a linear PCA preprocessing
followed by nonlinear t-SNE and UMAP embeddings.

Mathematical background (for reference):

PCA:
-----
Given centered data X ∈ R^{n×d} and covariance S = (1/(n-1)) XᵀX,
solve S v_i = λ_i v_i, i=1,…,d.
Projection:  Z = X V_k ∈ R^{n×k}.

t-SNE:
-------
High-D affinities:
    p_{j|i} = exp(-||x_i - x_j||² / (2σ_i²)) / Σ_{k≠i} exp(-||x_i - x_k||² / (2σ_i²))
Perplexity:
    Perp(P_i) = 2^{H(P_i)},  H(P_i) = -Σ_j p_{j|i} log₂ p_{j|i}
Symmetric probabilities:
    p_{ij} = (p_{j|i} + p_{i|j}) / (2n)
Low-D affinities (Student-t kernel):
    q_{ij} ∝ (1 + ||y_i - y_j||²)^{-1}
Objective:
    KL(P‖Q) = Σ_{i≠j} p_{ij} log(p_{ij} / q_{ij})

UMAP:
------
High-D fuzzy graph weights:
    μ_{ij} = exp(-max(0, d(x_i, x_j) - ρ_i)/σ_i)
Low-D similarity:
    q_{ij} = 1 / (1 + a||y_i - y_j||^{2b})
Loss (cross-entropy):
    L = Σ_{(i,j)} [-μ_{ij} log q_{ij} - (1 - μ_{ij}) log(1 - q_{ij})]

Bayes reminder:
    P(y|x) = P(x|y) P(y) / P(x) ∝ P(x|y) P(y)
"""

import numpy as np
import matplotlib.pyplot as plt
from time import time
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

try:
    import umap
    HAS_UMAP = True
except Exception:
    HAS_UMAP = False
    print("UMAP not available. Install via: pip install umap-learn")

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

plt.rcParams["figure.figsize"] = (6, 5)
plt.rcParams["axes.grid"] = True


# --------------------------------------------------------------------------
# Data Loading & Preprocessing
# --------------------------------------------------------------------------

def load_iris():
    data = datasets.load_iris()
    return data.data, data.target, data.target_names


def load_digits():
    data = datasets.load_digits()
    X, y = data.data, data.target
    names = np.array([str(i) for i in np.unique(y)])
    return X, y, names


def pca_preprocess(X, n_pca=50):
    """Standardize and apply PCA for dimensionality reduction before t-SNE/UMAP."""
    Xs = StandardScaler().fit_transform(X)
    k = min(n_pca, Xs.shape[1])
    Z = PCA(n_components=k, random_state=RANDOM_STATE).fit_transform(Xs)
    return Xs, Z


# --------------------------------------------------------------------------
# Embedding & Plotting
# --------------------------------------------------------------------------

def plot_embedding(Y, y, title, target_names=None):
    """Scatter plot for embeddings with optional legend."""
    fig, ax = plt.subplots()
    sc = ax.scatter(Y[:, 0], Y[:, 1], c=y, s=16)
    ax.set_title(title)
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")

    if target_names is not None:
        uniq = np.unique(y)
        handles = [ax.scatter([], []) for _ in uniq]
        labels = [str(target_names[u]) if u < len(target_names) else str(u) for u in uniq]
        ax.legend(handles, labels, title="Classes", bbox_to_anchor=(1.04, 1), loc="upper left")

    plt.tight_layout()
    plt.show()


def run_tsne(Z, perplexity=30, n_iter=1000, init="pca", learning_rate="auto"):
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        n_iter=n_iter,
        init=init,
        learning_rate=learning_rate,
        random_state=RANDOM_STATE,
        verbose=0,
    )
    t0 = time()
    Y = tsne.fit_transform(Z)
    t1 = time()
    return Y, t1 - t0


def run_umap(Z, n_neighbors=15, min_dist=0.1, metric="euclidean"):
    if not HAS_UMAP:
        raise ImportError("UMAP not installed. Please run: pip install umap-learn")
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=RANDOM_STATE,
        n_components=2,
    )
    t0 = time()
    Y = reducer.fit_transform(Z)
    t1 = time()
    return Y, t1 - t0


# --------------------------------------------------------------------------
# Main Workflow
# --------------------------------------------------------------------------

def main():
    print("=== Linear PCA → t-SNE & UMAP Visualization ===")

    # --- Load Data ---
    X_i, y_i, names_i = load_iris()
    X_d, y_d, names_d = load_digits()

    print("Iris:", X_i.shape, "| Digits:", X_d.shape)

    # --- PCA preprocessing ---
    Xi_s, Zi = pca_preprocess(X_i, n_pca=4)
    Xd_s, Zd = pca_preprocess(X_d, n_pca=50)

    # --- PCA 2D baseline ---
    pca2_i = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(StandardScaler().fit_transform(X_i))
    plot_embedding(pca2_i, y_i, "Iris — PCA (2D linear)", names_i)

    pca2_d = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(StandardScaler().fit_transform(X_d))
    plot_embedding(pca2_d, y_d, "Digits — PCA (2D linear)", names_d)

    # --- t-SNE ---
    print("Running t-SNE...")
    Yti, dti = run_tsne(Zi, perplexity=30)
    plot_embedding(Yti, y_i, f"Iris — t-SNE on PCA ({dti:.2f}s)", names_i)

    Ytd, dtd = run_tsne(Zd, perplexity=30)
    plot_embedding(Ytd, y_d, f"Digits — t-SNE on PCA ({dtd:.2f}s)", names_d)

    # --- UMAP ---
    if HAS_UMAP:
        print("Running UMAP...")
        Yui, dui = run_umap(Zi, n_neighbors=15, min_dist=0.1)
        plot_embedding(Yui, y_i, f"Iris — UMAP on PCA ({dui:.2f}s)", names_i)

        Yud, dud = run_umap(Zd, n_neighbors=15, min_dist=0.1)
        plot_embedding(Yud, y_d, f"Digits — UMAP on PCA ({dud:.2f}s)", names_d)
    else:
        print("Skipping UMAP (module not available).")

    print("\nDone.")


# --------------------------------------------------------------------------
# Entrypoint
# --------------------------------------------------------------------------
if __name__ == "__main__":
    main()
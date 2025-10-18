import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons, make_circles, make_blobs
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
import itertools

# =========================
# Utility functions
# =========================

def k_distance_plot(X, k, title="k-distance plot"):
    nbrs = NearestNeighbors(n_neighbors=k, metric="euclidean").fit(X)
    distances, _ = nbrs.kneighbors(X)
    kth = np.sort(distances[:, -1])
    plt.figure()
    plt.plot(kth)
    plt.xlabel("Points (sorted)")
    plt.ylabel(f"Distance to {k}-th neighbor")
    plt.title(title)
    plt.show()

def safe_silhouette(X, labels):
    mask = labels != -1
    if mask.sum() <= 2:
        return None
    unique = np.unique(labels[mask])
    if unique.size < 2:
        return None
    try:
        return float(silhouette_score(X[mask], labels[mask]))
    except Exception:
        return None

# =========================
# Naive from-scratch DBSCAN
# =========================

def dbscan_naive(X, eps=0.3, min_samples=5):
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    labels = -np.ones(n, dtype=int)
    diffs = X[:, None, :] - X[None, :, :]
    dist2 = np.sum(diffs * diffs, axis=2)
    eps2 = eps * eps
    neighbors = [np.where(dist2[i] <= eps2)[0] for i in range(n)]
    visited = np.zeros(n, dtype=bool)
    cluster_id = 0

    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True
        Ni = neighbors[i]
        if Ni.size < min_samples:
            labels[i] = -1
            continue
        labels[i] = cluster_id
        seeds = list(Ni.tolist())
        k = 0
        while k < len(seeds):
            p = seeds[k]
            if not visited[p]:
                visited[p] = True
                Np = neighbors[p]
                if Np.size >= min_samples:
                    for q in Np:
                        if q not in seeds:
                            seeds.append(int(q))
            if labels[p] < 0:
                labels[p] = cluster_id
            k += 1
        cluster_id += 1
    return labels

# =========================
# Visualization helpers
# =========================

def run_and_plot_dbscan(X, eps=0.3, min_samples=5, title="DBSCAN result"):
    Xs = StandardScaler().fit_transform(X)
    db = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean').fit(Xs)
    labels = db.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    sil = safe_silhouette(Xs, labels)

    plt.figure()
    for lab in sorted(set(labels)):
        mask = labels == lab
        plt.scatter(Xs[mask, 0], Xs[mask, 1], s=10, label=("Noise" if lab == -1 else f"Cluster {lab}"))
    subtitle = f"clusters={n_clusters}"
    if sil is not None:
        subtitle += f", silhouette={sil:.3f}"
    plt.title(f"{title} ({subtitle})")
    plt.legend()
    plt.show()
    return labels

# =========================
# Parameter grid search
# =========================

def grid_search_dbscan(X, eps_list, min_samples_list):
    Xs = StandardScaler().fit_transform(X)
    out = []
    for eps, ms in itertools.product(eps_list, min_samples_list):
        db = DBSCAN(eps=eps, min_samples=ms).fit(Xs)
        labels = db.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        sil = safe_silhouette(Xs, labels)
        out.append((eps, ms, n_clusters, (sil if sil is not None else float('nan'))))
    return out

# =========================
# Main demonstration
# =========================

def main():
    print("=== DBSCAN Clustering Demo ===")

    # 1. Demo with from-scratch DBSCAN
    X_small, _ = make_moons(n_samples=400, noise=0.06, random_state=42)
    labels_small = dbscan_naive(X_small, eps=0.25, min_samples=5)
    plt.figure()
    for lab in sorted(set(labels_small)):
        mask = labels_small == lab
        plt.scatter(X_small[mask, 0], X_small[mask, 1], s=12, label=f"Cluster {lab}")
    plt.title("From-scratch DBSCAN (naive) on two moons")
    plt.legend()
    plt.show()

    # 2. sklearn DBSCAN demos
    X_moons, _ = make_moons(n_samples=1000, noise=0.07, random_state=0)
    k_distance_plot(StandardScaler().fit_transform(X_moons), k=4, title="k-distance (k=4) for two moons")
    run_and_plot_dbscan(X_moons, eps=0.25, min_samples=5, title="DBSCAN on Two Moons")

    X_circ, _ = make_circles(n_samples=1000, factor=0.5, noise=0.06, random_state=1)
    k_distance_plot(StandardScaler().fit_transform(X_circ), k=4, title="k-distance (k=4) for circles")
    run_and_plot_dbscan(X_circ, eps=0.25, min_samples=5, title="DBSCAN on Circles")

    X_blobs, _ = make_blobs(n_samples=900, centers=4, cluster_std=[1.0, 2.2, 0.7, 1.5], random_state=2)
    X_blobs = np.vstack([X_blobs, np.random.uniform(low=-10, high=10, size=(50, 2))])
    k_distance_plot(StandardScaler().fit_transform(X_blobs), k=5, title="k-distance (k=5) for blobs+noise")
    run_and_plot_dbscan(X_blobs, eps=0.25, min_samples=6, title="DBSCAN on Blobs + Noise")

    # 3. Parameter grid search
    eps_list = [0.15, 0.2, 0.25, 0.3, 0.35]
    min_samples_list = [4, 5, 8, 12]
    grid_results = grid_search_dbscan(X_moons, eps_list, min_samples_list)
    print("\neps\tmin_samples\tclusters\tsilhouette(ignoring noise)")
    for eps, ms, nc, sil in grid_results:
        if np.isnan(sil):
            print(f"{eps:.2f}\t{ms:>3}\t\t{nc:>3}\t\tNaN")
        else:
            print(f"{eps:.2f}\t{ms:>3}\t\t{nc:>3}\t\t{sil:.3f}")

    # 4. Cosine metric example
    Xv, _ = make_blobs(n_samples=600, centers=3, cluster_std=[1.0, 2.5, 1.4], n_features=5, random_state=7)
    Xv_norm = normalize(Xv)
    db = DBSCAN(eps=0.1, min_samples=6, metric='cosine').fit(Xv_norm)
    labels = db.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print("\nCosine metric example — clusters found:", n_clusters)

if __name__ == "__main__":
    main()

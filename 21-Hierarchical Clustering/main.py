import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, dendrogram, cophenet
from scipy.spatial.distance import pdist


# ==============================================================
# 1. From-scratch implementation for educational purposes
# ==============================================================
def hac_average_linkage(X):
    """
    Hierarchical Agglomerative Clustering (Average Linkage)
    Returns: Z (n-1, 4) linkage-like array:
      [idx_A, idx_B, merge_distance, size_of_new_cluster]
    """
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    D = pdist(X, metric="euclidean")

    def key(i, j):
        return (i, j) if i < j else (j, i)

    dist = {}
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            dist[(i, j)] = D[idx]
            idx += 1

    active = {i: 1 for i in range(n)}
    Z = []
    next_label = n

    while len(active) > 1:
        labels = sorted(active.keys())
        best = None
        pair = None
        for a_i in range(len(labels)):
            for b_i in range(a_i + 1, len(labels)):
                a, b = labels[a_i], labels[b_i]
                d_ab = dist.get(key(a, b))
                if best is None or d_ab < best:
                    best = d_ab
                    pair = (a, b)

        a, b = pair
        size_a, size_b = active[a], active[b]
        new_label = next_label
        next_label += 1

        Z.append([a, b, float(best), size_a + size_b])

        # Update distances
        for k in list(active.keys()):
            if k in (a, b):
                continue
            d_ak = dist.get(key(a, k))
            d_bk = dist.get(key(b, k))
            new_d = (size_a * d_ak + size_b * d_bk) / (size_a + size_b)
            dist[key(new_label, k)] = new_d

        # Clean old keys
        for k in list(active.keys()):
            if k not in (a, b):
                dist.pop(key(a, k), None)
                dist.pop(key(b, k), None)
        dist.pop(key(a, b), None)

        # Update active
        del active[a]
        del active[b]
        active[new_label] = size_a + size_b

    return np.array(Z, dtype=float)


# ==============================================================
# 2. Silhouette sweep function
# ==============================================================
def sweep_silhouette(X, link, ks):
    scores = []
    for k in ks:
        model = AgglomerativeClustering(n_clusters=k, linkage=link)
        labels = model.fit_predict(X)
        if k > 1:
            score = silhouette_score(X, labels, metric="euclidean")
        else:
            score = np.nan
        scores.append(score)
    return np.array(scores, dtype=float)


# ==============================================================
# 3. Main pipeline
# ==============================================================
def main():
    print("=== Hierarchical Clustering Pipeline ===")

    # --------------- Data Generation ----------------
    X_blobs, y_blobs = make_blobs(
        n_samples=600,
        centers=4,
        cluster_std=[1.0, 1.2, 0.9, 1.1],
        random_state=42,
    )
    X_moons, y_moons = make_moons(n_samples=600, noise=0.08, random_state=42)

    Xb = StandardScaler().fit_transform(X_blobs)
    Xm = StandardScaler().fit_transform(X_moons)

    print("Data prepared:", Xb.shape, Xm.shape)

    # --------------- Dendrograms ----------------
    print("\nGenerating dendrograms...")
    Z_blobs_ward = linkage(Xb, method="ward")
    Z_moons_avg = linkage(Xm, method="average", metric="euclidean")

    plt.figure(figsize=(10, 5))
    dendrogram(Z_blobs_ward, truncate_mode="level", p=5)
    plt.title("Dendrogram (Blobs, Ward)")
    plt.show()

    plt.figure(figsize=(10, 5))
    dendrogram(Z_moons_avg, truncate_mode="level", p=5)
    plt.title("Dendrogram (Moons, Average)")
    plt.show()

    # --------------- Cophenetic correlation ----------------
    c_blobs, _ = cophenet(Z_blobs_ward, pdist(Xb))
    c_moons, _ = cophenet(Z_moons_avg, pdist(Xm))
    print(f"Cophenetic corr (Blobs): {c_blobs:.4f}")
    print(f"Cophenetic corr (Moons): {c_moons:.4f}")

    # --------------- Agglomerative clustering ----------------
    k_blobs, k_moons = 4, 2
    agg_blobs = AgglomerativeClustering(n_clusters=k_blobs, linkage="ward")
    labels_blobs = agg_blobs.fit_predict(Xb)

    agg_moons = AgglomerativeClustering(n_clusters=k_moons, linkage="average")
    labels_moons = agg_moons.fit_predict(Xm)

    plt.figure(figsize=(6, 5))
    plt.scatter(Xb[:, 0], Xb[:, 1], c=labels_blobs)
    plt.title("Agglomerative (Ward) on Blobs, k=4")
    plt.show()

    plt.figure(figsize=(6, 5))
    plt.scatter(Xm[:, 0], Xm[:, 1], c=labels_moons)
    plt.title("Agglomerative (Average) on Moons, k=2")
    plt.show()

    # --------------- Silhouette analysis ----------------
    print("\nRunning silhouette analysis...")
    ks_blobs = list(range(2, 8))
    ks_moons = list(range(2, 8))
    scores_blobs = sweep_silhouette(Xb, "ward", ks_blobs)
    scores_moons = sweep_silhouette(Xm, "average", ks_moons)

    print("Silhouette (blobs):", np.round(scores_blobs, 3))
    print("Silhouette (moons):", np.round(scores_moons, 3))

    plt.figure(figsize=(6, 4))
    plt.plot(ks_blobs, scores_blobs, marker="o")
    plt.title("Silhouette vs k (Blobs, Ward)")
    plt.xlabel("k")
    plt.ylabel("Silhouette")
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.plot(ks_moons, scores_moons, marker="o")
    plt.title("Silhouette vs k (Moons, Average)")
    plt.xlabel("k")
    plt.ylabel("Silhouette")
    plt.show()

    # --------------- From-scratch average linkage demo ----------------
    print("\nRunning from-scratch HAC (subset of Moons)...")
    Xm_small = Xm[:50]
    Z_small = hac_average_linkage(Xm_small)

    plt.figure(figsize=(10, 5))
    dendrogram(Z_small)
    plt.title("From-scratch Average-Linkage Dendrogram (subset of Moons)")
    plt.show()

    # --------------- Summary ----------------
    print("\nPipeline complete ✅")
    print(f"Cophenetic correlations: blobs={c_blobs:.3f}, moons={c_moons:.3f}")


# ==============================================================
# Entry point
# ==============================================================
if __name__ == "__main__":
    main()


# Hierarchical Clustering — Theory, Implementation, and Visualization

This repository demonstrates **Hierarchical Clustering (HC)** both theoretically and practically — including the mathematical background, scikit-learn and SciPy implementations, visualization via dendrograms, and a simple from-scratch average-linkage implementation.

---

## 📘 Overview

Hierarchical Clustering is a **bottom-up (agglomerative)** unsupervised learning algorithm.  
It successively merges the most similar clusters until only one remains, building a **dendrogram** (tree structure).  
The dendrogram can be cut at any height to yield a desired number of clusters.

---

## 🧠 Mathematical Background

We consider a dataset X = {x₁, x₂, …, xₙ} where each xᵢ is a vector in Rᵈ (a d-dimensional real space) and we define a pairwise distance function d(xᵢ, xⱼ).


### Common Distance Metrics

- **Euclidean Distance**
  $$ d_2(\\mathbf{a}, \\mathbf{b}) = \\sqrt{\\sum_{k=1}^{d} (a_k - b_k)^2} $$

- **Manhattan Distance**
  $$ d_1(\\mathbf{a}, \\mathbf{b}) = \\sum_{k=1}^{d} |a_k - b_k| $$

- **Cosine Distance**
  $$ d_{\\text{cos}}(\\mathbf{a}, \\mathbf{b}) = 1 - \\frac{\\mathbf{a}^T \\mathbf{b}}{\\|\\mathbf{a}\\| \\; \\|\\mathbf{b}\\|} $$

---

## 🔗 Linkage Criteria

Let $A$ and $B$ be two clusters. Define their inter-cluster dissimilarity $d(A,B)$ as one of the following:

- **Single Linkage**  
  $$ d_{\\text{single}}(A,B) = \\min_{\\mathbf{x}\\in A,\\,\\mathbf{y}\\in B} d(\\mathbf{x}, \\mathbf{y}) $$

- **Complete Linkage**  
  $$ d_{\\text{complete}}(A,B) = \\max_{\\mathbf{x}\\in A,\\,\\mathbf{y}\\in B} d(\\mathbf{x}, \\mathbf{y}) $$

- **Average Linkage (UPGMA)**  
  $$ d_{\\text{average}}(A,B) = \\frac{1}{|A|\\,|B|} \\sum_{\\mathbf{x}\\in A} \\sum_{\\mathbf{y}\\in B} d(\\mathbf{x},\\mathbf{y}) $$

- **Ward’s Method (Variance-Minimizing)**  
  $$ \\Delta(A,B) = \\frac{|A|\\,|B|}{|A| + |B|}\\,\\|\\boldsymbol{\\mu}_A - \\boldsymbol{\\mu}_B\\|_2^2 $$

---

## ⚙️ Lance–Williams Recurrence

When merging $A$ and $B$ into $C = A \\cup B$, the distances to another cluster $K$ can be updated with:

$$
\\begin{aligned}
d(C, K) &= \\alpha_A d(A,K) + \\alpha_B d(B,K) + \\beta d(A,B) + \\gamma |d(A,K) - d(B,K)|
\\end{aligned}
$$

For average linkage:  
$\\alpha_A = \\frac{|A|}{|A| + |B|}$, $\\alpha_B = \\frac{|B|}{|A| + |B|}$, $\\beta = 0$, $\\gamma = 0$.

---

## 🌳 Dendrograms and Complexity

- **Output:** A tree (dendrogram) representing the merge history.  
- **Stopping rule:** Cut the dendrogram at a chosen height or pick $k$ clusters.  
- **Complexity:** $O(n^3)$ time and $O(n^2)$ memory (can be optimized to $O(n^2)$ with specialized methods).

---

## 📊 Implementation Highlights

### 1. SciPy and scikit-learn

We use:

```python
from scipy.cluster.hierarchy import linkage, dendrogram, cophenet
from sklearn.cluster import AgglomerativeClustering
```

To compute and visualize hierarchical merges via:

```python
Z = linkage(X, method="ward")
dendrogram(Z)
```

### 2. Silhouette Analysis

The **silhouette score** measures cluster cohesion and separation:

$$ s = \\frac{b - a}{\\max(a,b)} $$

where:
- $a$: mean intra-cluster distance  
- $b$: mean nearest-cluster distance  

Higher $s$ indicates better clustering quality.

---

## 🧩 From-Scratch Implementation

For educational purposes, an $O(n^3)$ average-linkage algorithm is included.

The Lance–Williams update rule used:

$$
d(A\\cup B, K) = \\frac{|A|}{|A|+|B|} d(A,K) + \\frac{|B|}{|A|+|B|} d(B,K)
$$

---

## 🧮 Usage

### Run the main program

```bash
python main.py
```

### What it does

1. Generates synthetic datasets (**blobs** and **two moons**).  
2. Standardizes features.  
3. Computes and visualizes **dendrograms**.  
4. Calculates **cophenetic correlations**.  
5. Performs **agglomerative clustering** (Ward & Average).  
6. Runs **silhouette analysis**.  
7. Demonstrates a **from-scratch HAC** (for small subsets).

---

## 📈 Visualizations

- Dendrogram (Ward linkage)
- Dendrogram (Average linkage)
- Scatter plots of final cluster assignments
- Silhouette vs number of clusters

---

## 💡 Practical Tips

- Always **scale features** before applying Ward’s method.
- **Single/average linkage** capture non-convex shapes but are noise-sensitive.
- For large datasets, **truncate dendrograms** or sample subsets.
- Use **cosine distance** for high-dimensional data.
- Evaluate structure with **cophenetic correlation** and **silhouette**.

---

## 🧪 Environment

```
Python >= 3.10
NumPy >= 1.23
SciPy >= 1.9
scikit-learn >= 1.3
matplotlib >= 3.7
```

---

## 📚 References

- Rokach, Lior. *Data Clustering: Methods and Algorithms*, Springer, 2005.  
- Murtagh & Contreras (2012). *Algorithms for hierarchical clustering: an overview.* Wiley Interdisciplinary Reviews: Data Mining and Knowledge Discovery.

---

## 🧩 License

This project is released under the MIT License.

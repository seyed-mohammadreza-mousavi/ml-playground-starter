# DBSCAN Clustering — Theory, Implementation, and Visualization

This repository contains a **complete educational implementation** and demonstration of the **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)** algorithm.

It includes both a **Jupyter Notebook** (`DBSCAN_Clustering.ipynb`) and a **Python script** (`DBSCAN_main.py`) version for experimentation, visualization, and parameter exploration.

---

## 📘 Overview

DBSCAN is a **density-based clustering** algorithm that groups together closely packed points and labels low-density points as **outliers**. Unlike k-means, it does **not require predefining the number of clusters**, and it can discover **arbitrarily shaped clusters**.

This project includes:

- Full conceptual overview of DBSCAN (intuitive explanation).
- A **from-scratch educational implementation** using pure NumPy (O(n²)).
- The **scikit-learn optimized version** with visual demos.
- Tools for **parameter selection** (`eps` and `min_samples`).
- Grid search for sensitivity analysis.
- Visualizations on multiple synthetic datasets (moons, circles, blobs).
- Example of using **different distance metrics** (Euclidean, cosine).

---

## 🧩 Files

| File | Description |
|------|--------------|
| `DBSCAN_Clustering.ipynb` | Interactive notebook version with equations, code, and inline plots. |
| `DBSCAN_main.py` | Script version with a `main()` function — fully runnable from terminal. |
| `README_DBSCAN.md` | This documentation file. |

---

## 🧠 Features

### 1. Educational Naïve Implementation
A fully commented O(n²) implementation of DBSCAN from scratch using pairwise distances.  
Best for understanding algorithmic steps — not for production-scale data.

### 2. Scikit-Learn Implementation
Demonstrates usage of `sklearn.cluster.DBSCAN` with real-world parameter tuning and visualization.

### 3. Parameter Selection Tools
Includes utilities to compute **k-distance plots** and perform **grid searches** over parameter values to understand sensitivity to `eps` and `min_samples`.

### 4. Multiple Synthetic Datasets
Visual experiments on:
- Two-moons dataset
- Concentric circles
- Blobs with noise
- Custom 5D blobs (cosine metric)

---

## ⚙️ Usage

### Run the Notebook
1. Open `DBSCAN_Clustering.ipynb` in Jupyter or Google Colab.
2. Execute each cell sequentially to explore the implementation interactively.

### Run the Python Script
```bash
python DBSCAN_main.py
```
The script automatically runs all demonstrations and plots results interactively.

---

## 📈 Example Output

The notebook and script generate the following visualizations:

- DBSCAN clustering results on various datasets.
- k-distance elbow plots for parameter tuning.
- Scatter plots with noise and cluster labeling.
- Printed grid search table for silhouette scores (ignoring noise).

---

## 🧮 Dependencies

Install all required dependencies using:

```bash
pip install numpy matplotlib scikit-learn
```

All examples run on standard Python 3.8+ environments.

---

## 💡 Notes & Tips

- **Scaling matters** — always normalize or standardize input features before clustering.
- Use **k-distance plots** to choose `eps` intelligently.
- **Increasing `min_samples`** creates stricter density requirements → fewer clusters and more noise.
- For **datasets with varying density**, consider **HDBSCAN** (a hierarchical extension).

---

## 📚 References

- Ester, Kriegel, Sander, Xu (1996). *A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise.* KDD.
- scikit-learn Documentation — [DBSCAN](https://scikit-learn.org/stable/modules/clustering.html#dbscan).
- McInnes, Healy, Astels (2017). *hdbscan: Hierarchical density-based clustering.*

---

## 🧑‍💻 Author

Developed as part of an educational and research demonstration for clustering algorithms.  
Includes clear structure for integration into teaching materials or ML repositories.

---

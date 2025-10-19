# Linear t-SNE & UMAP Visualization (with PCA Baseline)

This repository demonstrates a **complete pipeline for visualizing high-dimensional data** using a **linear PCA preprocessing step** followed by **nonlinear t-SNE** and **UMAP** embeddings.

The workflow includes:
- **Mathematical intuition** behind PCA, t-SNE, and UMAP  
- **Implementation in Python** using `scikit-learn` and `umap-learn`  
- **Visualizations** for the **Iris** and **Digits** datasets  
- A **fully functional script** (`main.py`) and **notebook** (`linear_tsne_umap.ipynb`)  

---

## 🔍 Overview

Dimensionality reduction is essential when exploring or visualizing data with many features.  
This project illustrates how to:
1. Apply **Principal Component Analysis (PCA)** for initial linear reduction.  
2. Use **t-Distributed Stochastic Neighbor Embedding (t-SNE)** for nonlinear manifold visualization.  
3. Use **Uniform Manifold Approximation and Projection (UMAP)** as an efficient alternative with better global structure preservation.  

Each step is clearly implemented and benchmarked for runtime and visual quality.

---

## 📂 Project Structure

```
├── main.py                    # Full implementation script with main() entrypoint
├── linear_tsne_umap.ipynb     # Jupyter notebook version with LaTeX math and commentary
├── README.md                  # This file
```

---

## ⚙️ Installation

Make sure you have Python 3.8+ installed.  
Clone the repository and install dependencies:

```bash
git clone https://github.com/yourusername/linear-tsne-umap.git
cd linear-tsne-umap
pip install -r requirements.txt
```

If you do not have `umap-learn` installed yet, simply run:

```bash
pip install umap-learn
```

---

## 🚀 Usage

Run the visualization directly from the terminal:

```bash
python main.py
```

Or open the notebook version:

```bash
jupyter notebook linear_tsne_umap.ipynb
```

This will:
- Load the **Iris** and **Digits** datasets.
- Apply **PCA (2D)** as a linear baseline.
- Apply **PCA-50 → t-SNE** and **PCA-50 → UMAP** for nonlinear visualization.
- Display color-coded 2D scatter plots for each embedding.

---

## 📊 Output Examples

The script produces clear visual comparisons:
- **PCA (Linear 2D)**: fast, interpretable, but may not separate classes well.
- **t-SNE on PCA-50**: emphasizes local cluster structure.
- **UMAP on PCA-50**: balances local and global structure efficiently.

Each plot is generated interactively with legends and color-coded class labels.

---

## 💡 Practical Notes

- Always **standardize features** before applying PCA or UMAP/t-SNE.
- Start with **PCA → 50D** preprocessing for speed and denoising.
- Tune **t-SNE perplexity (5–50)** and **UMAP n_neighbors (5–50)** for best results.
- Use a **fixed random seed** for reproducibility.
- t-SNE often produces tighter clusters, while UMAP preserves global relationships better.

---

## 🧠 Dependencies

| Library | Purpose |
|----------|----------|
| `numpy` | Numerical computation |
| `matplotlib` | Visualization |
| `scikit-learn` | PCA, t-SNE, and datasets |
| `umap-learn` | UMAP embedding (optional but recommended) |

Install all dependencies via:

```bash
pip install numpy matplotlib scikit-learn umap-learn
```

---

## 📘 References

- van der Maaten, L., & Hinton, G. (2008). *Visualizing Data using t-SNE.*  
- McInnes, L., Healy, J., & Melville, J. (2018). *UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction.*  
- Scikit-learn documentation: [https://scikit-learn.org/](https://scikit-learn.org/)  
- UMAP documentation: [https://umap-learn.readthedocs.io/](https://umap-learn.readthedocs.io/)

---

## 🧩 License

This project is released under the **MIT License**.  
Feel free to use, modify, and distribute it with attribution.

---
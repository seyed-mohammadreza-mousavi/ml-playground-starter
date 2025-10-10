
# 📉 Principal Component Analysis (PCA) & Dimensionality Reduction

This project demonstrates **Principal Component Analysis (PCA)** — a fundamental technique in machine learning for **dimensionality reduction**, **feature extraction**, and **data visualization**.

---

## 📘 Overview

**Principal Component Analysis (PCA)** is a statistical method that transforms a dataset into a set of linearly uncorrelated variables called **principal components**.  
These components capture the directions of **maximum variance** in the data.

- **Type:** Unsupervised Learning  
- **Goal:** Reduce dimensions while retaining maximum information  
- **Use Cases:** Visualization, noise reduction, compression, and preprocessing

---

## 🧮 Mathematical Foundation

Given a dataset $X \in \mathbb{R}^{n \times d}$ with $n$ samples and $d$ features:

### Steps:

1. **Standardize** the data (zero mean, unit variance):  
   $X_{std} = X - \mu$

2. **Compute the covariance matrix:**  
   $\Sigma = \frac{1}{n-1} X_{std}^T X_{std}$

3. **Eigen decomposition** of the covariance matrix:  
   $\Sigma v = \lambda v$  
   - $v$: eigenvectors (principal directions)  
   - $\lambda$: eigenvalues (explained variance)

4. **Sort eigenvectors** by descending eigenvalues.

5. **Project** data onto the top-$k$ components:  
   $X_{proj} = X_{std} W_k$

Where $W_k$ is the matrix of the top-k eigenvectors.

---

## ⚙️ Files Included

```
PCA/
│
├── PCA_Dimensionality_Reduction.ipynb   # Full notebook with math, code, and visuals
├── main.py                          # Python script (clean runnable version)
└── README.md                            # Project documentation
```

---

## 🚀 How to Run

### 1️⃣ Install dependencies
```bash
pip install numpy matplotlib scikit-learn
```

### 2️⃣ Run the script
```bash
python main_pca.py
```

### 3️⃣ Or open the notebook
```bash
jupyter notebook PCA_Dimensionality_Reduction.ipynb
```

---

## 🎨 Visualization

The project produces:

- **2D scatter plots** of data projected onto the first two principal components  
- **Explained variance bar chart** showing the contribution of each component  
- **Comparison** with scikit-learn’s PCA implementation  

---

## 📊 Example Output

```
Explained variance ratio (scratch): [0.9246 0.0530]
Explained variance ratio (scikit-learn): [0.9246 0.0530]
```

✅ The first component explains ~92% of the variance in the Iris dataset.

---

## 🧩 Key Learnings

- PCA identifies orthogonal directions (principal components) capturing maximum variance.  
- Eigenvectors define new coordinate axes.  
- Reduces noise and redundancy in features.  
- Enables 2D visualization of high-dimensional data.  

---

## 📜 License

This project is licensed under the **MIT License**.

---

**Next in the ML Playground →** *t-SNE & UMAP Visualization*

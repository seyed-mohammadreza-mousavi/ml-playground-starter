
# 🧠 K-Nearest Neighbors (KNN) from Scratch

This project demonstrates a **complete implementation of the K-Nearest Neighbors (KNN)** algorithm from scratch using NumPy, along with visualization and comparison to scikit-learn’s implementation.

---

## 📘 Overview
**K-Nearest Neighbors (KNN)** is a **non-parametric**, **instance-based** learning algorithm that classifies a new data point based on the majority label of its K closest training samples.

- **Type:** Supervised Learning  
- **Used for:** Classification and Regression  
- **Core idea:** “Show me your neighbors, and I’ll tell you who you are.”  

---

## 🧮 Mathematical Foundation

For two data points \( \mathbf{x}_i, \mathbf{x}_j \in \mathbb{R}^d \), their Euclidean distance is:

\[
d(\mathbf{x}_i, \mathbf{x}_j)
= \lVert \mathbf{x}_i - \mathbf{x}_j \rVert_2
= \sqrt{\sum_{k=1}^{d} \left( x_{i,k} - x_{j,k} \right)^2 } \, .
\]



### Algorithm Steps
1. Choose the number of neighbors **K**.  
2. Compute the **Euclidean distance** between the query point and all training samples.  
3. Select the **K nearest samples**.  
4. Assign the label that appears most frequently among the neighbors (majority vote).  

---

## ⚙️ Files Included

```
KNN/
│
├── KNN_from_Scratch.ipynb     # Full notebook with explanations and math
├── main_knn.py                # Python script (clean and executable)
└── README.md                  # You’re reading this file
```

---

## 🚀 How to Run

### 1️⃣ Install dependencies
```bash
pip install numpy matplotlib scikit-learn
```

### 2️⃣ Run the script
```bash
python main_knn.py
```

### 3️⃣ Or open the notebook
```bash
jupyter notebook KNN_from_Scratch.ipynb
```

---

## 🎨 Visualization

The project generates **decision boundary plots** for:
- KNN implemented from scratch  
- KNN using scikit-learn  

This helps you visualize how the algorithm partitions the feature space for different classes.

---

## 🧩 Key Learnings

- Understand the **math behind KNN**  
- Implement the algorithm **from scratch**  
- Visualize **decision boundaries**  
- Compare custom implementation with **scikit-learn’s KNN**  

---

## 📊 Example Output

```
Accuracy (Scratch KNN): 0.925
Accuracy (scikit-learn KNN): 0.925
```

Both implementations should achieve similar performance on the same dataset.

---

## 🧠 Concept Summary

| Concept | Description |
|----------|-------------|
| **Learning Type** | Supervised |
| **Model Type** | Instance-based, non-parametric |
| **Distance Metric** | Euclidean (default) |
| **Complexity** | High at prediction time |
| **Best For** | Low-dimensional, small datasets |

---

## 📜 License
This project is licensed under the **MIT License**.

---

**Next in the ML Playground:** [Naive Bayes Classifier →]

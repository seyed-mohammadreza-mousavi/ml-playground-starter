# 🌲 Random Forest Playground
A compact educational repository demonstrating **Random Forests** for both **classification** and **regression** — complete with math explanations, implementation, and visualization.

---

## 📘 Overview
This project is part of the **ML Playground Starter** series — simple, self-contained demos for core machine learning algorithms.

It includes:
- 📗 A **Jupyter Notebook** (`random_forest_full.ipynb`) explaining theory and math behind Random Forests  
- 🐍 A **Python script** (`random_forest_demo.py`) showing a clean, production-style implementation  
- 📊 Visualizations of decision boundaries, feature importance, and partial dependence plots  

---

## 🧠 What is a Random Forest?
A **Random Forest** is an ensemble of Decision Trees trained on random subsets of data and features.  
It combines predictions from many de-correlated trees to reduce variance and improve generalization.

**Mathematically:**
\[
\hat{f}(x) = \frac{1}{B} \sum_{b=1}^{B} T_b(x)
\]

Each \(T_b(x)\) is a tree trained on a bootstrap sample with random feature selection.

---

## ⚙️ Features
- **Theory section** with impurity functions (Gini, entropy, variance)
- **Classification demo** with decision boundary visualization  
- **Regression demo** with RMSE & R² metrics  
- **Partial dependence plots** for feature effect analysis  
- **Hyperparameter tuning** example using Grid Search  
- **OOB score evaluation** (out-of-bag estimation)  

---

## 🚀 How to Run

### 1️⃣ Clone the repository
```bash
git clone https://github.com/seyed-mohammadreza-mousavi/ml-playground-starter.git
cd ml-playground-starter
```

### 2️⃣ Install dependencies
```bash
pip install -r requirements.txt
```
*(Or just install the essentials)*
```bash
pip install scikit-learn matplotlib seaborn numpy pandas
```

### 3️⃣ Run the notebook
Open in Jupyter or Google Colab:
```bash
jupyter notebook random_forest_full.ipynb
```

### 4️⃣ Run the Python script
```bash
python random_forest_demo.py
```

---

## 🧩 Repository Structure
```
ml-playground-starter/
│
├── random_forest_full.ipynb      # Educational Jupyter notebook with theory + math
├── random_forest_demo.py         # Clean Python implementation with main()
├── README.md                     # This documentation
└── requirements.txt              # (Optional) Dependencies file
```

---

## 📈 Example Output
- **RMSE / R² metrics**
- **Predicted vs True plot**
- **Partial dependence plots** showing feature effects

---

## 👤 Author
**Seyed Mohammadreza Mousavi**  
Machine Learning Engineer & Researcher  
📧 [error.2013@yahoo.com](mailto:error.2013@yahoo.com)  
🌐 [LinkedIn Profile](https://www.linkedin.com/in/seyed-mohammadreza-mousavi)  

---

## 🧾 License
MIT License © 2025 Seyed Mohammadreza Mousavi

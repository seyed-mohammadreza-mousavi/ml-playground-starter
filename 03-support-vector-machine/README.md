# Support Vector Machines (SVM) – Comprehensive Tutorial

This repository contains a **hands-on tutorial** for Support Vector Machines (SVM) implemented with **scikit-learn**. It includes both a **Jupyter Notebook** and a **standalone Python script (`main.py`)** that cover classification, regression, visualization, and advanced SVM topics.

---

## 📖 Contents

### Notebook (`SVM_Comprehensive_Tutorial.ipynb`)
- **Intuition & Math**
  - Hard margin vs. soft margin SVM
  - Regularization parameter \(C\)
  - Kernel trick (linear, polynomial, RBF)
- **Clean reusable code**
  - Pipelines with `StandardScaler`
  - `GridSearchCV` for model selection
- **Visualization**
  - Decision boundaries (2D synthetic datasets)
  - Confusion matrices
  - ROC and Precision-Recall curves
  - Learning curves
- **Datasets**
  - Synthetic (moons, circles, linear)
  - Real: Iris, Breast Cancer
- **Advanced Topics**
  - Class imbalance (`class_weight='balanced'`)
  - Probability calibration (`probability=True`)
  - Multiclass strategies (OvR vs. OvO)
- **Regression**
  - Support Vector Regression (SVR)
  - Metrics: RMSE, \(R^2\)
  - Predicted vs. actual plots

### Script (`main.py`)
The script reproduces all major experiments from the notebook in a **CLI-friendly** way:
- Saves figures into `./svm_outputs`
- Prints metrics to console
- Supports section selection and headless mode

---

## 🚀 Getting Started

### Installation
Clone the repository and install dependencies:

```bash
git clone https://github.com/your-username/svm-tutorial.git
cd svm-tutorial
pip install -r requirements.txt
```

**Requirements**
- Python ≥ 3.8  
- numpy  
- matplotlib  
- scikit-learn  

---

### Usage

#### 1. Run the Notebook
Open in Jupyter or Google Colab:

```bash
jupyter notebook SVM_Comprehensive_Tutorial.ipynb
```

#### 2. Run the Script
Execute all sections and save plots (without opening windows):

```bash
python main.py --no-show
```

Run specific sections:

```bash
python main.py --sections linear rbf svr --no-show
```

Change output directory:

```bash
python main.py --outdir results --no-show
```

Available sections:
- `linear` – Linear SVM (2D synthetic)
- `rbf` – RBF kernel SVM
- `poly` – Polynomial SVM
- `grid` – GridSearchCV on Breast Cancer
- `multi` – Multiclass (Iris dataset)
- `imbalance` – Class imbalance demo
- `calibration` – Probability calibration
- `learning` – Learning curves
- `scaling` – Why scaling matters
- `svr` – Support Vector Regression

---

## 📂 Project Structure

```
.
├── SVM_Comprehensive_Tutorial.ipynb   # Notebook with code + theory
├── main.py                            # Script version
├── requirements.txt                   # Dependencies
├── svm_outputs/                       # Auto-generated plots (after running main.py)
└── README.md                          # Project documentation
```

---

## ✨ Features

- Step-by-step math with LaTeX in the notebook  
- Clean scikit-learn pipelines  
- Easy comparison across kernels  
- Covers both **classification** and **regression**  
- Handles **imbalanced data** and **multiclass tasks**  
- Reproducible results with saved outputs  

---

## 📝 License

This project is released under the MIT License.  
Feel free to use, modify, and share!

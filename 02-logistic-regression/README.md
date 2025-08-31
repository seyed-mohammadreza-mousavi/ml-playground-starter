# Logistic Regression Tutorial

This module is part of the **ML Playground Starter** repository and provides a complete, hands-on tutorial on **Logistic Regression**.

It includes both an interactive Jupyter notebook and a Python CLI script, covering theory, implementation, evaluation, and advanced topics.

---

## 📂 Contents

- `logistic_regression_complete_tutorial.ipynb`  
  Step-by-step Jupyter notebook tutorial with math, from-scratch NumPy implementation, scikit-learn pipelines, visualizations, and exercises.

- `main.py`  
  Full Python script version of the tutorial. Can be run end-to-end or in parts via CLI flags.

- `outputs/`  
  Directory where generated plots (ROC, PR, decision boundaries, calibration, etc.) and trained models are saved.

---

## 🚀 Features

- **Math & Theory**
  - Sigmoid, log-loss, gradient derivation
- **From Scratch**
  - NumPy implementation with L2 regularization + early stopping
- **scikit-learn Pipelines**
  - Baseline logistic regression with preprocessing
- **Evaluation**
  - Accuracy, precision, recall, F1, ROC/PR curves, confusion matrix
- **Advanced Topics**
  - Hyperparameter tuning (GridSearchCV)
  - Regularization paths
  - Class imbalance handling (class weights)
  - Calibration curves
  - Polynomial features for non-linear decision boundaries
  - Multiclass classification (Iris dataset)
- **Deployment**
  - Save/load models with `joblib`
  - Outputs saved in `outputs/`

---

## ⚙️ Installation

Create a virtual environment and install dependencies:

```bash
pip install -r requirements.txt
```

If no `requirements.txt` yet, install manually:

```bash
pip install numpy pandas matplotlib scikit-learn joblib
```

---

## 📒 Notebook Usage

Launch Jupyter and open the tutorial notebook:

```bash
jupyter notebook 02-logistic-regression/logistic_regression_complete_tutorial.ipynb
```

---

## 🖥️ Script Usage

Run the full tutorial:

```bash
python 02-logistic-regression/main.py --all
```

Or run specific sections:

```bash
python 02-logistic-regression/main.py --from_scratch --sklearn
python 02-logistic-regression/main.py --tune --interpret --regpath
python 02-logistic-regression/main.py --poly --calibrate --imbalance --multiclass --save
```

All plots and models are saved in `outputs/`.

---

## 📊 Example Outputs

- Loss curve (from scratch)  
- ROC and PR curves  
- Decision boundaries (2D moons)  
- Regularization path plots  
- Calibration curves  
- Saved trained pipeline (`logreg_pipeline.joblib`)

---

## 📝 Exercises

Try extending the notebook or script by:
- Implementing L1 (Lasso) regularization from scratch
- Adding learning-rate schedules in gradient descent
- Exploring threshold tuning for F1 optimization
- Comparing logistic regression against other classifiers

---

## 📌 Notes

- This tutorial is for **educational purposes** and demonstrates best practices for building, training, and evaluating logistic regression models.  
- Contributions and improvements are welcome! 🚀

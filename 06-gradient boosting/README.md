# Gradient Boosting Models — XGBoost & LightGBM

This repository provides a **comprehensive demonstration** of Gradient Boosting algorithms using
[XGBoost](https://xgboost.readthedocs.io) and [LightGBM](https://lightgbm.readthedocs.io).
It includes complete examples for **classification**, **regression**, and **early stopping**.

---

## 🚀 Features

- **Classification Demo:**  
  Generates a 2D synthetic dataset, trains XGBoost and LightGBM classifiers, and visualizes decision boundaries.  

- **Regression Demo:**  
  Demonstrates nonlinear regression on synthetic data, comparing both models’ fits and reporting RMSE and R².  

- **Early Stopping Demo:**  
  Shows how both libraries automatically stop training when the validation loss no longer improves.  

- **Visualization:**  
  Includes decision boundaries, feature importances, and partial dependence plots for interpretability.  

---

## 📦 Requirements

Install the required libraries before running:

```bash
pip install numpy pandas matplotlib scikit-learn xgboost lightgbm
```

---

## ▶️ Usage

Run the full demonstration with:

```bash
python main.py
```

Each section will automatically execute:
- Classification comparison
- Regression fit comparison
- Early stopping training behavior

---

## 📊 Output

You will see:
- **Decision boundary plots** for classification  
- **Feature importance and partial dependence** visualizations  
- **Regression curve fits** comparing model predictions to noisy truth  
- **Early stopping results** showing best iteration counts

---

## 🧠 Author

Developed by **Seyed Mohammadreza Mousavi**  
*(AI/ML Researcher & Lecturer)*

---

## 📄 License

This project is released under the **MIT License**.

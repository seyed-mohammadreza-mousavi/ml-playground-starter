# Principal Component Regression (PCR)

This repository contains a complete implementation of **Principal Component Regression (PCR)** in Python, including data generation, model training, visualization, and performance evaluation.

## Overview

Principal Component Regression (PCR) is a two-step approach that combines **Principal Component Analysis (PCA)** with **Linear Regression** to address the issue of multicollinearity among predictor variables. The main idea is to first transform correlated features into a set of uncorrelated principal components and then perform regression on these components.

### Why PCR?
- Handles **multicollinearity** effectively.
- Reduces **overfitting** in high-dimensional datasets.
- Provides a **dimensionality reduction** step before regression.
- Works well when the number of features is large compared to the number of samples.

---

## Project Structure

```
├── main_pcr.py                     # Main Python script with full implementation
├── Principal_Component_Regression_PCR.ipynb  # Jupyter notebook (math + code + visualization)
├── README.md                       # Project documentation
```

---

## Features

✅ Generate synthetic regression data with multicollinearity  
✅ Perform PCA and select optimal number of components using cross-validation  
✅ Fit PCR, OLS, and Ridge regression models for comparison  
✅ Visualize explained variance ratios, cross-validation errors, predictions, and residuals  
✅ Recover and interpret coefficients in the original feature space  
✅ Compatible with Python 3.9+ and scikit-learn >= 1.0  

---

## How to Run

1. **Clone the repository:**

```bash
git clone https://github.com/yourusername/Principal-Component-Regression.git
cd Principal-Component-Regression
```

2. **Install dependencies:**

```bash
pip install -r requirements.txt
```

Or manually install:

```bash
pip install numpy pandas scikit-learn matplotlib
```

3. **Run the main script:**

```bash
python main_pcr.py
```

4. **Open the notebook (optional):**

```bash
jupyter notebook Principal_Component_Regression_PCR.ipynb
```

---

## Visualizations

- **Explained Variance Ratio per Component** – to understand the variance captured by each principal component.  
- **Cross-Validation Curve** – to select the optimal number of components $k$.  
- **Predicted vs True Scatter Plot** – to visualize regression accuracy.  
- **Residual Plot** – to detect nonlinearity or heteroscedasticity patterns.

---

## Example Output

```
Best number of components: k = 7

       Model         Test MSE     Test R^2
0        PCR         139.231      0.8821
1   OLS (std)        150.982      0.8733
2  Ridge (α=5)       141.442      0.8809

Top 10 coefficients (raw feature space):
   feature     beta_raw
0      x5     42.5135
1      x2     33.2714
2      x1     30.8249
3      x3     26.1141
...
```

---

## Notes

- PCA is **unsupervised**, meaning it doesn’t use the target variable $y$ when finding components. If you need supervised dimensionality reduction, consider **Partial Least Squares (PLS)**.
- Always **standardize** data before applying PCA.
- The number of components $k$ should ideally be chosen using **cross-validation**.

---

## License

This project is licensed under the MIT License — feel free to use, modify, and distribute.

---

# 🧠 Bagging & Ensemble Methods

This repository demonstrates **Bagging (Bootstrap Aggregating)** and related ensemble methods such as **Random Forests**, using both classification and regression examples in Python with scikit-learn.

It includes:
- Full mathematical background (bias–variance decomposition, correlation effects)
- Implementation using `BaggingClassifier`, `RandomForestClassifier`, `BaggingRegressor`, and `RandomForestRegressor`
- Decision-boundary and feature-importance visualizations
- Regression performance evaluation and OOB analysis

---

## 📚 1. Introduction

Bagging (Bootstrap Aggregating) reduces variance by training $B$ base learners on **bootstrap samples** of the original data and aggregating their predictions.

Given a dataset $$\mathcal{D} = \{ (\mathbf{x}_i, y_i) \}_{i=1}^{n}$$, each bootstrap sample $\mathcal{D}^{(b)}$ is formed by sampling $n$ points **with replacement**.

For regression, the bagged predictor is defined as:

$$
\hat{f}_{\text{bag}}(\mathbf{x}) = \frac{1}{B} \sum_{b=1}^{B} h^{(b)}(\mathbf{x})
$$

For binary classification (majority voting):

$$
\hat{y}_{\text{bag}}(\mathbf{x}) = \operatorname*{mode}\big(h^{(1)}(\mathbf{x}), h^{(2)}(\mathbf{x}), \ldots, h^{(B)}(\mathbf{x})\big)
$$

If each base learner outputs class probabilities $\hat{p}^{(b)}(y \mid \mathbf{x})$, we can average probabilities and take the $\arg\max_y$.

---

## 📊 2. Bias–Variance Decomposition

The expected mean squared error (MSE) decomposes as:

$$
\begin{aligned}
\mathbb{E}\big[(Y - \hat{f}(\mathbf{x}))^2\big]
&= \operatorname{Bias}[\hat{f}(\mathbf{x})]^2 + \operatorname{Var}[\hat{f}(\mathbf{x})] + \sigma^2
\end{aligned}
$$

If we average $B$ estimators $\hat{f}^{(b)}$ with variance $\tau^2$ and pairwise correlation $\rho$, then:

$$
\operatorname{Var}\left[ \frac{1}{B} \sum_{b=1}^{B} \hat{f}^{(b)} \right]
= \tau^2 \left( \rho + \frac{1 - \rho}{B} \right)
$$

Thus, as $B \to \infty$, the variance approaches $\rho \tau^2$. Bagging helps most when base learners are **high-variance** and **weakly correlated**.

---

## 🔢 3. Out-of-Bag (OOB) Estimation

Each bootstrap sample leaves out about 36.8% of the training data. These are called **out-of-bag (OOB)** samples. We can estimate test error without a separate validation set.

$$
\begin{aligned}
\hat{f}_{\text{OOB}}(\mathbf{x}_i)
&= \frac{1}{|\mathcal{B}_i|} \sum_{b \in \mathcal{B}_i} h^{(b)}(\mathbf{x}_i)
\end{aligned}
$$

where $\mathcal{B}_i$ is the set of models for which $(\mathbf{x}_i, y_i)$ was **not** included in the bootstrap sample.

---

## 🌲 4. Random Forests

Random Forests extend bagging by adding **feature subsampling**, which decorrelates trees. At each split, only $m_{\text{try}}$ features are considered.

Typical defaults:
- Classification: $m_{\text{try}} = \lfloor \sqrt{d} \rfloor$
- Regression: $m_{\text{try}} = \lfloor d / 3 \rfloor$

---

## 🧮 5. Majority Vote Error (Binary, i.i.d.)

If each base classifier has error $\varepsilon < \frac{1}{2}$, then the probability that the majority vote is wrong is:

$$
\Pr\{\text{vote wrong}\} = \sum_{k=\lceil (B+1)/2 \rceil}^{B} \binom{B}{k} \varepsilon^k (1 - \varepsilon)^{B-k}
$$

This probability decreases exponentially with $B$ when base learners are independent.

---

## 💻 6. Implementation Summary

This repository provides:
- **Classification demo**: Decision Tree, Bagging, and Random Forest on `make_moons` and Breast Cancer datasets.
- **Regression demo**: Decision Tree, Bagging, and Random Forest on the Diabetes dataset.
- **Visualization**: decision boundaries, confusion matrices, feature importances, and OOB performance.

---

## 🧰 7. Running the Code

### Clone and run
```bash
git clone https://github.com/yourusername/Bagging-Ensemble-Methods.git
cd Bagging-Ensemble-Methods
python main.py
```

The program will:
1. Train and visualize classifiers on synthetic 2D data.
2. Evaluate models on real-world datasets.
3. Plot feature importances and OOB metrics.

---

## 🧠 8. Key Equations Summary

$$
\hat{f}_{\text{bag}}(\mathbf{x}) = \frac{1}{B} \sum_{b=1}^{B} h^{(b)}(\mathbf{x})
$$

$$
\operatorname{Var}\left[ \frac{1}{B} \sum_{b=1}^{B} \hat{f}^{(b)} \right]
= \tau^2 \left( \rho + \frac{1 - \rho}{B} \right)
$$

$$
\Pr\{\text{OOB}\} \approx e^{-1} \approx 0.368
$$

---

## 🧾 9. License

This project is released under the **MIT License**.

---

_Last updated: 2025-10-14_
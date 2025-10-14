# 🧠 Linear AdaBoost Classifier — Theory, Implementation & Visualization

This repository contains a **complete educational implementation** of the **Linear AdaBoost Classifier**, including:

- Mathematical foundations of AdaBoost  
- From-scratch Python implementation (Decision Stumps as weak learners)  
- Visualizations of decision boundaries  
- Comparison with `sklearn.ensemble.AdaBoostClassifier`  
- Diagnostics and margin analysis  

---

## 📘 Theoretical Background

AdaBoost (Adaptive Boosting) combines many weak learners into a single strong classifier:

$$
F_{T}(\mathbf{x}) = \sum_{t=1}^{T} \alpha_{t} \, h_{t}(\mathbf{x})
$$

The final prediction is given by:

y-hat(x) = sign( F-T(x) )

AdaBoost minimizes the **exponential loss** over all training samples:

$$
\mathcal{L}(F) = \sum_{i=1}^{n} \exp\!\left(-y_{i} \, F(\mathbf{x}_{i})\right)
$$

---

### 🔹 Weighted Error and Learner Weight

At iteration $t$, given sample weights $D_{t}(i)$ such that $\sum_{i} D_{t}(i) = 1$, the **weighted classification error** is:

$$
\varepsilon_{t} = \sum_{i=1}^{n} D_{t}(i)\, \mathbb{1}\{y_{i} \neq h_{t}(\mathbf{x}_{i})\}
$$

The **learner weight** (influence of each weak learner) is computed as:

$$
\alpha_{t} = \tfrac{1}{2}\,\ln\!\left(\frac{1 - \varepsilon_{t}}{\varepsilon_{t}}\right)
$$

After each round, sample weights are updated:

$$
D_{t+1}(i) = \frac{ D_{t}(i)\, \exp\!\left(-\alpha_{t}\,y_{i}\,h_{t}(\mathbf{x}_{i})\right) }{ Z_{t} }
$$

where $Z_{t}$ is a normalization constant ensuring that $\sum_{i} D_{t+1}(i) = 1$.

---

### 🔹 Interpretation

- The model $F_{T}(\mathbf{x})$ is **linear in the weak learners** $h_{t}$.  
- Misclassified points get **higher weights**, forcing new learners to focus on them.  
- The **sign** of $F_{T}(\mathbf{x})$ gives the class label, and its magnitude corresponds to the **confidence margin**.  

---

## ⚙️ Implementation Overview

The core algorithm is implemented from scratch using **Decision Stumps** (1D threshold classifiers) as weak learners.

The repository also provides:

- Synthetic datasets (`make_classification`, `make_moons`)  
- Visualization of decision boundaries  
- Comparison with `AdaBoostClassifier` from scikit-learn (compatible with v1.6+)  
- Margin distribution histograms for model diagnostics  

---

## 🧩 Code Structure

```
├── main.py                         # Full training and visualization script
├── Linear_AdaBoost_Classifier.ipynb # Interactive Jupyter version
├── README.md                        # Project documentation (this file)
```

---

## 🧠 Usage

### 🔧 Run via Python script

```bash
python main.py
```

### 💻 Or interactively in Jupyter

```bash
jupyter notebook Linear_AdaBoost_Classifier.ipynb
```

---

## 🧮 Mathematical Summary

The iterative update mechanism of AdaBoost can be summarized as:

$$
\begin{aligned}
\text{Initialize:}& \quad D_{1}(i) = \frac{1}{n} \\
\text{For each round } t=1,\ldots,T:\\
&\quad h_{t} = \arg\min_{h}\, \varepsilon_{t} = \sum_{i} D_{t}(i) \,[y_{i} \neq h(\mathbf{x}_{i})] \\
&\quad \alpha_{t} = \frac{1}{2}\ln\!\left(\frac{1 - \varepsilon_{t}}{\varepsilon_{t}}\right) \\
&\quad D_{t+1}(i) = \frac{ D_{t}(i)\,\exp(-\alpha_{t}y_{i}h_{t}(\mathbf{x}_{i})) }{ Z_{t} } \\
&\quad F_{T}(\mathbf{x}) = \sum_{t=1}^{T} \alpha_{t} h_{t}(\mathbf{x})
\end{aligned}
$$

The final classifier is obtained as:

y_hat(x) = sign(F_T(x))

---

## 📚 References

- Freund, Y. & Schapire, R. E. (1997). *A Decision-Theoretic Generalization of On-Line Learning and an Application to Boosting.* JCSS.  
- Hastie, T., Tibshirani, R., Friedman, J. (2009). *The Elements of Statistical Learning*, 2nd ed. — Chapter 10.  
- Schapire, R. E. (2013). *Explaining AdaBoost.*

---

## 🧩 License

MIT License © 2025

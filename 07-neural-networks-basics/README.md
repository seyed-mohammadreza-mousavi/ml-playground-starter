# Neural Networks from Scratch (NumPy)

This repository contains a **fully self-contained Python implementation** of core neural network concepts — built entirely from scratch using only **NumPy** and **Matplotlib**.

It is designed for **students, researchers, and developers** who want to deeply understand the math, logic, and practical implementation of neural networks without relying on deep learning frameworks like TensorFlow or PyTorch.

---

## 📘 Features

### 🧠 Mathematical Foundations
- Derivations for logistic regression, MLP, and backpropagation
- Gradient formulas for sigmoid and softmax cross-entropy
- Step-by-step comments linking math to implementation

### ⚙️ Implementations
- **Logistic Regression** (binary classification)
- **Multi-Layer Perceptron (MLP)** with:
  - ReLU, Tanh, or Sigmoid activations
  - Backpropagation from scratch
  - L2 Regularization
  - Xavier / He Initialization

### 📊 Visualization
- 2D datasets: moons, circles, blobs
- Loss curve plotting
- Decision boundary visualization for trained models

---

## 🧩 File Structure

```
.
├── nn_basics.py     # main script with MLP and logistic regression
├── README.md        # this file
```

---

## 🚀 How to Run

### 1️⃣ Requirements

```bash
pip install numpy matplotlib
```

### 2️⃣ Run the Script

```bash
python nn_basics.py
```

This will:
- Train a logistic regression model on a toy dataset
- Train a two-layer MLP
- Plot loss curves and decision boundaries

---

## 🧮 Example Output

When you run the code, you’ll see:

- Logistic regression decision boundary (linear separator)
- MLP nonlinear decision boundary (curved regions)
- Training loss curves showing convergence

---

## 🧠 Mathematical Summary

### Logistic Regression
$$
z = w^\top x + b, \quad \hat{y} = \sigma(z) = \frac{1}{1 + e^{-z}}
$$

**Loss:**

$$
\ell = -\big[\,y \log(\hat{y}) + (1 - y)\log(1 - \hat{y})\,\big]
$$

**Gradients:**

$$
\frac{\partial \ell}{\partial z} = \hat{y} - y, \quad
\frac{\partial \ell}{\partial w} = (\hat{y} - y)x, \quad
\frac{\partial \ell}{\partial b} = \hat{y} - y
$$

---

**Backpropagation**

$$
h = f(u), \quad u = W x + b
$$

Upstream gradient:

$$
\delta = \frac{\partial \ell}{\partial h}
$$

Then:

$$
\frac{\partial \ell}{\partial u} = \delta \odot f'(u), \qquad
\frac{\partial \ell}{\partial W} = (\delta \odot f'(u))\,x^\top, \qquad
\frac{\partial \ell}{\partial b} = \delta \odot f'(u)
$$

---

### Softmax (Multi-Class)
$$
\mathrm{softmax}(z)_k = \frac{e^{z_k}}{\sum_j e^{z_j}}, \quad
\frac{\partial \ell}{\partial z_k} = \hat{y}_k - y_k
$$

---

## 🧑‍💻 Author
Created by **SeyedMohammadreza Mousavi | error.2013@yahoo.com** for educational use.  
You can freely modify and extend the code for your own experiments.

---

## 🪄 Future Ideas
- Add momentum and Adam optimizers
- Add dropout and batch normalization
- Extend to multi-class classification

---

## 📜 License
MIT License – use freely with attribution.

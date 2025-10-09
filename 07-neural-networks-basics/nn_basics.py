#!/usr/bin/env python3
"""
Neural Networks from Scratch: Math, Implementation, and Visualization
----------------------------------------------------------------------
Implements:
- Data generation utilities
- Logistic Regression
- Multi-Layer Perceptron (MLP) with backpropagation
- L2 regularization, Xavier/He initialization
- Visualizations for data, decision boundaries, and training curves

Author: SeyedMohammadreza Mousavi | error.2013@yahoo.com
"""

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------

def set_seed(seed=42):
    np.random.seed(seed)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_grad(z):
    s = sigmoid(z)
    return s * (1 - s)


# ---------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------

def make_moons(n_samples=600, noise=0.1):
    n = n_samples // 2
    theta = np.linspace(0, np.pi, n)
    r = 1.0
    x1 = np.c_[r * np.cos(theta), r * np.sin(theta)]
    x2 = np.c_[r * np.cos(theta) + 1.0, -r * np.sin(theta) + 0.25]
    X = np.vstack([x1, x2])
    y = np.hstack([np.zeros(n), np.ones(n)])
    X += noise * np.random.randn(*X.shape)
    return X, y.astype(int)


def train_val_split(X, y, val_ratio=0.2, seed=42):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(X))
    rng.shuffle(idx)
    split = int(len(X) * (1 - val_ratio))
    tr, va = idx[:split], idx[split:]
    return X[tr], y[tr], X[va], y[va]


# ---------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------

def plot_data(X, y, title="Dataset"):
    plt.figure(figsize=(5, 5))
    plt.scatter(X[:, 0], X[:, 1], c=y, s=12)
    plt.title(title)
    plt.axis("equal")
    plt.show()


def plot_decision_boundary(predict_fn, X, y, title="Decision Boundary", h=0.02):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = predict_fn(grid).reshape(xx.shape)
    plt.figure(figsize=(5, 5))
    plt.contourf(xx, yy, Z, alpha=0.5)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=12)
    plt.title(title)
    plt.axis("equal")
    plt.show()


# ---------------------------------------------------------------------
# Logistic Regression (from scratch)
# ---------------------------------------------------------------------

class LogisticRegression:
    """
    Simple logistic regression using gradient descent.
    """

    def __init__(self, input_dim, lr=0.1, reg=0.0):
        self.w = np.zeros(input_dim)
        self.b = 0.0
        self.lr = lr
        self.reg = reg

    def forward(self, X):
        return sigmoid(X @ self.w + self.b)

    def loss(self, y, yhat):
        eps = 1e-9
        return -np.mean(y * np.log(yhat + eps) + (1 - y) * np.log(1 - yhat + eps))

    def train(self, X, y, steps=2000):
        losses = []
        for t in range(steps):
            yhat = self.forward(X)
            grad_w = (X.T @ (yhat - y)) / len(X) + self.reg * self.w
            grad_b = np.mean(yhat - y)
            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b
            losses.append(self.loss(y, yhat))
        return np.array(losses)

    def predict(self, X):
        return (self.forward(X) > 0.5).astype(int)


# ---------------------------------------------------------------------
# Multi-Layer Perceptron
# ---------------------------------------------------------------------

class MLP:
    def __init__(self, in_dim, h1=32, h2=32, out_dim=1, act="relu", seed=42, init="xavier"):
        rng = np.random.default_rng(seed)
        self.act_name = act

        def init_w(m, n, kind="xavier"):
            if kind == "xavier":
                scale = np.sqrt(2.0 / (m + n))
            elif kind == "he":
                scale = np.sqrt(2.0 / m)
            else:
                scale = 0.01
            return rng.normal(0.0, scale, size=(m, n))

        self.W1 = init_w(h1, in_dim, init)
        self.b1 = np.zeros((h1,))
        self.W2 = init_w(h2, h1, init)
        self.b2 = np.zeros((h2,))
        self.W3 = init_w(out_dim, h2, init)
        self.b3 = np.zeros((out_dim,))

    # Activation
    def act(self, u):
        if self.act_name == "relu":
            return np.maximum(0, u)
        elif self.act_name == "tanh":
            return np.tanh(u)
        else:
            return sigmoid(u)

    def act_grad(self, u):
        if self.act_name == "relu":
            return (u > 0).astype(float)
        elif self.act_name == "tanh":
            return 1 - np.tanh(u) ** 2
        else:
            return sigmoid_grad(u)

    def forward(self, X):
        u1 = X @ self.W1.T + self.b1
        h1 = self.act(u1)
        u2 = h1 @ self.W2.T + self.b2
        h2 = self.act(u2)
        z = h2 @ self.W3.T + self.b3
        yhat = sigmoid(z)
        cache = (X, u1, h1, u2, h2, z, yhat)
        return yhat, cache

    def loss(self, y, yhat, l2=0.0):
        eps = 1e-9
        ce = -np.mean(y * np.log(yhat + eps) + (1 - y) * np.log(1 - yhat + eps))
        reg = 0.5 * l2 * (np.sum(self.W1 ** 2) + np.sum(self.W2 ** 2) + np.sum(self.W3 ** 2))
        return ce + reg

    def backward(self, cache, y, l2=0.0):
        X, u1, h1, u2, h2, z, yhat = cache
        n = X.shape[0]
        delta3 = (yhat - y.reshape(-1, 1)) / n
        gW3 = delta3.T @ h2 + l2 * self.W3
        gb3 = delta3.sum(axis=0)
        dh2 = delta3 @ self.W3
        du2 = dh2 * self.act_grad(u2)
        gW2 = du2.T @ h1 + l2 * self.W2
        gb2 = du2.sum(axis=0)
        dh1 = du2 @ self.W2
        du1 = dh1 * self.act_grad(u1)
        gW1 = du1.T @ X + l2 * self.W1
        gb1 = du1.sum(axis=0)
        return gW1, gb1, gW2, gb2, gW3, gb3

    def step(self, grads, lr=0.01):
        gW1, gb1, gW2, gb2, gW3, gb3 = grads
        self.W1 -= lr * gW1
        self.b1 -= lr * gb1
        self.W2 -= lr * gW2
        self.b2 -= lr * gb2
        self.W3 -= lr * gW3
        self.b3 -= lr * gb3


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    set_seed(0)
    X, y = make_moons(n_samples=600, noise=0.15)
    Xtr, ytr, Xva, yva = train_val_split(X, y)

    print("Training Logistic Regression...")
    logreg = LogisticRegression(input_dim=2, lr=0.3, reg=1e-3)
    losses = logreg.train(Xtr, ytr, steps=1500)
    plt.plot(losses)
    plt.title("Logistic Regression Loss")
    plt.show()
    plot_decision_boundary(logreg.predict, Xva, yva, title="Logistic Regression Decision Boundary")

    print("Training MLP (ReLU, He init)...")
    net = MLP(in_dim=2, h1=32, h2=32, act="relu", init="he")
    steps, lr, l2 = 3000, 0.05, 1e-3
    losses = []
    for t in range(steps):
        idx = np.random.choice(len(Xtr), size=128, replace=False)
        Xb, yb = Xtr[idx], ytr[idx]
        yhat, cache = net.forward(Xb)
        loss = net.loss(yb, yhat, l2=l2)
        grads = net.backward(cache, yb, l2=l2)
        net.step(grads, lr=lr)
        if (t + 1) % 100 == 0:
            losses.append(loss)
    plt.plot(losses)
    plt.title("MLP Training Loss")
    plt.show()

    def predict_mlp(Z):
        yhat, _ = net.forward(Z)
        return (yhat.ravel() > 0.5).astype(int)

    plot_decision_boundary(predict_mlp, Xva, yva, title="MLP Decision Boundary")
    print("Done ✅")


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------

if __name__ == "__main__":
    main()

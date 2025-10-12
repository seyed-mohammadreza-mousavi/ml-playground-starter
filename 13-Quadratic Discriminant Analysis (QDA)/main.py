
import numpy as np
from numpy.linalg import slogdet, inv
from dataclasses import dataclass
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

@dataclass
class QDA:
    reg_lambda: float = 0.0
    random_state: int = 42

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        n, d = X.shape
        self.means_, self.covs_, self.priors_ = {}, {}, {}
        for c in self.classes_:
            Xc = X[y == c]
            nc = Xc.shape[0]
            mu = Xc.mean(axis=0)
            diff = Xc - mu
            Sigma = (diff.T @ diff) / nc
            if self.reg_lambda > 0.0:
                alpha = np.trace(Sigma) / Sigma.shape[0]
                Sigma = (1 - self.reg_lambda) * Sigma + self.reg_lambda * alpha * np.eye(d)
            self.means_[c], self.covs_[c], self.priors_[c] = mu, Sigma, nc / n
        return self

    def _log_gaussian(self, X, mu, Sigma):
        X = np.atleast_2d(X)
        d = X.shape[1]
        sign, logdet = slogdet(Sigma)
        if sign <= 0:
            Sigma = Sigma + 1e-9 * np.eye(d)
            sign, logdet = slogdet(Sigma)
        invS = inv(Sigma)
        diff = X - mu
        quad = np.sum(diff @ invS * diff, axis=1)
        return -0.5 * (logdet + quad)

    def decision_function(self, X):
        X = np.asarray(X, dtype=float)
        scores = np.zeros((X.shape[0], len(self.classes_)))
        for j, c in enumerate(self.classes_):
            mu, Sigma = self.means_[c], self.covs_[c]
            scores[:, j] = self._log_gaussian(X, mu, Sigma) + np.log(self.priors_[c])
        return scores

    def predict(self, X):
        scores = self.decision_function(X)
        return self.classes_[np.argmax(scores, axis=1)]

    def predict_proba(self, X):
        scores = self.decision_function(X)
        scores -= scores.max(axis=1, keepdims=True)
        exp_s = np.exp(scores)
        return exp_s / exp_s.sum(axis=1, keepdims=True)


def main():
    # Generate 2D dataset
    X, y = make_classification(
        n_samples=600, n_features=2, n_redundant=0, n_informative=2,
        n_clusters_per_class=1, class_sep=1.2, random_state=7
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=7, stratify=y)

    # From-scratch QDA
    qda = QDA(reg_lambda=0.05).fit(X_train, y_train)
    y_pred = qda.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("From-scratch QDA accuracy:", acc)
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    print("Report:\n", classification_report(y_test, y_pred))

    # Decision boundary
    x_min, x_max = X[:,0].min() - 1.0, X[:,0].max() + 1.0
    y_min, y_max = X[:,1].min() - 1.0, X[:,1].max() + 1.0
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = qda.predict(grid).reshape(xx.shape)
    plt.figure()
    plt.contourf(xx, yy, Z, alpha=0.25)
    plt.scatter(X_train[:,0], X_train[:,1], s=14, label="train")
    plt.scatter(X_test[:,0], X_test[:,1], s=14, marker="x", label="test")
    plt.title("QDA Decision Regions (from scratch)")
    plt.xlabel("x1"); plt.ylabel("x2"); plt.legend(); plt.show()

    # scikit-learn comparison
    sk_qda = QuadraticDiscriminantAnalysis(reg_param=0.05)
    sk_qda.fit(X_train, y_train)
    y_pred_sk = sk_qda.predict(X_test)
    acc_sk = accuracy_score(y_test, y_pred_sk)
    print("scikit-learn QDA accuracy:", acc_sk)

    # CV comparison
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
    pipe_std = make_pipeline(StandardScaler(), QuadraticDiscriminantAnalysis(reg_param=0.05))
    pipe_raw = QuadraticDiscriminantAnalysis(reg_param=0.05)
    scores_std = cross_val_score(pipe_std, X, y, cv=cv)
    scores_raw = cross_val_score(pipe_raw, X, y, cv=cv)
    print("CV accuracy with StandardScaler:", scores_std.mean(), "±", scores_std.std())
    print("CV accuracy raw:", scores_raw.mean(), "±", scores_raw.std())


if __name__ == "__main__":
    main()

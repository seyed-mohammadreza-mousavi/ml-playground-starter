
"""
Gaussian Mixture Models (GMM) from Scratch + Visualization
----------------------------------------------------------
This script demonstrates:
- GMM mathematical formulation (in comments)
- From-scratch EM implementation (NumPy)
- Visualization in 1D and 2D
- Comparison with sklearn GaussianMixture
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score

# =============================================================
# Utility: Log multivariate normal
# =============================================================
def log_multivariate_normal(x, mean, cov):
    D = mean.shape[0]
    xc = x - mean
    try:
        L = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        jitter = 1e-6 * np.eye(D)
        L = np.linalg.cholesky(cov + jitter)
        cov = cov + jitter
    logdet = 2.0 * np.sum(np.log(np.diag(L)))
    sol = np.linalg.solve(L, xc)
    return -0.5 * (D*np.log(2*np.pi) + logdet + sol.T @ sol)


# =============================================================
# Gaussian Mixture Model (from scratch)
# =============================================================
class GMMFromScratch:
    def __init__(self, n_components=3, max_iter=200, tol=1e-4, reg_covar=1e-6, init='kmeans', random_state=42):
        self.K = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.reg_covar = reg_covar
        self.init = init
        self.random_state = random_state
        self.weights_ = None
        self.means_ = None
        self.covariances_ = None
        self.log_likelihoods_ = []

    def _initialize_parameters(self, X):
        N, D = X.shape
        rng = np.random.default_rng(self.random_state)
        if self.init == 'kmeans':
            km = KMeans(n_clusters=self.K, n_init=10, random_state=self.random_state).fit(X)
            means = km.cluster_centers_
        else:
            idx = rng.choice(N, size=self.K, replace=False)
            means = X[idx]
        cov = np.cov(X.T)
        covariances = np.array([cov + self.reg_covar*np.eye(D) for _ in range(self.K)])
        weights = np.ones(self.K) / self.K
        self.means_ = means
        self.covariances_ = covariances
        self.weights_ = weights

    def _e_step(self, X):
        N, D = X.shape
        log_resp = np.zeros((N, self.K))
        for k in range(self.K):
            for n in range(N):
                log_resp[n, k] = np.log(self.weights_[k] + 1e-12) + log_multivariate_normal(X[n], self.means_[k], self.covariances_[k])
        max_log = np.max(log_resp, axis=1, keepdims=True)
        log_resp -= max_log
        resp = np.exp(log_resp)
        resp /= np.sum(resp, axis=1, keepdims=True)
        log_likelihood = np.sum(max_log + np.log(np.sum(np.exp(log_resp), axis=1, keepdims=True)))
        return resp, log_likelihood

    def _m_step(self, X, resp):
        N, D = X.shape
        Nk = np.sum(resp, axis=0) + 1e-12
        weights = Nk / N
        means = (resp.T @ X) / Nk[:, None]
        covariances = np.zeros((self.K, D, D))
        for k in range(self.K):
            xc = X - means[k]
            cov = (resp[:, k][:, None] * xc).T @ xc / Nk[k]
            cov += self.reg_covar * np.eye(D)
            covariances[k] = cov
        self.weights_ = weights
        self.means_ = means
        self.covariances_ = covariances

    def fit(self, X):
        self._initialize_parameters(X)
        prev_ll = -np.inf
        for it in range(self.max_iter):
            resp, ll = self._e_step(X)
            self._m_step(X, resp)
            self.log_likelihoods_.append(ll)
            if it > 0 and np.abs(ll - prev_ll) < self.tol:
                break
            prev_ll = ll
        return self

    def predict_proba(self, X):
        N = X.shape[0]
        log_resp = np.zeros((N, self.K))
        for k in range(self.K):
            for n in range(N):
                log_resp[n, k] = np.log(self.weights_[k] + 1e-12) + log_multivariate_normal(X[n], self.means_[k], self.covariances_[k])
        max_log = np.max(log_resp, axis=1, keepdims=True)
        resp = np.exp(log_resp - max_log)
        resp /= np.sum(resp, axis=1, keepdims=True)
        return resp

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def score(self, X):
        N = X.shape[0]
        log_probs = []
        for n in range(N):
            s = 0.0
            for k in range(self.K):
                s += self.weights_[k] * np.exp(log_multivariate_normal(X[n], self.means_[k], self.covariances_[k]))
            log_probs.append(np.log(s + 1e-12))
        return np.mean(log_probs)

    def bic(self, X):
        N, D = X.shape
        n_params = (self.K - 1) + self.K * D + self.K * (D * (D + 1) // 2)
        ll = self.score(X) * N
        return n_params * np.log(N) - 2 * ll

    def aic(self, X):
        N, D = X.shape
        n_params = (self.K - 1) + self.K * D + self.K * (D * (D + 1) // 2)
        ll = self.score(X) * N
        return 2 * n_params - 2 * ll


def mixture_pdf_grid(gmm, X, grid_size=150, padding=2.5):
    x_min, x_max = X[:,0].min()-padding, X[:,0].max()+padding
    y_min, y_max = X[:,1].min()-padding, X[:,1].max()+padding
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_size),
                         np.linspace(y_min, y_max, grid_size))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = np.zeros(grid.shape[0])
    for i, pt in enumerate(grid):
        s = 0.0
        for k in range(gmm.K):
            s += gmm.weights_[k]*np.exp(log_multivariate_normal(pt, gmm.means_[k], gmm.covariances_[k]))
        Z[i] = s
    return xx, yy, Z.reshape(xx.shape)


def plot_log_likelihood_curve(gmm):
    plt.figure()
    plt.plot(gmm.log_likelihoods_)
    plt.title("EM Log-Likelihood Progress")
    plt.xlabel("Iteration")
    plt.ylabel("Log-Likelihood")
    plt.show()


def main():
    np.random.seed(42)

    # 2D example
    X, _ = make_blobs(n_samples=900, centers=3, cluster_std=[1.0, 1.5, 0.8], random_state=42, n_features=2)
    gmm = GMMFromScratch(n_components=3, max_iter=300, tol=1e-5, reg_covar=1e-6, init='kmeans', random_state=42)
    gmm.fit(X)

    print("Weights:", np.round(gmm.weights_, 3))
    print("Means:\\n", np.round(gmm.means_, 3))
    print("BIC:", gmm.bic(X))
    print("AIC:", gmm.aic(X))

    xx, yy, Z = mixture_pdf_grid(gmm, X)
    plt.figure()
    plt.contour(xx, yy, Z, levels=10)
    plt.scatter(X[:,0], X[:,1], s=8, alpha=0.6)
    plt.title("GMM (from scratch) – 2D mixture contours")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.show()

    plot_log_likelihood_curve(gmm)

    # sklearn comparison
    sk_gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=42, n_init=5)
    sk_gmm.fit(X)
    sil = silhouette_score(X, sk_gmm.predict(X))
    print("sklearn weights:", np.round(sk_gmm.weights_, 3))
    print("sklearn means:\\n", np.round(sk_gmm.means_, 3))
    print("Silhouette score (sklearn labels):", round(sil, 4))

    # 1D example
    rng = np.random.default_rng(123)
    N = 1000
    mix = rng.choice([0,1], size=N, p=[0.4, 0.6])
    x = np.zeros(N)
    x[mix==0] = rng.normal(loc=-2.5, scale=0.7, size=np.sum(mix==0))
    x[mix==1] = rng.normal(loc=1.5, scale=0.5, size=np.sum(mix==1))
    X1 = x.reshape(-1,1)

    gmm1d = GMMFromScratch(n_components=2, max_iter=300, tol=1e-6, random_state=0)
    gmm1d.fit(X1)

    xs = np.linspace(x.min()-2, x.max()+2, 400).reshape(-1,1)
    pdf_vals = []
    for xi in xs:
        s = 0.0
        for k in range(gmm1d.K):
            s += gmm1d.weights_[k]*np.exp(log_multivariate_normal(xi, gmm1d.means_[k], gmm1d.covariances_[k]))
        pdf_vals.append(s)
    pdf_vals = np.array(pdf_vals)

    plt.figure()
    plt.hist(x, bins=40, density=True, alpha=0.6)
    plt.plot(xs, pdf_vals, color='r')
    plt.title("1D GMM fit – histogram and fitted pdf")
    plt.xlabel("x")
    plt.ylabel("density")
    plt.show()

    print("1D Weights:", np.round(gmm1d.weights_, 3))
    print("1D Means:", np.round(gmm1d.means_.ravel(), 3))
    print("1D Variances:", [np.round(np.diag(cov)[0], 3) for cov in gmm1d.covariances_])


if __name__ == "__main__":
    main()
